"""Feed-forward reconstruction — the GPU stage.

Default engine is **MASt3R** (the more accurate successor to DUSt3R); DUSt3R is
available as a fallback via ``backend="dust3r"``. Both predict a 3D point per
pixel and recover the camera poses with no COLMAP and no provided calibration;
we then fuse the per-frame pointmaps into one coloured cloud and tidy it.

Requires a CUDA GPU and the MASt3R repo cloned + importable (the Colab notebook's
setup cell does this)::

    !git clone --recursive https://github.com/naver/mast3r
    !pip install -r mast3r/requirements.txt
    !pip install -r mast3r/dust3r/requirements.txt
    import sys; sys.path += ["mast3r", "mast3r/dust3r"]
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DUST3R_MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
MAST3R_MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

# The two engines use different confidence scales, so default the threshold per backend.
_DEFAULT_MIN_CONF = {"mast3r": 1.5, "dust3r": 3.0}


@dataclass
class FrameResult:
    """Per-frame output, kept so semantics can later lift 2D labels -> 3D."""
    image_path: str
    rgb: np.ndarray      # (H, W, 3) uint8
    pts3d: np.ndarray    # (H, W, 3) float — a 3D point per pixel
    mask: np.ndarray     # (H, W) bool   — points kept after confidence filtering


@dataclass
class Reconstruction:
    """The fused point cloud plus per-frame data for downstream semantics."""
    points: np.ndarray   # (N, 3) float
    colors: np.ndarray   # (N, 3) uint8
    frames: list = field(default_factory=list)  # list[FrameResult]


def clean_cloud(points, colors, keep_percentile=99.0, max_points=500_000, seed=0):
    """Drop non-finite points + far-flung flyers, and cap the cloud size.

    A cheap heuristic trim (finite check + radial-percentile cut from the median
    centre + a random cap) — not full statistical denoising, but enough to make
    the output clean and fast to view. Returns (points, colors).
    """
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    if len(points) == 0:
        raise ValueError("No valid 3D points after filtering.")

    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    keep = radius <= np.percentile(radius, keep_percentile)
    points, colors = points[keep], colors[keep]

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]

    return points, colors


# ---------------------------------------------------------------------------
# Lazy imports (heavy deps only load when actually reconstructing)
# ---------------------------------------------------------------------------
def _import_mast3r():
    try:
        from mast3r.model import AsymmetricMASt3R
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        import mast3r.utils.path_to_dust3r  # noqa: F401  (puts the dust3r submodule on path)
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MASt3R is not importable. In Colab, run the setup cell:\n"
            "  !git clone --recursive https://github.com/naver/mast3r\n"
            "  !pip install -r mast3r/requirements.txt\n"
            "  !pip install -r mast3r/dust3r/requirements.txt\n"
            "  import sys; sys.path += ['mast3r', 'mast3r/dust3r']\n"
            f"(original error: {exc})"
        ) from exc
    return AsymmetricMASt3R, sparse_global_alignment, load_images, make_pairs


def _import_dust3r():
    try:
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "DUSt3R is not importable. It ships as a submodule of MASt3R; make "
            "sure the setup cell ran and 'mast3r/dust3r' is on sys.path.\n"
            f"(original error: {exc})"
        ) from exc
    return (inference, AsymmetricCroCo3DStereo, load_images, make_pairs,
            global_aligner, GlobalAlignerMode)


def _scene_graph_for(n):
    # Complete pair graph is most accurate but grows O(n^2); fall back to a
    # sliding window above 25 images to stay inside GPU memory.
    return "complete" if n <= 25 else "swin"


# ---------------------------------------------------------------------------
# Per-backend reconstruction. Each returns a list of
#   (rgb (H,W,3) uint8, pts3d (H,W,3) float, conf (H,W) float)
# ---------------------------------------------------------------------------
def _run_mast3r(image_paths, device, image_size):
    AsymmetricMASt3R, sparse_global_alignment, load_images, make_pairs = _import_mast3r()

    print(f"Loading MASt3R ({MAST3R_MODEL})...")
    model = AsymmetricMASt3R.from_pretrained(MAST3R_MODEL).to(device)

    images = load_images(image_paths, size=image_size)
    scene_graph = _scene_graph_for(len(images))
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"MASt3R sparse global alignment on {len(pairs)} pairs ({scene_graph} graph)...")

    cache_dir = tempfile.mkdtemp(prefix="mast3r_cache_")
    scene = sparse_global_alignment(
        image_paths, pairs, cache_dir, model,
        lr1=0.07, niter1=500, lr2=0.014, niter2=200,
        device=device, opt_depth=True, shared_intrinsics=False,
        matching_conf_thr=5.0,
    )

    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)
    imgs = scene.imgs

    per_frame = []
    for img, pts, conf in zip(imgs, pts3d, confs):
        img = np.asarray(img)
        h, w = img.shape[:2]
        pts_np = pts.detach().cpu().numpy().reshape(h, w, 3)
        conf_np = conf.detach().cpu().numpy().reshape(h, w)
        rgb_np = (img * 255).astype(np.uint8)
        per_frame.append((rgb_np, pts_np, conf_np))
    return per_frame


def _run_dust3r(image_paths, device, image_size, niter):
    (inference, AsymmetricCroCo3DStereo, load_images, make_pairs,
     global_aligner, GlobalAlignerMode) = _import_dust3r()

    print(f"Loading DUSt3R ({DUST3R_MODEL})...")
    model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_MODEL).to(device)

    images = load_images(image_paths, size=image_size)
    scene_graph = _scene_graph_for(len(images))
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"DUSt3R inference on {len(pairs)} pairs ({scene_graph} graph)...")
    output = inference(pairs, model, device, batch_size=1)

    print("Global alignment (recovering poses + fusing pointmaps)...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)

    imgs = scene.imgs
    pts3d = scene.get_pts3d()
    confs = scene.get_conf()

    per_frame = []
    for img, pts, conf in zip(imgs, pts3d, confs):
        rgb_np = (np.asarray(img) * 255).astype(np.uint8)
        pts_np = pts.detach().cpu().numpy()      # (H, W, 3)
        conf_np = conf.detach().cpu().numpy()    # (H, W)
        per_frame.append((rgb_np, pts_np, conf_np))
    return per_frame


# ---------------------------------------------------------------------------
# Fuse + clean
# ---------------------------------------------------------------------------
def _fuse(per_frame, image_paths, min_conf):
    all_pts, all_cols, frames = [], [], []
    for path, (rgb_np, pts_np, conf_np) in zip(image_paths, per_frame):
        # Keep confident, finite pixels — this is what drops the low-confidence "haze".
        mask_np = np.isfinite(pts_np).all(axis=2) & (conf_np > min_conf)
        all_pts.append(pts_np[mask_np])
        all_cols.append(rgb_np[mask_np])
        frames.append(FrameResult(image_path=path, rgb=rgb_np, pts3d=pts_np, mask=mask_np))

    raw_points = np.concatenate(all_pts, axis=0)
    raw_colors = np.concatenate(all_cols, axis=0)
    points, colors = clean_cloud(raw_points, raw_colors)
    print(f"Reconstructed {len(points):,} points "
          f"(from {len(raw_points):,} raw) across {len(image_paths)} frames.")
    return Reconstruction(points=points, colors=colors, frames=frames)


def run(images_dir, backend="mast3r", device="cuda", image_size=512,
        niter=300, min_conf=None):
    """Reconstruct a coloured point cloud from the images in `images_dir`.

    Args:
        images_dir: folder of frame_*.jpg (prepare it with v3d.frames).
        backend:    "mast3r" (default, most accurate) or "dust3r" (fallback).
        device:     "cuda" (required for real use) or "cpu" (very slow).
        image_size: 512 or 224 — the resolution the model runs at.
        niter:      DUSt3R global-alignment iterations (ignored by MASt3R).
        min_conf:   confidence cutoff; None -> a sensible per-backend default.

    Returns:
        Reconstruction(points, colors, frames).
    """
    backend = backend.lower()
    image_paths = [str(p) for p in sorted(Path(images_dir).glob("frame_*.jpg"))]
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(image_paths)} in {images_dir}.")
    if min_conf is None:
        min_conf = _DEFAULT_MIN_CONF.get(backend, 3.0)

    if backend == "mast3r":
        per_frame = _run_mast3r(image_paths, device, image_size)
    elif backend == "dust3r":
        per_frame = _run_dust3r(image_paths, device, image_size, niter)
    else:
        raise ValueError(f"Unknown backend {backend!r} (use 'mast3r' or 'dust3r').")

    return _fuse(per_frame, image_paths, min_conf)
