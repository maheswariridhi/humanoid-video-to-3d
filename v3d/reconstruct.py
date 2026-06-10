"""Feed-forward reconstruction — the GPU stage.

Uses **MASt3R**: given a folder of images it predicts a 3D point per pixel,
recovers the camera poses, and fuses everything into one coloured point cloud —
no COLMAP and no provided calibration. We then tidy the fused cloud.

Requires a CUDA GPU and the MASt3R repo cloned + importable (the Colab notebook's
setup cell does this; MASt3R ships DUSt3R as a submodule that it imports)::

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

MAST3R_MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DEFAULT_MIN_CONF = 1.5


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


def run(images_dir, device="cuda", image_size=512, min_conf=DEFAULT_MIN_CONF):
    """Reconstruct a coloured point cloud from the images in `images_dir`.

    Args:
        images_dir: folder of frame_*.jpg (prepare it with v3d.frames).
        device:     "cuda" (required for real use) or "cpu" (very slow).
        image_size: 512 or 224 — the resolution MASt3R runs at.
        min_conf:   confidence cutoff; raise for fewer but cleaner points.

    Returns:
        Reconstruction(points, colors, frames).
    """
    AsymmetricMASt3R, sparse_global_alignment, load_images, make_pairs = _import_mast3r()

    image_paths = [str(p) for p in sorted(Path(images_dir).glob("frame_*.jpg"))]
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(image_paths)} in {images_dir}.")

    print(f"Loading MASt3R ({MAST3R_MODEL})...")
    model = AsymmetricMASt3R.from_pretrained(MAST3R_MODEL).to(device)

    images = load_images(image_paths, size=image_size)
    # Complete pair graph is most accurate but grows O(n^2); fall back to a
    # sliding window above 25 images to stay inside GPU memory.
    scene_graph = "complete" if len(images) <= 25 else "swin"
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

    all_pts, all_cols, frames = [], [], []
    for path, img, pts, conf in zip(image_paths, imgs, pts3d, confs):
        img = np.asarray(img)
        h, w = img.shape[:2]
        pts_np = pts.detach().cpu().numpy().reshape(h, w, 3)
        conf_np = conf.detach().cpu().numpy().reshape(h, w)
        rgb_np = (img * 255).astype(np.uint8)
        # Keep confident, finite pixels — drops the low-confidence "haze".
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
