"""DUSt3R reconstruction — the GPU stage.

Wraps NAVER's DUSt3R: given a folder of images it predicts a 3D point for every
pixel ("pointmaps"), recovers the camera poses, and fuses everything into one
coloured point cloud. No COLMAP and no camera calibration required, and it stays
robust on the low-texture / low-parallax shots where classical SfM falls over.

Requires:
  * a CUDA GPU (Colab's free T4 is enough for a few dozen frames), and
  * the `dust3r` repo cloned and importable — the Colab notebook's setup cell
    does this with:
        !git clone --recursive https://github.com/naver/dust3r
        !pip install -r dust3r/requirements.txt
        import sys; sys.path.append('dust3r')
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"


@dataclass
class FrameResult:
    """Per-frame reconstruction output, kept so semantics can lift 2D -> 3D."""
    image_path: str
    rgb: np.ndarray      # (H, W, 3) uint8  — the image as DUSt3R saw it
    pts3d: np.ndarray    # (H, W, 3) float  — a 3D point per pixel
    mask: np.ndarray     # (H, W) bool      — confident pixels


@dataclass
class Reconstruction:
    """The fused point cloud plus per-frame data for downstream semantics."""
    points: np.ndarray   # (N, 3) float
    colors: np.ndarray   # (N, 3) uint8
    frames: list = field(default_factory=list)  # list[FrameResult]


def clean_cloud(points, colors, keep_percentile=99.5, max_points=500_000, seed=0):
    """Tidy a raw fused cloud: drop non-finite points + far-flung flyers, and cap size.

    DUSt3R fuses a point per confident pixel, which leaves some NaN/Inf entries
    and a few outliers floating far from the scene. This is a cheap heuristic
    trim (finite check + radial percentile from the median centre + a random cap)
    — not full statistical denoising, but enough to make the output clean and fast
    to view. Returns (points, colors).
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


def _import_dust3r():
    try:
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "DUSt3R is not importable. In Colab, run the setup cell:\n"
            "  !git clone --recursive https://github.com/naver/dust3r\n"
            "  !pip install -r dust3r/requirements.txt\n"
            "  import sys; sys.path.append('dust3r')\n"
            f"(original error: {exc})"
        ) from exc
    return (inference, AsymmetricCroCo3DStereo, load_images, make_pairs,
            global_aligner, GlobalAlignerMode)


def run(images_dir, model_name=DEFAULT_MODEL, device="cuda",
        image_size=512, niter=300):
    """Reconstruct a coloured point cloud from the images in `images_dir`.

    Args:
        images_dir: folder of frame_*.jpg (use v3d.frames to prepare it).
        model_name: DUSt3R checkpoint on the HuggingFace hub.
        device:     "cuda" (required for real use) or "cpu" (very slow).
        image_size: 512 or 224 — the resolution DUSt3R runs at.
        niter:      global-alignment optimisation iterations.

    Returns:
        Reconstruction(points, colors, frames).
    """
    (inference, AsymmetricCroCo3DStereo, load_images, make_pairs,
     global_aligner, GlobalAlignerMode) = _import_dust3r()

    image_paths = [str(p) for p in sorted(Path(images_dir).glob("frame_*.jpg"))]
    if len(image_paths) < 2:
        raise ValueError(
            f"Need at least 2 images, found {len(image_paths)} in {images_dir}."
        )

    print(f"Loading DUSt3R ({model_name})...")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    images = load_images(image_paths, size=image_size)
    # A complete pair graph is most accurate but grows as O(n^2); fall back to a
    # sliding window for larger sets so we stay inside GPU memory.
    scene_graph = "complete" if len(images) <= 25 else "swin"
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"Inference on {len(pairs)} image pairs ({scene_graph} graph)...")
    output = inference(pairs, model, device, batch_size=1)

    print("Global alignment (recovering poses + fusing pointmaps)...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)

    imgs = scene.imgs               # list of (H, W, 3) float in [0, 1]
    pts3d = scene.get_pts3d()       # list of (H, W, 3) tensors
    masks = scene.get_masks()       # list of (H, W) bool tensors

    all_pts, all_cols, frames = [], [], []
    for path, img, pts, mask in zip(image_paths, imgs, pts3d, masks):
        pts_np = pts.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        rgb_np = (np.asarray(img) * 255).astype(np.uint8)
        all_pts.append(pts_np[mask_np])
        all_cols.append(rgb_np[mask_np])
        frames.append(FrameResult(image_path=path, rgb=rgb_np, pts3d=pts_np, mask=mask_np))

    raw_points = np.concatenate(all_pts, axis=0)
    raw_colors = np.concatenate(all_cols, axis=0)
    points, colors = clean_cloud(raw_points, raw_colors)
    print(f"Reconstructed {len(points):,} points "
          f"(from {len(raw_points):,} raw) across {len(image_paths)} frames.")
    return Reconstruction(points=points, colors=colors, frames=frames)
