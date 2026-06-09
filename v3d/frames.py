"""Frame extraction + blur filtering — the CPU-only prep stage.

Needs only ffmpeg (on PATH) and OpenCV; no GPU. Turns a video into a clean set
of sharp, non-redundant frames sized to fit a GPU's memory for reconstruction.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

# COLMAP/DUSt3R need a handful of overlapping views to triangulate anything.
MIN_FRAMES = 10


def extract(video_path, images_dir, fps=2):
    """Split a video into still JPEGs at `fps` frames/second using ffmpeg.

    Returns the sorted list of extracted frame paths.
    """
    video_path, images_dir = Path(video_path), Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Clear frames from a previous run so stale images can't leak in.
    for old in images_dir.glob("frame_*.jpg"):
        old.unlink()

    pattern = str(images_dir / "frame_%04d.jpg")  # frame_0001.jpg, frame_0002.jpg, ...
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",            # high-quality JPEGs help feature matching
        pattern,
    ]
    subprocess.run(cmd, check=True)

    frames = sorted(images_dir.glob("frame_*.jpg"))
    print(f"Extracted {len(frames)} frames.")
    if len(frames) < MIN_FRAMES:
        print("WARNING: very few frames — video may be too short, or fps too low.")
    return frames


def variance_of_laplacian(gray):
    """Sharpness score: higher variance of the Laplacian == sharper image."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def filter_blurry(images_dir, min_frames=MIN_FRAMES):
    """Drop frames well below the median sharpness, never going under min_frames.

    Returns the sorted list of frames that remain.
    """
    images_dir = Path(images_dir)
    frames = sorted(images_dir.glob("frame_*.jpg"))
    if len(frames) <= min_frames:
        print(f"Skipping blur filter ({len(frames)} frames — at/below safe minimum).")
        return frames

    scores = {}
    for frame in frames:
        gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        scores[frame] = variance_of_laplacian(gray)
    if not scores:
        print("Skipping blur filter (could not read any frames).")
        return frames

    threshold = float(np.median(list(scores.values()))) * 0.5
    # Candidates below threshold, blurriest first.
    candidates = sorted((f for f, s in scores.items() if s < threshold),
                        key=lambda f: scores[f])
    # Safety: never delete so many that fewer than min_frames remain.
    max_removable = max(0, len(scores) - min_frames)
    for frame in candidates[:max_removable]:
        frame.unlink()

    kept = sorted(images_dir.glob("frame_*.jpg"))
    print(f"Removed {len(frames) - len(kept)} blurry frames, kept {len(kept)}.")
    return kept


def subsample(images_dir, max_frames=20):
    """Uniformly thin the frames to at most `max_frames`.

    DUSt3R/VGGT process frames jointly, so GPU memory scales with frame count.
    A few dozen well-spread frames reconstruct a small room fine; this keeps us
    inside a free Colab T4's budget. Returns the frames that remain.
    """
    images_dir = Path(images_dir)
    frames = sorted(images_dir.glob("frame_*.jpg"))
    if len(frames) <= max_frames:
        return frames

    keep_idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    keep = {frames[i] for i in keep_idx}
    for frame in frames:
        if frame not in keep:
            frame.unlink()

    kept = sorted(images_dir.glob("frame_*.jpg"))
    print(f"Subsampled to {len(kept)} frames (from {len(frames)}).")
    return kept
