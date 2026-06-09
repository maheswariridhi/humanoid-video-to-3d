#!/usr/bin/env python3
"""
reconstruct_colmap.py — Classical baseline: phone video -> 3D point cloud via COLMAP.

This is the local, no-deep-learning path. It's a thin, well-behaved wrapper
around two external tools:

    ffmpeg   -> splits the video into still images
    COLMAP   -> Structure-from-Motion / Multi-View Stereo (the actual 3D engine)

For the modern, GPU/Colab path (DUSt3R + optional semantics), use the
`reconstruct.ipynb` notebook and the `v3d/` package instead — see the README.

Pipeline
--------
    1. Validate inputs and check that ffmpeg + COLMAP are installed.
    2. Create the output folder structure (images/, colmap/).
    3. Extract frames from the video with ffmpeg.
    4. Drop the blurriest frames (variance of the Laplacian).
    5. Run COLMAP's automatic_reconstructor.
    6. Report where the results are.

Usage
-----
    python reconstruct_colmap.py VIDEO_PATH OUTPUT_DIR [--fps 2] [--quality medium]

Example
-------
    python reconstruct_colmap.py room.mp4 ./out --fps 2 --quality medium
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# OpenCV / NumPy are Python dependencies (not PATH tools), so fail early and
# clearly if they're missing rather than dying with a raw ImportError mid-run.
try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit(
        "ERROR: OpenCV / NumPy are not installed.\n"
        "       Install them with:  pip install opencv-python numpy"
    )

# Keep at least this many frames around no matter what — COLMAP needs a handful
# of overlapping views to triangulate anything at all.
MIN_FRAMES = 10


# ---------------------------------------------------------------------------
# External-command helper
# ---------------------------------------------------------------------------
def run_external_command(cmd):
    """Run a command, streaming its output live, and exit cleanly on failure."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}\n")
    try:
        # No capture_output -> child stdout/stderr stream straight to the console.
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(f"ERROR: command not found: {cmd[0]!r}. Is it installed and on PATH?")
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"ERROR: the following command failed (exit code {exc.returncode}):\n"
            f"       {printable}"
        )


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def check_dependencies():
    """Verify ffmpeg and COLMAP are on PATH; otherwise explain how to install."""
    missing = [tool for tool in ("ffmpeg", "colmap") if shutil.which(tool) is None]
    if not missing:
        return

    system = platform.system()
    hints = {
        "ffmpeg": {
            "Windows": "winget install Gyan.FFmpeg   (or: choco install ffmpeg)",
            "Darwin": "brew install ffmpeg",
            "Linux": "sudo apt install ffmpeg",
        },
        "colmap": {
            "Windows": "Download the prebuilt binary from https://github.com/colmap/colmap/releases "
                       "and add its folder to PATH",
            "Darwin": "brew install colmap",
            "Linux": "sudo apt install colmap   (or build from https://colmap.github.io)",
        },
    }

    print("ERROR: required tool(s) not found on PATH:\n", file=sys.stderr)
    for tool in missing:
        hint = hints[tool].get(system, f"See the {tool} install docs.")
        print(f"  - {tool}\n      install: {hint}\n", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 3 — frame extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path, images_dir, fps):
    """Split the video into still JPEGs at `fps` frames per second using ffmpeg."""
    # Remove any frames from a previous run so stale images can't leak into COLMAP.
    for old in images_dir.glob("frame_*.jpg"):
        old.unlink()

    pattern = str(images_dir / "frame_%04d.jpg")  # frame_0001.jpg, frame_0002.jpg, ...
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",                       # overwrite without an interactive prompt
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",           # high-quality JPEGs help feature matching
        pattern,
    ]
    run_external_command(cmd)

    count = len(list(images_dir.glob("frame_*.jpg")))
    print(f"Extracted {count} frames.")
    if count < MIN_FRAMES:
        print("WARNING: Very few frames — the video may be too short, or --fps too low.")
    return count


# ---------------------------------------------------------------------------
# Step 4 — blur filtering (the one improvement over a naive wrapper)
# ---------------------------------------------------------------------------
def variance_of_laplacian(gray):
    """Sharpness score: higher variance of the Laplacian == sharper image."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def filter_blurry_frames(images_dir):
    """Drop frames well below the median sharpness, but never go under MIN_FRAMES."""
    frames = sorted(images_dir.glob("frame_*.jpg"))
    if len(frames) <= MIN_FRAMES:
        print(f"Skipping blur filter ({len(frames)} frames — at or below the safe minimum).")
        return

    scores = {}
    for frame in frames:
        gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue  # unreadable file — leave it for COLMAP to ignore
        scores[frame] = variance_of_laplacian(gray)

    if not scores:
        print("Skipping blur filter (could not read any frames).")
        return

    # Drop frames well below the median sharpness.
    threshold = float(np.median(list(scores.values()))) * 0.5

    # Candidates below threshold, blurriest first.
    candidates = sorted(
        (f for f, s in scores.items() if s < threshold),
        key=lambda f: scores[f],
    )

    # Safety: never delete so many that fewer than MIN_FRAMES remain.
    max_removable = max(0, len(scores) - MIN_FRAMES)
    to_remove = candidates[:max_removable]

    for frame in to_remove:
        frame.unlink()

    removed = len(to_remove)
    print(f"Removed {removed} blurry frames, kept {len(scores) - removed}.")


# ---------------------------------------------------------------------------
# Step 5 — COLMAP reconstruction
# ---------------------------------------------------------------------------
def run_colmap(images_dir, work_dir, quality):
    """Run COLMAP's all-in-one pipeline: features -> matching -> SfM -> (MVS)."""
    print("\nRunning COLMAP — this can take several minutes...")
    cmd = [
        "colmap", "automatic_reconstructor",
        "--workspace_path", str(work_dir),
        "--image_path", str(images_dir),
        "--quality", quality,
    ]
    run_external_command(cmd)

    # COLMAP writes the sparse model into {work_dir}/sparse/0/ (or /1, /2 ...).
    sparse_models = [p for p in (work_dir / "sparse").glob("*") if p.is_dir()]
    if not sparse_models:
        sys.exit(
            "ERROR: COLMAP produced no reconstruction.\n"
            "       Likely causes: too few frames, too little camera movement,\n"
            "       or a textureless scene (blank walls, etc.).\n"
            "       Try a slower sweep of the room, or raise --fps."
        )


# ---------------------------------------------------------------------------
# Step 6 — report
# ---------------------------------------------------------------------------
def report_results(work_dir):
    """Tell the user where the outputs are and how to view them."""
    sparse_path = work_dir / "sparse" / "0"
    # Dense output lives at dense/fused.ply or dense/0/fused.ply depending on version.
    dense_clouds = sorted(work_dir.glob("dense/**/fused.ply"))

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Sparse model (always):  {sparse_path}")
    if dense_clouds:
        for ply in dense_clouds:
            print(f"Dense point cloud:      {ply}")
    else:
        print("Dense point cloud:      none "
              "(dense MVS needs an NVIDIA/CUDA GPU; the sparse model is still valid).")
    print(f"\nTo view: open the COLMAP GUI -> File -> Import Model -> {sparse_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct a 3D point cloud from a phone video using ffmpeg + COLMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path", help="Path to the input video (e.g. room.mp4)")
    parser.add_argument("output_dir", help="Directory for frames + COLMAP output")
    parser.add_argument("--fps", type=float, default=2,
                        help="Frames to extract per second of video")
    parser.add_argument("--quality", choices=["low", "medium", "high", "extreme"],
                        default="medium", help="COLMAP reconstruction quality")
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1 — validate.
    check_dependencies()

    video_path = Path(args.video_path)
    if not video_path.is_file():
        sys.exit(f"ERROR: video file not found: {video_path}")
    if args.fps <= 0:
        sys.exit("ERROR: --fps must be greater than 0.")

    # Step 2 — folder structure.
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    colmap_dir = output_dir / "colmap"
    try:
        images_dir.mkdir(parents=True, exist_ok=True)
        colmap_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.exit(f"ERROR: cannot write to output directory {output_dir}: {exc}")

    # Steps 3–6.
    extract_frames(video_path, images_dir, args.fps)
    filter_blurry_frames(images_dir)
    run_colmap(images_dir, colmap_dir, args.quality)
    report_results(colmap_dir)


if __name__ == "__main__":
    main()
