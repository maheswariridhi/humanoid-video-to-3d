# video-to-3d

Turn a short phone video of a room into a **3D point cloud**.

This repo has **two paths to the same goal**:

| Path | Engine | Where it runs | When to use it |
|------|--------|---------------|----------------|
| **Modern (recommended)** | [DUSt3R](https://github.com/naver/dust3r) — a learned, feed-forward 3D model | **Colab GPU** (free T4) via [`reconstruct.ipynb`](reconstruct.ipynb) | No local GPU; robust on phone video; gives a dense coloured cloud directly |
| **Classical baseline** | [COLMAP](https://colmap.github.io) Structure-from-Motion + MVS | **Local** via [`reconstruct_colmap.py`](reconstruct_colmap.py) | You have COLMAP installed (and ideally an NVIDIA GPU for the dense step) |

The modern path is robust where classical SfM struggles (blank walls, little
camera motion, few frames) and needs **no GPU on your own machine** — Colab lends
you one. The classical path is kept as a transparent, dependency-free baseline.

---

## Quick start — modern path (Colab, no local GPU)

1. Push this repo to your own GitHub (see *Submitting* below).
2. Open [`reconstruct.ipynb`](reconstruct.ipynb) in Google Colab.
3. `Runtime → Change runtime type → T4 GPU`.
4. Edit `REPO_URL` in the setup cell to your fork.
5. `Runtime → Run all`, upload a short clip, and download the resulting
   `point_cloud.ply`.

What the notebook does, end to end:

```
upload video → extract frames → blur filter → subsample → DUSt3R → view inline → save .ply
   (ffmpeg)        (CPU)          (CPU)        (CPU)       (GPU)      (plotly)
```

**Capture tip:** a slow 10–30 s sweep of a small room, walking *around* objects
(not just rotating on the spot), with even lighting and no motion blur.

---

## Quick start — classical path (local COLMAP)

```bash
pip install -r requirements.txt          # OpenCV, NumPy, plotly
# install ffmpeg + COLMAP (see "Installing the local tools" below)
python reconstruct_colmap.py room.mp4 ./out --fps 2 --quality medium
```

| Argument     | Default  | Meaning                                               |
|--------------|----------|-------------------------------------------------------|
| `video_path` | —        | Path to the input `.mp4`                              |
| `output_dir` | —        | Where frames + COLMAP output go                       |
| `--fps`      | `2`      | Frames extracted per second of video                  |
| `--quality`  | `medium` | COLMAP quality: `low` / `medium` / `high` / `extreme` |

Output: `out/colmap/sparse/0/` always; `out/colmap/dense/.../fused.ply` only on a
CUDA GPU. View with the COLMAP GUI → *File → Import Model*.

---

## Repository layout

```
video-to-3d/
├── reconstruct.ipynb        # ← Colab notebook (the modern path you actually run)
├── v3d/                     # the real code, imported by the notebook
│   ├── frames.py            # extract frames + blur filter + subsample   (CPU)
│   ├── reconstruct.py       # DUSt3R → coloured point cloud               (GPU)
│   ├── semantics.py         # 2D→3D semantic labels                       (roadmap)
│   └── pointcloud.py        # binary-PLY writer + inline 3D viewer        (CPU)
├── reconstruct_colmap.py    # classical COLMAP baseline (local, standalone)
├── examples/                # sample input + committed example outputs
├── requirements.txt
└── README.md
```

The notebook is deliberately thin — each cell just calls a `v3d` function — so
the logic lives in readable, reusable modules rather than in notebook cells.

---

## Installing the local tools (classical path only)

The script checks for both at startup and prints these hints if either is missing.

| Tool   | Windows | macOS | Linux |
|--------|---------|-------|-------|
| ffmpeg | `winget install Gyan.FFmpeg` | `brew install ffmpeg` | `sudo apt install ffmpeg` |
| COLMAP | [prebuilt release](https://github.com/colmap/colmap/releases) → add to PATH | `brew install colmap` | `sudo apt install colmap` |

---

## Roadmap

- [x] Classical COLMAP baseline (local)
- [x] CPU frame prep: extraction + blur filter + subsampling
- [x] DUSt3R reconstruction on Colab GPU → coloured point cloud
- [x] Inline 3D viewer + PLY export
- [ ] **Semantic labels in 3D** — Grounded-SAM-2 per frame, lifted onto the
      DUSt3R pointmaps and majority-voted per 3D point, so labels stay aligned
      with the geometry (`v3d/semantics.py` has the interface + plan)
- [ ] Optional: 3D Gaussian Splatting on the recovered poses for a photorealistic,
      explorable result

---

## Design choices

- **Two engines, one repo.** COLMAP is the transparent classical reference;
  DUSt3R is a modern feed-forward model that trades a little metric precision for
  a lot of robustness on casual phone video — and removes the COLMAP install and
  the SfM "all-or-nothing" failure mode.
- **Frame prep matters more than the engine.** Sampling at a fixed fps, dropping
  the blurriest frames (relative to the median, with a hard floor), and thinning
  to a few dozen well-spread views is cheap and high-impact — it's also what
  keeps the joint GPU step inside a free T4's memory.
- **Thin notebook, real modules.** All logic sits in `v3d/*.py`; the notebook is
  a 6-step driver. Easy to read, reuse, and run locally if you do have a GPU.
- **Dependency-light I/O.** A hand-rolled binary PLY writer and a plotly viewer
  keep the output portable (MeshLab/CloudCompare/Blender) with no heavy 3D libs.

---

## Submitting / pushing to GitHub

```bash
cd video-to-3d
git init
git add .
git commit -m "video-to-3d: DUSt3R (Colab) + COLMAP baseline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/video-to-3d.git
git push -u origin main
```

Then set `REPO_URL` in `reconstruct.ipynb` to that remote, and (optionally) add an
[Open in Colab](https://colab.research.google.com/github/YOUR_USERNAME/video-to-3d/blob/main/reconstruct.ipynb)
badge to the top of this README.
