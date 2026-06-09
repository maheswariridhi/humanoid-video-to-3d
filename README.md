# video-to-3d

Turn a short phone video of a room into a **3D point cloud** — using
[DUSt3R](https://github.com/naver/dust3r), a learned, feed-forward 3D model.
No COLMAP, no camera calibration, and **no GPU on your own machine** (it runs on
a free Colab T4).

DUSt3R predicts a 3D point for every pixel and recovers the camera poses in one
pass, so it stays robust on casual phone video — including the low-texture,
low-parallax shots where classical Structure-from-Motion tends to fail outright.

```
upload video → extract frames → blur filter → subsample → DUSt3R → view inline → save .ply
   (ffmpeg)        (CPU)          (CPU)        (CPU)       (GPU)      (plotly)
```

---

## Quick start (Google Colab — no local GPU)

1. Make sure this repo is on GitHub (see *Pushing to GitHub* below).
2. Open the notebook in Colab:
   `https://colab.research.google.com/github/maheswariridhi/video-to-3d/blob/main/reconstruct.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**
4. **Runtime → Run all**, then upload a short clip when prompted.
5. View the cloud inline and download `point_cloud.ply`.

**Capture tip:** a slow 10–30 s sweep of a small room or a desk, walking
*around* objects (not just rotating on the spot), with even lighting and no
motion blur. More parallax = better depth.

---

## Repository layout

```
video-to-3d/
├── reconstruct.ipynb        # ← the Colab notebook you run (6 steps, top to bottom)
├── v3d/                     # the real code, imported by the notebook
│   ├── frames.py            # extract frames + blur filter + subsample   (CPU)
│   ├── reconstruct.py       # DUSt3R → coloured point cloud               (GPU)
│   ├── semantics.py         # 2D→3D semantic labels                       (roadmap)
│   └── pointcloud.py        # binary-PLY writer + inline 3D viewer        (CPU)
├── examples/                # sample input + committed example outputs
├── requirements.txt
└── README.md
```

The notebook is deliberately thin — each cell just calls a `v3d` function — so
the logic lives in readable, reusable modules rather than in notebook cells.

---

## Roadmap

- [x] CPU frame prep: extraction + blur filter + subsampling
- [x] DUSt3R reconstruction on Colab GPU → coloured point cloud
- [x] Inline 3D viewer + PLY export
- [ ] **Semantic labels in 3D** — Grounded-SAM-2 per frame, lifted onto the
      DUSt3R pointmaps and majority-voted per 3D point, so labels stay aligned
      with the geometry (`v3d/semantics.py` has the interface + plan)
- [ ] Optional: 3D Gaussian Splatting on the recovered poses for a
      photorealistic, explorable result

---

## Design choices

- **Why DUSt3R instead of classical SfM.** A learned feed-forward model trades a
  little metric precision for a lot of robustness on casual phone video, removes
  the COLMAP install, produces a dense coloured cloud directly, and avoids the
  "all-or-nothing" failure mode of incremental SfM on textureless scenes.
- **Frame prep matters more than the engine.** Sampling at a fixed fps, dropping
  the blurriest frames (relative to the median, with a hard floor), and thinning
  to a few dozen well-spread views is cheap and high-impact — it's also what
  keeps the joint GPU step inside a free T4's memory.
- **Thin notebook, real modules.** All logic sits in `v3d/*.py`; the notebook is
  a 6-step driver. Easy to read, reuse, and run locally if you do have a GPU.
- **Dependency-light I/O.** A hand-rolled binary PLY writer and a plotly viewer
  keep the output portable (MeshLab/CloudCompare/Blender) with no heavy 3D libs.

---

## Pushing to GitHub

```bash
cd video-to-3d
git add .
git commit -m "video-to-3d: DUSt3R reconstruction pipeline"
git push
```

The repo must be **public** for the notebook's `git clone` step to work without
authentication (and for submission). You can also add an
[Open in Colab](https://colab.research.google.com/github/maheswariridhi/video-to-3d/blob/main/reconstruct.ipynb)
badge to the top of this README.
