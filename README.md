# video-to-3d

Turn a short phone video into a **3D point cloud**, with optional semantic labels.

```
video → frames → blur filter → MASt3R → 3D point cloud → semantic labels (optional)
```

---

## Example

### Input
<!-- drag your .mov into the GitHub README editor to embed it -->
https://your-github-asset-url/input.mov

### RGB reconstruction
<!-- screenshot of preview.html in browser -->
![RGB point cloud](https://your-github-asset-url/rgb_pointcloud.png)

### Semantic labels
<!-- screenshot of the semantic-coloured cloud -->
![Semantic labels](https://your-github-asset-url/semantic_pointcloud.png)

---

## How to run

1. Open the notebook in Colab:
   [`reconstruct.ipynb`](https://colab.research.google.com/github/maheswariridhi/humanoid-video-to-3d/blob/main/reconstruct.ipynb)
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**, then upload your video when prompted
4. Outputs saved to `out/`: `point_cloud.ply`, `point_cloud_semantic.ply`, `preview.html`

**Capture tip:** slow 10–30 s walk-around of a small room, even lighting, no motion blur.

---

## Approach

- **MASt3R** predicts a 3D point per pixel and recovers camera poses — no COLMAP, no calibration needed
- Frames are extracted at 2 fps, blurry ones dropped, then thinned to 25 to fit GPU memory
- Semantic labels use **SegFormer (ADE20K)** on each frame, lifted to 3D via the MASt3R pointmaps, then voxel-majority-voted across frames for multi-view consistency
- Output is a binary PLY (opens in MeshLab, Blender, CloudCompare) + standalone interactive `preview.html`

---

## Design choices

**MASt3R over classical SfM** — handles casual phone video without COLMAP, produces a dense coloured cloud directly, and doesn't fail all-or-nothing on textureless scenes.

**Frame prep matters** — dropping blurry frames and capping at 25 views is cheap and keeps the joint GPU step inside a free Colab T4.

**Thin notebook, real modules** — all logic is in `v3d/*.py`, the notebook is just a 6-step driver. Easy to read and reuse.
