# humanoid-video-to-3d

Turn a short video into a **3D construction**, with semantic labels.

```
video → frames → blur filter → MASt3R → 3D point cloud → semantic labels 
```

---

## Example

### Input


https://github.com/user-attachments/assets/993dda47-cb36-4e2a-ba56-5998d1e649f6



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

**MASt3R over classical SfM** — COLMAP fails silently when features are sparse; MASt3R predicts depth and poses from learned priors so it works on textureless scenes and gives metric scale without calibration.

**Filter before reconstruct** — blurry frames produce low-confidence pointmaps that hurt the global alignment, not just add noise. Dropping them first improves quality, not just speed.

**Semantics aligned by construction** — each pixel already has a 3D point from MASt3R, so 2D labels are inherited directly rather than projected back, avoiding pose errors and resampling artifacts. Voxel majority vote then resolves cross-frame disagreements.
