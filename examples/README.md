# Examples

A real result produced by `reconstruct.ipynb` (MASt3R, 25 frames → ~500k points)
from a handheld 180° walk-around of a chair and bench:

```
examples/output/
├── preview.png         # static render (shown in the top-level README)
├── preview.html        # interactive viewer — open in a browser and rotate
├── point_cloud.ply     # the reconstructed cloud (MeshLab / CloudCompare / Blender)
├── report.json         # frames used + point count
└── images/             # the input frames that were used
```

To regenerate: run `reconstruct.ipynb` in Colab, then unzip the downloaded
`video_to_3d_outputs.zip` into `examples/output/`.
