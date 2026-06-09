"""Semantic labelling of the 3D cloud (optional extension) — ROADMAP / not yet built.

Planned approach (2D -> 3D label lifting), which keeps semantics *aligned with
the geometry* because both come from the same camera/pointmap solution:

  1. For each frame, run an open-vocabulary 2D segmenter (Grounded-SAM-2) with
     text prompts like "chair, table, sofa, monitor" to get per-pixel masks.
  2. Each frame already carries a 3D point per pixel and a confidence mask
     (see v3d.reconstruct.FrameResult.pts3d / .mask), so project each 2D mask
     label onto the corresponding 3D points.
  3. Aggregate per-3D-point labels across all frames by majority vote, so the
     labelling is multi-view consistent rather than per-frame flicker.

The geometry pipeline (frames -> reconstruct -> pointcloud) runs end to end
without this module; semantics is the next milestone.
"""
from __future__ import annotations


def label(reconstruction, prompts="chair, table, sofa, monitor", **kwargs):
    """Assign a semantic class to each 3D point. Not yet implemented."""
    raise NotImplementedError(
        "Semantic labelling is the next milestone (Grounded-SAM-2 + 2D->3D "
        "lifting). The geometry pipeline works end to end without it — see the "
        "README 'Roadmap' section."
    )
