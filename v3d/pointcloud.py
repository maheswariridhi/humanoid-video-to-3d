"""Point-cloud I/O and inline visualisation — CPU only.

Deliberately dependency-light: a hand-rolled binary PLY writer (so any viewer —
MeshLab, CloudCompare, Blender — can open the result) and a plotly scatter that
renders an interactive 3D view right inside a Colab/Jupyter notebook.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_ply(points, colors, path):
    """Write (N,3) points + (N,3) uint8 colors as a binary little-endian PLY."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    vertex = np.empty(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vertex.tobytes())
    print(f"Saved {n:,} points -> {path}")
    return path


def show(points, colors=None, max_points=80_000, point_size=1.5):
    """Interactive 3D scatter inside the notebook (plotly).

    Randomly subsamples to `max_points` so the browser stays responsive.
    Returns the plotly Figure, so callers can also do ``fig.write_html(...)``
    to save a standalone viewer that opens without rerunning anything.
    """
    import plotly.graph_objects as go

    points = np.asarray(points)
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
        colors = None if colors is None else np.asarray(colors)[idx]

    marker_color = None
    if colors is not None:
        colors = np.asarray(colors)
        marker_color = ["rgb(%d,%d,%d)" % (r, g, b) for r, g, b in colors[:, :3]]

    fig = go.Figure(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=point_size, color=marker_color),
    ))
    fig.update_layout(scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=0, b=0))
    fig.show()
    return fig
