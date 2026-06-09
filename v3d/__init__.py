"""video-to-3d: reconstruct a 3D point cloud from a phone video.

Submodules (import the one you need so heavy deps load lazily):
    frames      - extract + deblur video frames                (CPU; ffmpeg + OpenCV)
    reconstruct - DUSt3R geometry -> coloured point cloud       (GPU; torch + dust3r)
    semantics   - 2D->3D semantic labels (optional)             (GPU) [roadmap]
    pointcloud  - PLY I/O + inline 3D viewer                    (CPU; numpy + plotly)
"""

__version__ = "0.1.0"
__all__ = ["frames", "reconstruct", "semantics", "pointcloud"]
