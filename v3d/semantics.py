"""Semantic labelling of the 3D cloud (optional extension).

2D -> 3D label lifting, so labels are aligned with the geometry *by construction*
(both come from the same MASt3R camera/pointmap solution):

  1. Run a 2D semantic segmenter (SegFormer trained on ADE20K, which covers
     indoor furniture: chair, sofa, table, bed, cabinet, ...) on each frame.
  2. Every frame already carries a 3D point + confidence mask per pixel (see
     v3d.reconstruct.FrameResult), so each kept 3D point inherits its pixel's
     class label.
  3. Voxel-majority-vote across all frames, so labels are multi-view consistent
     and denoised rather than per-frame flicker.

Runs on the Colab GPU and only needs `transformers` + `pillow` (both already on
Colab). Non-target stuff (walls, floor, clutter) is grouped as "other".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEGFORMER_MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"

# ADE20K label *substrings* we treat as foreground objects to colour.
DEFAULT_TARGETS = [
    "chair", "armchair", "sofa", "couch", "bench", "stool", "seat",
    "table", "desk", "coffee table", "bed", "cabinet", "shelf", "wardrobe",
]

# Distinct colours for object classes; "other" is rendered grey.
_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
], dtype=np.uint8)


@dataclass
class LabeledReconstruction:
    points: np.ndarray   # (N, 3) float
    colors: np.ndarray   # (N, 3) uint8 — original RGB
    labels: np.ndarray   # (N,) int   — class id, or OTHER_ID for non-target
    id2name: dict        # class id -> name (includes the "other" id)
    other_id: int


def _import_segmenter():
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Semantic labelling needs `transformers` and `pillow`.\n"
            "  In Colab: !pip install -q transformers\n"
            f"(original error: {exc})"
        ) from exc
    return torch, SegformerForSemanticSegmentation, SegformerImageProcessor, Image


def _segment_frames(frames, device):
    """Run SegFormer on each frame -> list of (H, W) ADE20K class-id maps."""
    torch, SegformerForSemanticSegmentation, SegformerImageProcessor, Image = _import_segmenter()

    print(f"Loading segmenter ({SEGFORMER_MODEL})...")
    proc = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL).to(device).eval()
    id2name = {int(k): v for k, v in model.config.id2label.items()}

    maps = []
    for frame in frames:
        image = Image.fromarray(frame.rgb)
        inputs = proc(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits  # (1, C, h/4, w/4)
        upsampled = torch.nn.functional.interpolate(
            logits, size=frame.rgb.shape[:2], mode="bilinear", align_corners=False)
        maps.append(upsampled.argmax(1)[0].to("cpu").numpy().astype(np.int32))
    return maps, id2name


def _voxel_majority(points, labels, n_voxels=200):
    """Snap labels to the majority label within each voxel (multi-view consistency)."""
    lo = points.min(axis=0)
    span = points.max(axis=0) - lo
    voxel_size = max(float(span.max()) / n_voxels, 1e-6)

    keys = np.floor((points - lo) / voxel_size).astype(np.int64)
    _, vox = np.unique(keys, axis=0, return_inverse=True)   # voxel index per point

    n_labels = int(labels.max()) + 1
    code = vox.astype(np.int64) * n_labels + labels.astype(np.int64)
    uniq, counts = np.unique(code, return_counts=True)
    u_vox, u_lab = uniq // n_labels, uniq % n_labels

    # Within each voxel, keep the label with the highest count.
    order = np.lexsort((-counts, u_vox))
    u_vox, u_lab = u_vox[order], u_lab[order]
    first = np.empty(len(u_vox), dtype=bool)
    first[0] = True
    first[1:] = u_vox[1:] != u_vox[:-1]

    best = np.zeros(int(vox.max()) + 1, dtype=np.int64)
    best[u_vox[first]] = u_lab[first]
    return best[vox]


def label(reconstruction, target_classes=None, device="cuda",
          n_voxels=200, max_points=600_000, seed=0):
    """Assign a semantic class to each 3D point.

    Args:
        reconstruction: a v3d.reconstruct.Reconstruction (needs per-frame data).
        target_classes: substrings of ADE20K names to keep as objects
                        (default: common furniture). Everything else -> "other".
        device:         "cuda" recommended.
        n_voxels:       voxel-grid resolution for the consistency vote.
        max_points:     cap on returned points.

    Returns:
        LabeledReconstruction(points, colors, labels, id2name, other_id).
    """
    frames = reconstruction.frames
    if not frames:
        raise ValueError("Reconstruction has no per-frame data (frames=[]); "
                         "run reconstruct.run(...) first.")

    targets = [t.lower() for t in (target_classes or DEFAULT_TARGETS)]
    maps, id2name = _segment_frames(frames, device)

    other_id = max(id2name) + 1
    target_ids = sorted(i for i, n in id2name.items()
                        if any(t in n.lower() for t in targets))
    id2name_out = {i: id2name[i] for i in target_ids}
    id2name_out[other_id] = "other"
    target_arr = np.asarray(target_ids, dtype=np.int32)

    all_pts, all_cols, all_lab = [], [], []
    for frame, class_map in zip(frames, maps):
        m = frame.mask
        ade = class_map[m]
        lab = np.where(np.isin(ade, target_arr), ade, other_id)
        all_pts.append(frame.pts3d[m])
        all_cols.append(frame.rgb[m])
        all_lab.append(lab.astype(np.int64))

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    labels = np.concatenate(all_lab, axis=0)

    finite = np.isfinite(points).all(axis=1)
    points, colors, labels = points[finite], colors[finite], labels[finite]

    labels = _voxel_majority(points, labels, n_voxels=n_voxels)

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), max_points, replace=False)
        points, colors, labels = points[idx], colors[idx], labels[idx]

    n_obj = int((labels != other_id).sum())
    print(f"Labelled {len(points):,} points; {n_obj:,} on target objects "
          f"({len(target_ids)} candidate classes).")
    return LabeledReconstruction(points=points, colors=colors, labels=labels,
                                 id2name=id2name_out, other_id=other_id)


def semantic_colors(labeled):
    """(N, 3) uint8 colours: a distinct hue per object class, grey for 'other'."""
    object_ids = sorted(i for i in labeled.id2name if i != labeled.other_id)
    out = np.full((len(labeled.labels), 3), 128, dtype=np.uint8)  # grey
    for k, class_id in enumerate(object_ids):
        out[labeled.labels == class_id] = _PALETTE[k % len(_PALETTE)]
    return out


def summary(labeled):
    """Human-readable per-class point counts + the colour legend."""
    object_ids = sorted(i for i in labeled.id2name if i != labeled.other_id)
    color_of = {cid: _PALETTE[k % len(_PALETTE)] for k, cid in enumerate(object_ids)}

    uniq, counts = np.unique(labeled.labels, return_counts=True)
    counts_by_id = dict(zip(uniq.tolist(), counts.tolist()))

    lines = ["Semantic point counts:"]
    for cid in sorted(object_ids, key=lambda i: -counts_by_id.get(i, 0)):
        if counts_by_id.get(cid):
            r, g, b = color_of[cid]
            lines.append(f"  {labeled.id2name[cid]:18s} {counts_by_id[cid]:>8,}   rgb({r},{g},{b})")
    lines.append(f"  {'other':18s} {counts_by_id.get(labeled.other_id, 0):>8,}   rgb(128,128,128)")
    return "\n".join(lines)
