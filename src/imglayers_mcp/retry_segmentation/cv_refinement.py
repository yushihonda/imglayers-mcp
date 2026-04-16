"""CV-only retry refinement (no Grounded-SAM required).

Runs a denser connected-component pass on a low-confidence layer's bbox
and returns the best-fitting mask. Useful when the initial LayerD pass
merged an icon with its background or cut a button into fragments.

Not a perfect replacement for Grounded-SAM but works out of the box.
"""

from __future__ import annotations

import numpy as np

from ..core._cc import connected_components
from ..utils.bbox import Box
from ..utils.logging import get_logger
from .grounded_sam_engine import RefinedMaskResult

log = get_logger(__name__)


def refine_by_cc(
    rgba: np.ndarray,
    bbox: Box,
    *,
    padding: float = 0.1,
    min_area_ratio: float = 0.02,
) -> RefinedMaskResult | None:
    """Re-segment a bbox region using dense connected-components.

    Steps:
      1. Expand bbox by `padding` for context
      2. Compute fg mask from color diff against the ring (outer border)
      3. Run CC; keep the largest component that overlaps the original bbox
      4. Return its mask rebased to full-canvas coordinates
    """
    h, w = rgba.shape[:2]
    pad_x = bbox.w * padding
    pad_y = bbox.h * padding
    x1 = max(0, int(round(bbox.x - pad_x)))
    y1 = max(0, int(round(bbox.y - pad_y)))
    x2 = min(w, int(round(bbox.x + bbox.w + pad_x)))
    y2 = min(h, int(round(bbox.y + bbox.h + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None

    sub_rgb = rgba[y1:y2, x1:x2, :3]
    sub_h = y2 - y1
    sub_w = x2 - x1

    # Estimate "outside" color from the expanded ring.
    ring_w = max(1, min(sub_h, sub_w) // 10)
    ring = np.concatenate([
        sub_rgb[:ring_w].reshape(-1, 3),
        sub_rgb[-ring_w:].reshape(-1, 3),
        sub_rgb[:, :ring_w].reshape(-1, 3),
        sub_rgb[:, -ring_w:].reshape(-1, 3),
    ], axis=0)
    if ring.size == 0:
        return None
    ring_color = np.median(ring, axis=0).astype(np.int32)

    diff = np.linalg.norm(sub_rgb.astype(np.int32) - ring_color, axis=2)
    # Denser threshold for retry (catches subtle boundaries).
    fg = diff > 15

    comps = connected_components(fg)
    if not comps:
        return None

    # Pick the component with highest overlap with the original bbox
    # (mapped into sub coordinates).
    orig_x1 = max(0, int(round(bbox.x)) - x1)
    orig_y1 = max(0, int(round(bbox.y)) - y1)
    orig_x2 = min(sub_w, int(round(bbox.x + bbox.w)) - x1)
    orig_y2 = min(sub_h, int(round(bbox.y + bbox.h)) - y1)
    if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
        return None
    orig_mask = np.zeros((sub_h, sub_w), dtype=bool)
    orig_mask[orig_y1:orig_y2, orig_x1:orig_x2] = True

    min_area = int(bbox.w * bbox.h * min_area_ratio)
    best = None
    best_overlap = 0
    for comp in comps:
        if comp["area"] < min_area:
            continue
        overlap = int(np.logical_and(comp["mask"], orig_mask).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best = comp
    if best is None:
        return None

    # Rebase mask to full canvas.
    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y1:y2, x1:x2] = best["mask"]

    # New bbox is the refined mask's extent (clipped to the expanded region).
    ys, xs = np.nonzero(full_mask)
    if ys.size == 0 or xs.size == 0:
        return None
    new_bbox = Box(
        x=float(xs.min()),
        y=float(ys.min()),
        w=float(xs.max() - xs.min() + 1),
        h=float(ys.max() - ys.min() + 1),
    )

    # Score: overlap fraction of new mask with the original bbox.
    orig_area = max(1, int(orig_mask.sum()))
    score = float(best_overlap / orig_area)

    return RefinedMaskResult(
        mask=full_mask,
        bbox=new_bbox,
        score=score,
        prompt="cv-refine",
    )
