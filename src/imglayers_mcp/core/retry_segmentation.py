"""Retry hierarchy (spec WS5).

Tier ladder (cheapest → heaviest):
  1. cc_refine            — dense connected-components on the bbox ROI
  2. edge_refine          — tighten bbox to the strongest edge ring
  3. mask_expand_shrink   — morphological open/close + reselect
  4. grounded_sam         — optional, heavy, top-K only

The orchestrator runs tier 1 on every retry item, then escalates to the next
tier only if the previous one fails to improve the score. Grounded-SAM is
available only when the optional adapter initializes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.bbox import Box
from ..utils.logging import get_logger
from ._cc import connected_components

log = get_logger(__name__)


@dataclass
class RefinedMaskResult:
    mask: np.ndarray  # HxW bool
    bbox: Box
    score: float
    backend: str  # "cc-refine" | "edge-refine" | "mask-morph" | "grounded-sam"
    notes: str = ""


def refine(
    rgba: np.ndarray,
    bbox: Box,
    *,
    role: str | None = None,
    gsam_adapter: Any = None,
    sam2_adapter: Any = None,
    allow_grounded_sam: bool = False,
    allow_sam2: bool = False,
    tier_cap: int = 3,  # how many tiers to try
) -> RefinedMaskResult | None:
    """Run the tier ladder on a single bbox. Returns the best mask found."""
    tried: list[RefinedMaskResult] = []
    result = refine_cc(rgba, bbox)
    if result is not None:
        tried.append(result)
        if result.score >= 0.85:
            return result

    if tier_cap >= 2:
        er = refine_edge(rgba, bbox)
        if er is not None:
            tried.append(er)
            if er.score >= 0.85:
                return er

    if tier_cap >= 3:
        mm = refine_mask_morph(rgba, bbox)
        if mm is not None:
            tried.append(mm)

    if allow_sam2 and sam2_adapter is not None and getattr(sam2_adapter, "available", False):
        try:
            cand = sam2_adapter.refine_bbox(rgba[..., :3], bbox)
            if cand is not None:
                tried.append(RefinedMaskResult(
                    mask=cand.mask, bbox=cand.bbox, score=float(cand.score),
                    backend="sam2-prompt",
                ))
        except Exception as exc:
            log.debug("sam2 bbox refine failed: %s", exc)

    if allow_grounded_sam and gsam_adapter is not None and getattr(gsam_adapter, "available", False):
        try:
            gs = gsam_adapter.refine(rgba, prompt=role or "object", bbox=bbox)
            if gs is not None:
                tried.append(RefinedMaskResult(
                    mask=gs.mask, bbox=gs.bbox, score=float(gs.score),
                    backend="grounded-sam",
                ))
        except NotImplementedError:
            pass
        except Exception as exc:
            log.debug("grounded-sam refinement failed: %s", exc)

    if not tried:
        return None
    return max(tried, key=lambda r: r.score)


def refine_cc(rgba: np.ndarray, bbox: Box, *, padding: float = 0.1) -> RefinedMaskResult | None:
    h, w = rgba.shape[:2]
    pad_x, pad_y = bbox.w * padding, bbox.h * padding
    x1 = max(0, int(round(bbox.x - pad_x)))
    y1 = max(0, int(round(bbox.y - pad_y)))
    x2 = min(w, int(round(bbox.x + bbox.w + pad_x)))
    y2 = min(h, int(round(bbox.y + bbox.h + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    sub = rgba[y1:y2, x1:x2, :3]
    ring_w = max(1, min(sub.shape[:2]) // 10)
    ring = np.concatenate([
        sub[:ring_w].reshape(-1, 3), sub[-ring_w:].reshape(-1, 3),
        sub[:, :ring_w].reshape(-1, 3), sub[:, -ring_w:].reshape(-1, 3),
    ], axis=0)
    if ring.size == 0:
        return None
    ring_color = np.median(ring, axis=0).astype(np.int32)
    diff = np.linalg.norm(sub.astype(np.int32) - ring_color, axis=2)
    fg = diff > 15
    comps = connected_components(fg)
    if not comps:
        return None

    orig_x1 = max(0, int(round(bbox.x)) - x1)
    orig_y1 = max(0, int(round(bbox.y)) - y1)
    orig_x2 = min(sub.shape[1], int(round(bbox.x + bbox.w)) - x1)
    orig_y2 = min(sub.shape[0], int(round(bbox.y + bbox.h)) - y1)
    if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
        return None
    orig_mask = np.zeros(sub.shape[:2], dtype=bool)
    orig_mask[orig_y1:orig_y2, orig_x1:orig_x2] = True
    min_area = max(32, int(bbox.w * bbox.h * 0.02))

    best, best_overlap = None, 0
    for comp in comps:
        if comp["area"] < min_area:
            continue
        o = int(np.logical_and(comp["mask"], orig_mask).sum())
        if o > best_overlap:
            best_overlap = o
            best = comp
    if best is None:
        return None
    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y1:y2, x1:x2] = best["mask"]
    ys, xs = np.nonzero(full_mask)
    if ys.size == 0:
        return None
    new_bbox = Box(
        x=float(xs.min()), y=float(ys.min()),
        w=float(xs.max() - xs.min() + 1),
        h=float(ys.max() - ys.min() + 1),
    )
    orig_area = max(1, int(orig_mask.sum()))
    score = float(best_overlap / orig_area)
    return RefinedMaskResult(mask=full_mask, bbox=new_bbox, score=score, backend="cc-refine")


def refine_edge(rgba: np.ndarray, bbox: Box) -> RefinedMaskResult | None:
    """Tighten bbox to the strongest edge rectangle inside the region.

    Useful for cards/buttons whose CC partially bleeds; we look for the
    rectangle with the highest edge magnitude around its perimeter.
    """
    h, w = rgba.shape[:2]
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(w, int(round(bbox.x + bbox.w)))
    y2 = min(h, int(round(bbox.y + bbox.h)))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    sub = rgba[y1:y2, x1:x2, :3].astype(np.int32)
    gx = np.abs(np.diff(sub.sum(axis=2), axis=1))
    gy = np.abs(np.diff(sub.sum(axis=2), axis=0))
    col_edge = gx.mean(axis=0)  # length W-1
    row_edge = gy.mean(axis=1)  # length H-1
    # Pick left/right as top-1 edge-strength columns in outer 30% of region.
    w_sub = col_edge.shape[0]
    h_sub = row_edge.shape[0]
    if w_sub < 10 or h_sub < 10:
        return None
    left_cut = int(np.argmax(col_edge[: w_sub // 3]))
    right_cut = int(w_sub - 1 - np.argmax(col_edge[-w_sub // 3:][::-1]))
    top_cut = int(np.argmax(row_edge[: h_sub // 3]))
    bot_cut = int(h_sub - 1 - np.argmax(row_edge[-h_sub // 3:][::-1]))
    if right_cut <= left_cut + 10 or bot_cut <= top_cut + 10:
        return None
    new_x1 = x1 + left_cut
    new_y1 = y1 + top_cut
    new_x2 = x1 + right_cut
    new_y2 = y1 + bot_cut
    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[new_y1:new_y2, new_x1:new_x2] = True
    new_bbox = Box(
        x=float(new_x1), y=float(new_y1),
        w=float(new_x2 - new_x1), h=float(new_y2 - new_y1),
    )
    # Score = how much the new bbox shrinks by without losing area drastically.
    shrink = 1.0 - (new_bbox.area / max(1.0, bbox.area))
    score = max(0.0, 0.5 + shrink * 0.3)  # small positive for any shrink
    return RefinedMaskResult(mask=full_mask, bbox=new_bbox, score=score, backend="edge-refine")


def refine_mask_morph(rgba: np.ndarray, bbox: Box) -> RefinedMaskResult | None:
    """Open/close the CC mask to close holes + remove noise."""
    h, w = rgba.shape[:2]
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(w, int(round(bbox.x + bbox.w)))
    y2 = min(h, int(round(bbox.y + bbox.h)))
    if x2 <= x1 or y2 <= y1:
        return None
    sub = rgba[y1:y2, x1:x2, :3]
    ring = sub[:3].reshape(-1, 3) if sub.shape[0] >= 3 else sub.reshape(-1, 3)
    ring_color = np.median(ring, axis=0).astype(np.int32)
    diff = np.linalg.norm(sub.astype(np.int32) - ring_color, axis=2)
    mask = diff > 15

    # Morph close (dilate then erode) to fill holes.
    mask_c = mask.copy()
    for _ in range(2):
        mask_c[1:] |= mask_c[:-1]
        mask_c[:-1] |= mask_c[1:]
        mask_c[:, 1:] |= mask_c[:, :-1]
        mask_c[:, :-1] |= mask_c[:, 1:]
    for _ in range(2):
        mask_c[1:] &= mask_c[:-1]
        mask_c[:-1] &= mask_c[1:]
        mask_c[:, 1:] &= mask_c[:, :-1]
        mask_c[:, :-1] &= mask_c[:, 1:]

    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y1:y2, x1:x2] = mask_c
    ys, xs = np.nonzero(full_mask)
    if ys.size == 0:
        return None
    new_bbox = Box(
        x=float(xs.min()), y=float(ys.min()),
        w=float(xs.max() - xs.min() + 1),
        h=float(ys.max() - ys.min() + 1),
    )
    score = min(1.0, float(mask_c.mean()) * 1.5)
    return RefinedMaskResult(mask=full_mask, bbox=new_bbox, score=score, backend="mask-morph")
