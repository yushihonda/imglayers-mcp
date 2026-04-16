"""Pixel-level residual utilities.

Given two images of the same size, produce per-pixel difference maps and
aggregate them over bboxes or masks. Used by Verifier v2 and Metrics v2.
"""

from __future__ import annotations

import numpy as np

from ..utils.bbox import Box


def rgb_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel L1 RGB diff normalized to [0, 1]."""
    if a.shape[2] > 3:
        a = a[..., :3]
    if b.shape[2] > 3:
        b = b[..., :3]
    d = np.abs(a.astype(np.int32) - b.astype(np.int32)).mean(axis=2) / 255.0
    return d.astype(np.float32)


def bbox_stats(diff: np.ndarray, bbox: Box, canvas_h: int, canvas_w: int) -> dict:
    """Return summary stats of diff within bbox."""
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(canvas_w, int(round(bbox.x + bbox.w)))
    y2 = min(canvas_h, int(round(bbox.y + bbox.h)))
    if x2 <= x1 or y2 <= y1:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "area": 0}
    region = diff[y1:y2, x1:x2]
    flat = region.reshape(-1)
    return {
        "mean": float(flat.mean()),
        "p95": float(np.percentile(flat, 95)),
        "max": float(flat.max()),
        "area": int(flat.size),
    }


def border_stats(diff: np.ndarray, bbox: Box, canvas_h: int, canvas_w: int, thickness: int = 3) -> dict:
    """Residual along the bbox's outer border (edge leakage proxy)."""
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(canvas_w, int(round(bbox.x + bbox.w)))
    y2 = min(canvas_h, int(round(bbox.y + bbox.h)))
    if x2 - x1 <= thickness * 2 or y2 - y1 <= thickness * 2:
        return {"mean": 0.0, "p95": 0.0}
    top = diff[y1:y1 + thickness, x1:x2]
    bot = diff[y2 - thickness:y2, x1:x2]
    left = diff[y1:y2, x1:x1 + thickness]
    right = diff[y1:y2, x2 - thickness:x2]
    flat = np.concatenate([top.reshape(-1), bot.reshape(-1), left.reshape(-1), right.reshape(-1)])
    return {
        "mean": float(flat.mean()),
        "p95": float(np.percentile(flat, 95)),
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks."""
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter / union) if union > 0 else 0.0


def alpha_edge_mismatch(pred_alpha: np.ndarray, observed_fg: np.ndarray, ring: int = 2) -> float:
    """Measure mismatch along the alpha edge.

    pred_alpha: HxW 0..255 predicted alpha
    observed_fg: HxW bool mask from the source image
    Returns fraction of edge pixels where predicted alpha and observed fg disagree.
    """
    pred_bin = pred_alpha > 128
    # Edge of predicted = dilation xor erosion (simple ring).
    e = pred_bin.copy()
    for _ in range(ring):
        e[1:] |= pred_bin[:-1]
        e[:-1] |= pred_bin[1:]
        e[:, 1:] |= pred_bin[:, :-1]
        e[:, :-1] |= pred_bin[:, 1:]
    edge = e & ~pred_bin  # one-pixel ring outside
    if not edge.any():
        return 0.0
    disagreement = edge & observed_fg
    return float(disagreement.sum() / edge.sum())
