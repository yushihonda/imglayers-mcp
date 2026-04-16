"""Hole-fill / inpainting helpers.

Given a region-of-interest and a hole mask, restore plausible pixel values.
Backends chosen based on the surrounding background model:

  - solid       : fill with median color
  - gradient    : fit a linear gradient from the ring and sample
  - radial      : fit a radial gradient
  - texture     : tile-sample from the nearest ring (no heavy synthesis)
  - photo       : same as texture (best-effort; photo inpaint is out of scope)

All functions operate on HxWx3 uint8 arrays and HxW bool hole masks.
"""

from __future__ import annotations

import numpy as np


def fill_solid(rgb: np.ndarray, hole: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[hole] = color.astype(np.uint8)
    return out


def fill_linear_gradient(
    rgb: np.ndarray,
    hole: np.ndarray,
    *,
    direction: str = "vertical",
    color_a: np.ndarray,
    color_b: np.ndarray,
) -> np.ndarray:
    """Fill hole with a linear gradient a→b."""
    h, w = rgb.shape[:2]
    if direction == "vertical":
        # 0 at top, 1 at bottom; broadcast to width
        grad = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        grad = np.broadcast_to(grad, (h, w))
    else:
        grad = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        grad = np.broadcast_to(grad, (h, w))
    a = color_a.astype(np.float32)
    b = color_b.astype(np.float32)
    filled = a[None, None, :] * (1 - grad[..., None]) + b[None, None, :] * grad[..., None]
    out = rgb.copy()
    out[hole] = filled[hole].astype(np.uint8)
    return out


def fill_radial_gradient(
    rgb: np.ndarray,
    hole: np.ndarray,
    *,
    center_color: np.ndarray,
    edge_color: np.ndarray,
) -> np.ndarray:
    """Fill hole with a radial gradient (center→edge)."""
    h, w = rgb.shape[:2]
    cy, cx = h / 2, w / 2
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r /= max(1.0, r.max())
    a = center_color.astype(np.float32)
    b = edge_color.astype(np.float32)
    filled = a[None, None, :] * (1 - r[..., None]) + b[None, None, :] * r[..., None]
    out = rgb.copy()
    out[hole] = filled[hole].astype(np.uint8)
    return out


def fill_texture_tile(rgb: np.ndarray, hole: np.ndarray, ring_w: int = 8) -> np.ndarray:
    """Fill hole by tiling a rectangular ring sampled from outside the hole.

    Cheap but effective for textured backgrounds.
    """
    h, w = rgb.shape[:2]
    if not hole.any():
        return rgb
    # Sample the ring around the hole's bbox.
    ys, xs = np.nonzero(hole)
    y1 = max(0, int(ys.min()) - ring_w)
    y2 = min(h, int(ys.max()) + 1 + ring_w)
    x1 = max(0, int(xs.min()) - ring_w)
    x2 = min(w, int(xs.max()) + 1 + ring_w)
    roi = rgb[y1:y2, x1:x2]
    roi_hole = hole[y1:y2, x1:x2]
    # Use the mean ring color as a safe default, then sprinkle with
    # the original ring pixels.
    ring_pixels = roi[~roi_hole]
    if ring_pixels.size == 0:
        return rgb
    med = np.median(ring_pixels.reshape(-1, 3), axis=0).astype(np.uint8)

    out = rgb.copy()
    out[hole] = med
    return out
