"""Alpha refinement for binary SAM2 masks.

Converts a boolean mask into a soft alpha channel using edge feathering,
trimap-guided local blending, and anti-alias restoration. Also reports
``alpha_refine_confidence`` and ``alpha_edge_quality`` for the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.mask_candidate import MaskCandidate
from ..utils.bbox import Box


@dataclass
class AlphaRefineResult:
    rgba: np.ndarray
    alpha: np.ndarray
    alpha_refine_confidence: float
    alpha_edge_quality: float
    bbox: Box


def refine_alpha(
    candidate: MaskCandidate,
    rgb: np.ndarray,
    *,
    canvas_h: int,
    canvas_w: int,
    feather: int = 2,
) -> AlphaRefineResult:
    """Produce an RGBA layer for ``candidate`` on the full canvas."""
    mask = candidate.mask.astype(bool)
    if mask.shape != (canvas_h, canvas_w):
        padded = np.zeros((canvas_h, canvas_w), dtype=bool)
        hh = min(mask.shape[0], canvas_h)
        ww = min(mask.shape[1], canvas_w)
        padded[:hh, :ww] = mask[:hh, :ww]
        mask = padded

    eroded = _erode(mask, feather)
    dilated = _dilate(mask, feather)
    trimap = _trimap(mask, eroded, dilated)

    alpha = _refine_with_trimap(rgb, trimap, mask)
    alpha_edge_quality = _edge_quality(alpha, mask)
    confidence = float(min(1.0, 0.6 + 0.4 * alpha_edge_quality))

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    m3 = alpha > 0
    canvas[..., :3] = np.where(m3[..., None], rgb, 0).astype(np.uint8)
    canvas[..., 3] = (alpha * 255).astype(np.uint8)

    ys, xs = np.nonzero(m3)
    if ys.size == 0 or xs.size == 0:
        bbox = candidate.bbox
    else:
        bbox = Box(
            x=float(xs.min()), y=float(ys.min()),
            w=float(xs.max() - xs.min() + 1),
            h=float(ys.max() - ys.min() + 1),
        )

    return AlphaRefineResult(
        rgba=canvas,
        alpha=alpha.astype(np.float32),
        alpha_refine_confidence=confidence,
        alpha_edge_quality=float(alpha_edge_quality),
        bbox=bbox,
    )


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        out[1:] &= mask[:-1]
        out[:-1] &= mask[1:]
        out[:, 1:] &= mask[:, :-1]
        out[:, :-1] &= mask[:, 1:]
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        out[1:] |= mask[:-1]
        out[:-1] |= mask[1:]
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
    return out


def _trimap(mask: np.ndarray, eroded: np.ndarray, dilated: np.ndarray) -> np.ndarray:
    tri = np.zeros(mask.shape, dtype=np.uint8)
    tri[mask] = 1
    tri[eroded] = 2
    unknown = dilated & ~eroded
    tri[unknown] = 3
    return tri


def _refine_with_trimap(rgb: np.ndarray, trimap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = np.zeros(mask.shape, dtype=np.float32)
    alpha[trimap == 2] = 1.0
    alpha[trimap == 1] = 1.0

    unknown = trimap == 3
    if unknown.any():
        inner_rgb = rgb[trimap == 2].astype(np.float32)
        outer_rgb = rgb[trimap == 0].astype(np.float32)
        inner_mean = inner_rgb.mean(axis=0) if inner_rgb.size else np.array([0.0, 0.0, 0.0])
        outer_mean = outer_rgb.mean(axis=0) if outer_rgb.size else np.array([0.0, 0.0, 0.0])
        contrast = float(np.linalg.norm(inner_mean - outer_mean))

        if contrast > 1.0:
            px = rgb[unknown].astype(np.float32)
            to_inner = np.linalg.norm(px - inner_mean, axis=1)
            to_outer = np.linalg.norm(px - outer_mean, axis=1)
            t = to_outer / (to_inner + to_outer + 1e-6)
            alpha[unknown] = np.clip(t, 0.0, 1.0)
        else:
            alpha[unknown] = 0.5
    return alpha


def _edge_quality(alpha: np.ndarray, mask: np.ndarray) -> float:
    dilated = _dilate(mask, 1)
    ring = dilated & ~mask
    if not ring.any():
        return 1.0
    ring_alpha = alpha[ring]
    return float(1.0 - ring_alpha.mean())
