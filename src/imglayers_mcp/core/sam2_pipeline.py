"""SAM2 decomposition pipeline.

Inputs:
  - rgba image
  - image_type (from classifier)
  - OCR lines (for text promotion hints)

Outputs:
  - list[RawLayer] compatible with the existing merger / manifest builder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..adapters.sam2_adapter import SAM2Adapter
from ..models.mask_candidate import MaskCandidate
from ..utils.bbox import Box
from ..utils.color import average_color
from .alpha_refiner import refine_alpha
from .mask_filter_merge import process as process_masks
from .types import RawLayer
from .zorder_estimator import estimate as estimate_zorder


@dataclass
class SAM2PipelineResult:
    raw_layers: list[RawLayer]
    candidates: list[MaskCandidate]
    background_candidate: MaskCandidate | None
    device: str
    checkpoint: str


def run(
    rgba: np.ndarray,
    image_type: str,
    *,
    sam2: SAM2Adapter,
    text_boxes: list[Box] | None = None,
) -> SAM2PipelineResult:
    rgb = rgba[..., :3]
    h, w = rgb.shape[:2]

    raw_candidates = sam2.generate_masks(rgb)
    foreground, bg = process_masks(raw_candidates, rgb, image_type)

    zorder = estimate_zorder(foreground, bg, rgb)

    raw_layers: list[RawLayer] = []
    tboxes = text_boxes or []

    if bg is not None:
        bg_rgb = rgb.copy()
        for cand in foreground:
            bg_rgb[cand.mask] = average_color_array(rgb, bg.mask)
        bg_rgba = np.concatenate(
            [bg_rgb, np.full((h, w, 1), 255, dtype=np.uint8)],
            axis=2,
        )
        raw_layers.append(RawLayer(
            rgba=bg_rgba,
            bbox=Box(0, 0, w, h),
            kind="image",
            engine="sam2",
            confidence=0.7,
            z_index=0,
            debug={
                "role_guess": "background",
                "source": "sam2",
                "bg_color": average_color(rgb),
            },
        ))

    for z, cand in zorder:
        if bg is not None and cand is bg:
            continue
        refine = refine_alpha(cand, rgb, canvas_h=h, canvas_w=w, feather=2)
        overlaps_text = any(_bbox_overlap(cand.bbox, tb) > 0.5 for tb in tboxes)
        kind = "text" if overlaps_text else _guess_kind(cand)
        raw_layers.append(RawLayer(
            rgba=refine.rgba,
            bbox=refine.bbox,
            kind=kind,
            engine="sam2",
            confidence=float(min(1.0, 0.5 + 0.5 * cand.score)),
            z_index=z,
            debug={
                "source": "sam2",
                "checkpoint": sam2._checkpoint,
                "device": sam2._device,
                "score": float(cand.score),
                "stability": float(cand.stability),
                "predicted_iou": float(cand.predicted_iou),
                "alpha_refine_confidence": float(refine.alpha_refine_confidence),
                "alpha_edge_quality": float(refine.alpha_edge_quality),
                "mask_quality": float(min(1.0, 0.5 + 0.5 * cand.predicted_iou)),
            },
        ))

    return SAM2PipelineResult(
        raw_layers=raw_layers,
        candidates=raw_candidates,
        background_candidate=bg,
        device=sam2._device,
        checkpoint=sam2._checkpoint,
    )


def average_color_array(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = rgb[mask]
    if pixels.size == 0:
        return np.array([128, 128, 128], dtype=np.uint8)
    return pixels.astype(np.float32).mean(axis=0).astype(np.uint8)


def _guess_kind(cand: MaskCandidate) -> str:
    aspect = cand.bbox.w / max(1.0, cand.bbox.h)
    density = cand.area / max(1.0, cand.bbox.area)
    if density > 0.85 and min(cand.bbox.w, cand.bbox.h) >= 32:
        return "image"
    if density < 0.3:
        return "vector_like"
    return "unknown"


def _bbox_overlap(a: Box, b: Box) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0
