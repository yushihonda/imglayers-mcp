"""Text style estimation (spec §7.5).

Takes an OCR line + its original crop, runs a font classifier, and returns
a ReconstructedTextStyle with font_size, weight, color, line height, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.color import extract_foreground_color
from .font_classifier import FontCandidate, FontClassifier, FontConfig, HeuristicBackend


@dataclass
class ReconstructedTextStyle:
    font_family: str | None = None
    font_candidates: list[str] = field(default_factory=list)
    font_weight: int = 400
    font_size: float = 12.0
    line_height: float | None = None
    letter_spacing: float | None = None
    color: str | None = None
    text_align: str | None = None
    reconstruction_confidence: float = 0.0


def estimate_style(
    *,
    crop_rgb: np.ndarray,
    text: str,
    cfg: FontConfig,
    classifier: FontClassifier | None = None,
    canvas_bg_hex: str | None = None,
    canvas_width: int | None = None,
    line_bbox_x: float | None = None,
    line_bbox_w: float | None = None,
) -> ReconstructedTextStyle:
    if classifier is None:
        classifier = HeuristicBackend()

    # Font candidates.
    candidates: list[FontCandidate] = []
    try:
        candidates = classifier.rank_candidates(crop_rgb, text, cfg)
    except Exception:
        candidates = []

    # Font size ≈ 85% of crop height (cap-height + descender padding).
    h = crop_rgb.shape[0] if crop_rgb.ndim >= 2 else 12
    font_size = max(8.0, h * 0.85)

    # Color: extract foreground (non-bg) from crop.
    color = extract_foreground_color(crop_rgb, bg_hex=canvas_bg_hex)

    # Weight: rough stroke density.
    weight = _estimate_weight(crop_rgb, color)

    # Alignment: bbox center relative to canvas.
    text_align: str | None = None
    if canvas_width and line_bbox_x is not None and line_bbox_w is not None:
        cx = line_bbox_x + line_bbox_w / 2
        rel = cx / canvas_width
        if rel < 0.4:
            text_align = "left"
        elif rel > 0.6:
            text_align = "right"
        else:
            text_align = "center"

    best = candidates[0] if candidates else None
    confidence = best.score if best else 0.1

    return ReconstructedTextStyle(
        font_family=best.family if best else None,
        font_candidates=[c.family for c in candidates],
        font_weight=(best.weight if best else weight),
        font_size=round(font_size, 1),
        color=color,
        text_align=text_align,
        reconstruction_confidence=round(confidence, 3),
    )


def _estimate_weight(crop: np.ndarray, text_color: str | None) -> int:
    if crop.size == 0 or text_color is None:
        return 400
    from ..utils.color import hex_to_rgb

    target = np.array(hex_to_rgb(text_color), dtype=np.int32)
    if crop.ndim == 3:
        diff = np.linalg.norm(crop[..., :3].astype(np.int32) - target, axis=2)
    else:
        return 400
    density = float((diff < 48).mean())
    if density >= 0.45:
        return 900
    if density >= 0.30:
        return 700
    if density >= 0.18:
        return 600
    return 400
