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
    # WS2: 2-pass fitting diagnostics
    fit_score_pass1: float | None = None
    fit_score_pass2: float | None = None
    style_confidence: float | None = None
    pass2_updated: bool = False


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
    semantic_role: str | None = None,
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

    # Role-aware weight/alignment priors.
    weight, text_align_prior = _apply_role_priors(semantic_role, weight, crop_rgb)

    # Alignment: bbox center relative to canvas (overridden by role prior
    # when role is "button" or "badge" which are typically centered).
    text_align: str | None = text_align_prior
    if text_align is None and canvas_width and line_bbox_x is not None and line_bbox_w is not None:
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
    # Role-aware confidence: boost when font choice matches role prior.
    if semantic_role in ("headline", "subheadline") and best and best.weight >= 600:
        confidence = min(1.0, confidence + 0.1)
    if semantic_role == "body_text" and best and 400 <= best.weight <= 500:
        confidence = min(1.0, confidence + 0.1)

    return ReconstructedTextStyle(
        font_family=best.family if best else None,
        font_candidates=[c.family for c in candidates],
        font_weight=(best.weight if best else weight),
        font_size=round(font_size, 1),
        color=color,
        text_align=text_align,
        reconstruction_confidence=round(confidence, 3),
    )


def refine_with_final_role(
    *,
    pass1_style: ReconstructedTextStyle,
    crop_rgb: np.ndarray,
    text: str,
    cfg: FontConfig,
    classifier: FontClassifier | None,
    final_role: str | None,
) -> ReconstructedTextStyle:
    """Pass-2 fitting: re-rank candidates using the confirmed semantic role.

    Keeps the pass-1 result when pass-2's best candidate does not improve
    the rerender fit score.
    """
    if classifier is None:
        classifier = HeuristicBackend()

    # Re-rank.
    try:
        candidates = classifier.rank_candidates(crop_rgb, text, cfg)
    except Exception:
        candidates = []

    # Apply role priors to pick top candidate.
    if final_role in ("headline", "subheadline"):
        candidates_sorted = sorted(
            candidates,
            key=lambda c: (-(1.0 if c.weight >= 600 else 0.0), -c.score),
        )
    elif final_role == "button":
        candidates_sorted = sorted(
            candidates,
            key=lambda c: (-(1.0 if 500 <= c.weight <= 700 else 0.0), -c.score),
        )
    elif final_role == "body_text":
        candidates_sorted = sorted(
            candidates,
            key=lambda c: (-(1.0 if 400 <= c.weight <= 500 else 0.0), -c.score),
        )
    else:
        candidates_sorted = sorted(candidates, key=lambda c: -c.score)

    best2 = candidates_sorted[0] if candidates_sorted else None
    pass2_score = best2.score if best2 else 0.0
    pass1_score = pass1_style.fit_score_pass1 or pass1_style.reconstruction_confidence

    out = ReconstructedTextStyle(
        font_family=pass1_style.font_family,
        font_candidates=list(pass1_style.font_candidates),
        font_weight=pass1_style.font_weight,
        font_size=pass1_style.font_size,
        color=pass1_style.color,
        text_align=pass1_style.text_align,
        reconstruction_confidence=pass1_style.reconstruction_confidence,
        fit_score_pass1=pass1_score,
        fit_score_pass2=pass2_score,
    )

    if best2 and pass2_score > pass1_score + 0.02:
        out.font_family = best2.family
        out.font_weight = best2.weight
        out.font_candidates = [c.family for c in candidates_sorted[: cfg.top_k]]
        out.reconstruction_confidence = pass2_score
        out.pass2_updated = True

    out.style_confidence = max(pass1_score, pass2_score)
    return out


def _apply_role_priors(
    role: str | None, base_weight: int, crop: np.ndarray
) -> tuple[int, str | None]:
    """Adjust weight + default alignment based on semantic role."""
    if role in ("headline", "subheadline"):
        # Headlines are usually bold; bias up unless stroke density is very low.
        weight = max(base_weight, 700 if role == "headline" else 600)
        return weight, None
    if role == "button":
        # Button labels: medium/bold and centered.
        weight = max(base_weight, 600)
        return weight, "center"
    if role == "badge":
        weight = max(base_weight, 700)
        return weight, "center"
    if role == "body_text":
        # Body stays near the measured value (usually 400).
        return base_weight, None
    if role == "logo":
        weight = max(base_weight, 700)
        return weight, None
    return base_weight, None


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
