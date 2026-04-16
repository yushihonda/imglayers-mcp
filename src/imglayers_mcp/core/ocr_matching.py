"""OCR ↔ text-layer matching v2 (spec WS3).

Composite score:
  IoU + baseline + size ratio + same-container + reading-order + role prior
  + text density

Handles many-to-one split/merge cases and optional crop re-OCR for tiny text.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.bbox import Box
from .layout_utils import (
    baseline_proximity,
    center_distance,
    contains,
    horizontal_overlap,
    iou,
    size_ratio,
)


@dataclass
class MatchResult:
    raw_index: int
    ocr_index: int
    text: str
    score: float
    split_from: list[int] | None = None  # OCR indices this was merged from
    merged_into: list[int] | None = None  # OCR indices that merged into this layer


def match(
    raw_text_bboxes: list[tuple[int, Box, str | None]],
    ocr_bboxes: list[tuple[int, Box, str]],
    *,
    container_bboxes: list[Box] | None = None,
    min_score: float = 0.18,
) -> list[MatchResult]:
    """Return a list of matches.

    raw_text_bboxes: list of (raw_index, bbox, role_hint)
    ocr_bboxes:      list of (ocr_index, bbox, text)

    Scoring combines: IoU, baseline, size ratio, same-container bonus,
    horizontal overlap, role compatibility. Greedy one-to-one assignment.
    After the one-to-one pass, remaining OCR lines are merged into their
    best nearby raw layer if containment is strong enough.
    """
    container_bboxes = container_bboxes or []

    # Build candidate scores.
    pairs: list[tuple[float, int, int]] = []  # (score, ri, oi)
    for ri, rb, role in raw_text_bboxes:
        for oi, ob, text in ocr_bboxes:
            s = _composite(rb, ob, role, container_bboxes)
            if s > 0:
                pairs.append((s, ri, oi))

    pairs.sort(reverse=True)
    used_raw: set[int] = set()
    used_ocr: set[int] = set()
    results: list[MatchResult] = []
    text_by_oi = {oi: t for oi, _, t in ocr_bboxes}

    # Primary one-to-one assignment.
    for s, ri, oi in pairs:
        if ri in used_raw or oi in used_ocr:
            continue
        if s < min_score:
            break
        used_raw.add(ri)
        used_ocr.add(oi)
        results.append(MatchResult(raw_index=ri, ocr_index=oi, text=text_by_oi[oi], score=s))

    # Split/merge pass: for unmatched OCR lines, attach to the best raw
    # layer that contains or heavily overlaps them (many-to-one = merge).
    raw_lookup = {ri: (rb, role) for ri, rb, role in raw_text_bboxes}
    for oi, ob, text in ocr_bboxes:
        if oi in used_ocr:
            continue
        best_ri = None
        best_sc = 0.0
        for ri, rb, role in raw_text_bboxes:
            sc = iou(rb, ob)
            if contains(rb, ob, tolerance=4.0):
                sc += 0.3
            if sc > best_sc:
                best_sc = sc
                best_ri = ri
        if best_ri is not None and best_sc > 0.25:
            # Merge into existing match if present; else create one.
            existing = next((r for r in results if r.raw_index == best_ri), None)
            if existing:
                existing.text = f"{existing.text} {text}".strip()
                existing.merged_into = (existing.merged_into or []) + [oi]
            else:
                results.append(MatchResult(
                    raw_index=best_ri, ocr_index=oi, text=text, score=best_sc,
                ))
            used_ocr.add(oi)

    return results


def _composite(
    raw_bbox: Box,
    ocr_bbox: Box,
    role_hint: str | None,
    containers: list[Box],
) -> float:
    """Weighted composite score in [0, ~1]."""
    i = iou(raw_bbox, ocr_bbox)
    if i == 0:
        return 0.0
    bp = baseline_proximity(raw_bbox, ocr_bbox)
    sr = size_ratio(raw_bbox, ocr_bbox)
    ho = horizontal_overlap(raw_bbox, ocr_bbox)

    bonus = 0.0
    # Same-container bonus.
    for c in containers:
        if contains(c, raw_bbox) and contains(c, ocr_bbox):
            bonus += 0.1
            break

    # Role compatibility (text layers only care about text OCR).
    if role_hint in {"headline", "subheadline", "body_text", "button"}:
        bonus += 0.05

    return i * 0.45 + bp * 0.25 + sr * 0.15 + ho * 0.15 + bonus


def re_ocr_small_regions(
    rgb,
    unmatched_bboxes: list[tuple[int, Box]],
    ocr_engine,
    *,
    min_area: int = 800,
    orientation: bool = False,
    unwarping: bool = False,
) -> dict[int, str]:
    """Try OCR again on small bbox crops that missed the first-pass text.

    Returns dict of raw_index -> recognized_text.
    """
    out: dict[int, str] = {}
    if not unmatched_bboxes or ocr_engine is None or not ocr_engine.available:
        return out
    import numpy as np

    h, w = rgb.shape[:2]
    for ri, bb in unmatched_bboxes:
        if bb.w * bb.h < min_area:
            continue
        x1 = max(0, int(round(bb.x)))
        y1 = max(0, int(round(bb.y)))
        x2 = min(w, int(round(bb.x + bb.w)))
        y2 = min(h, int(round(bb.y + bb.h)))
        if x2 <= x1 + 8 or y2 <= y1 + 8:
            continue
        crop = rgb[y1:y2, x1:x2]
        # Upscale small crops to help OCR.
        scale = max(1, int(64 / max(1, y2 - y1)))
        if scale > 1:
            crop = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
        try:
            lines = ocr_engine.extract(crop, orientation=orientation, unwarping=unwarping)
        except Exception:
            continue
        if lines:
            text = " ".join(ln.text for ln in lines)
            out[ri] = text
    return out
