"""Post-process SAM2 mask candidates into design-oriented layer groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.mask_candidate import MaskCandidate
from ..utils.bbox import Box
from .layout_utils import iou


@dataclass
class FilterMergeConfig:
    min_area_ratio: float = 0.0005
    iou_suppress: float = 0.85
    inclusion_ratio: float = 0.92
    merge_color_delta: float = 12.0
    max_candidates: int = 96


def process(
    candidates: list[MaskCandidate],
    rgb: np.ndarray,
    image_type: str,
    cfg: FilterMergeConfig | None = None,
) -> tuple[list[MaskCandidate], MaskCandidate | None]:
    cfg = cfg or _cfg_for(image_type)
    h, w = rgb.shape[:2]
    if not candidates:
        return [], None

    min_area = int(h * w * cfg.min_area_ratio)
    kept = [c for c in candidates if c.area >= min_area]
    kept.sort(key=lambda c: -c.area)
    kept = kept[: cfg.max_candidates]

    kept = _suppress_near_duplicates(kept, cfg.iou_suppress)
    kept = _absorb_inclusions(kept, cfg.inclusion_ratio)
    kept = _adjacency_merge(kept, rgb, cfg.merge_color_delta)

    bg, remaining = _split_background(kept, h, w)
    remaining = _remove_contained_small_copies(remaining, cfg.inclusion_ratio)
    remaining.sort(key=lambda c: -c.area)
    return remaining, bg


def _cfg_for(image_type: str) -> FilterMergeConfig:
    if image_type in ("ui_mock", "banner"):
        return FilterMergeConfig(
            min_area_ratio=0.0008,
            iou_suppress=0.8,
            inclusion_ratio=0.9,
            merge_color_delta=14.0,
            max_candidates=96,
        )
    if image_type in ("photo_mixed", "illustration"):
        return FilterMergeConfig(
            min_area_ratio=0.0005,
            iou_suppress=0.85,
            inclusion_ratio=0.92,
            merge_color_delta=11.0,
            max_candidates=120,
        )
    return FilterMergeConfig()


def _suppress_near_duplicates(
    items: list[MaskCandidate], iou_threshold: float
) -> list[MaskCandidate]:
    kept: list[MaskCandidate] = []
    for cand in items:
        dup = False
        for k in kept:
            if iou(cand.bbox, k.bbox) > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


def _absorb_inclusions(
    items: list[MaskCandidate], ratio_threshold: float
) -> list[MaskCandidate]:
    out: list[MaskCandidate] = []
    for small in items:
        absorbed = False
        for big in out:
            if big.area < small.area:
                continue
            if _contained_ratio(small.mask, big.mask) >= ratio_threshold:
                absorbed = True
                break
        if not absorbed:
            out.append(small)
    return out


def _adjacency_merge(
    items: list[MaskCandidate], rgb: np.ndarray, color_delta: float
) -> list[MaskCandidate]:
    if len(items) < 2:
        return items
    colors = [_mean_color(c, rgb) for c in items]
    merged = [False] * len(items)
    out: list[MaskCandidate] = []
    for i, cand in enumerate(items):
        if merged[i]:
            continue
        group = [cand]
        for j in range(i + 1, len(items)):
            if merged[j]:
                continue
            if _are_adjacent(cand.mask, items[j].mask):
                if np.linalg.norm(colors[i] - colors[j]) <= color_delta:
                    group.append(items[j])
                    merged[j] = True
        if len(group) == 1:
            out.append(cand)
        else:
            out.append(_combine(group))
    return out


def _split_background(
    items: list[MaskCandidate], h: int, w: int
) -> tuple[MaskCandidate | None, list[MaskCandidate]]:
    """Separate out a full-canvas-like mask as the background candidate."""
    if not items:
        return None, []
    total = h * w
    biggest = items[0]
    if biggest.area >= 0.7 * total:
        return biggest, items[1:]
    full_mask = np.zeros((h, w), dtype=bool)
    for c in items:
        full_mask |= c.mask
    residual_area = total - int(full_mask.sum())
    if residual_area > 0.4 * total:
        residual = ~full_mask
        ys, xs = np.nonzero(residual)
        if ys.size > 0:
            bbox = Box(
                x=float(xs.min()), y=float(ys.min()),
                w=float(xs.max() - xs.min() + 1),
                h=float(ys.max() - ys.min() + 1),
            )
            bg = MaskCandidate(
                mask=residual,
                bbox=bbox,
                area=int(residual.sum()),
                score=1.0,
                source="sam2",
                notes=["synthesized_background"],
            )
            return bg, items
    return None, items


def _remove_contained_small_copies(
    items: list[MaskCandidate], ratio_threshold: float
) -> list[MaskCandidate]:
    out: list[MaskCandidate] = []
    for cand in sorted(items, key=lambda c: -c.area):
        nested = False
        for prev in out:
            if prev.area <= cand.area:
                continue
            if _contained_ratio(cand.mask, prev.mask) > ratio_threshold:
                bbox_ratio = cand.bbox.area / max(1.0, prev.bbox.area)
                if bbox_ratio < 0.5:
                    continue
                nested = True
                break
        if not nested:
            out.append(cand)
    return out


def _contained_ratio(small: np.ndarray, big: np.ndarray) -> float:
    small_area = int(small.sum())
    if small_area == 0:
        return 0.0
    inter = int(np.logical_and(small, big).sum())
    return inter / small_area


def _are_adjacent(a: np.ndarray, b: np.ndarray, radius: int = 2) -> bool:
    ring = a.copy()
    for _ in range(radius):
        ring[1:] |= a[:-1]
        ring[:-1] |= a[1:]
        ring[:, 1:] |= a[:, :-1]
        ring[:, :-1] |= a[:, 1:]
    return bool(np.logical_and(ring, b).any())


def _mean_color(cand: MaskCandidate, rgb: np.ndarray) -> np.ndarray:
    pixels = rgb[cand.mask]
    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    return pixels.astype(np.float32).mean(axis=0)


def _combine(group: list[MaskCandidate]) -> MaskCandidate:
    mask = np.zeros_like(group[0].mask)
    for c in group:
        mask |= c.mask
    ys, xs = np.nonzero(mask)
    bbox = Box(
        x=float(xs.min()), y=float(ys.min()),
        w=float(xs.max() - xs.min() + 1),
        h=float(ys.max() - ys.min() + 1),
    )
    return MaskCandidate(
        mask=mask,
        bbox=bbox,
        area=int(mask.sum()),
        score=max(c.score for c in group),
        stability=max(c.stability for c in group),
        predicted_iou=max(c.predicted_iou for c in group),
        source=group[0].source,
        checkpoint=group[0].checkpoint,
        device=group[0].device,
        notes=["merged_adjacent"],
    )
