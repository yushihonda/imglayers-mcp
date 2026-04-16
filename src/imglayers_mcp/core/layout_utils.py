"""Layout helpers: bbox geometry, containment, reading-order, grid detection."""

from __future__ import annotations

from ..utils.bbox import Box


def center(b: Box) -> tuple[float, float]:
    return (b.x + b.w / 2, b.y + b.h / 2)


def center_distance(a: Box, b: Box) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def size_ratio(a: Box, b: Box) -> float:
    """min-over-max of the two bbox areas (1.0 = same size)."""
    aa = max(1.0, a.area)
    ba = max(1.0, b.area)
    return min(aa, ba) / max(aa, ba)


def baseline_proximity(a: Box, b: Box) -> float:
    """How close the vertical centers are, normalized by mean height."""
    a_cy = a.y + a.h / 2
    b_cy = b.y + b.h / 2
    mean_h = max(1.0, (a.h + b.h) / 2)
    return max(0.0, 1.0 - abs(a_cy - b_cy) / mean_h)


def horizontal_overlap(a: Box, b: Box) -> float:
    """Horizontal overlap width normalized by min width."""
    overlap = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    min_w = max(1.0, min(a.w, b.w))
    return max(0.0, overlap / min_w)


def contains(outer: Box, inner: Box, tolerance: float = 2.0) -> bool:
    return (
        outer.x - tolerance <= inner.x
        and outer.y - tolerance <= inner.y
        and outer.x + outer.w + tolerance >= inner.x + inner.w
        and outer.y + outer.h + tolerance >= inner.y + inner.h
    )


def same_container(a: Box, b: Box, candidates: list[Box]) -> bool:
    """Return True when some candidate container bbox encloses both a and b."""
    for c in candidates:
        if contains(c, a) and contains(c, b) and c.area > max(a.area, b.area):
            return True
    return False


def reading_order_key(b: Box) -> tuple[float, float]:
    """Top-to-bottom, then left-to-right."""
    return (round(b.y / 10) * 10, b.x)


def reading_order_consistency(a: Box, b: Box, role_a: str, role_b: str) -> float:
    """Bonus when headline is above body, etc."""
    if role_a in {"headline"} and role_b in {"subheadline", "body_text"}:
        return 1.0 if a.y < b.y else 0.3
    if role_a in {"subheadline"} and role_b in {"body_text"}:
        return 1.0 if a.y < b.y else 0.3
    return 0.5


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
