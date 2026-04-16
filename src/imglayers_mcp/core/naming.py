"""Naming and semantic-role inference (spec §11)."""

from __future__ import annotations

from collections import defaultdict

from ..models.manifest import LayerType, SemanticRole
from ..utils.bbox import Box


ROLE_PREFIX: dict[SemanticRole, str] = {
    "background": "bg",
    "headline": "headline",
    "subheadline": "subheadline",
    "body_text": "body",
    "button": "button",
    "card": "card",
    "icon": "icon",
    "logo": "logo",
    "product_image": "image",
    "illustration": "image",
    "decoration": "deco",
    "unknown": "unknown",
    "text_group": "textgroup",
    "container_group": "container",
    "sibling_cluster": "cluster",
    "row_cluster": "row",
}


class NameAllocator:
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)

    def allocate(self, role: str) -> str:
        prefix = ROLE_PREFIX.get(role, "unknown")
        self._counters[prefix] += 1
        return f"{prefix}_{self._counters[prefix]:03d}"


def infer_semantic_role(
    *,
    layer_type: LayerType,
    bbox: Box,
    canvas_w: int,
    canvas_h: int,
    text_content: str | None,
    area_rank: int,
    text_size_rank: int | None,
    nearby_text_below_headline: bool,
    is_full_canvas: bool,
    contains_text: bool = False,
) -> SemanticRole:
    if is_full_canvas and layer_type in ("image", "vector_like"):
        return "background"

    if layer_type == "text" and text_content is not None:
        if text_size_rank == 0:
            return "headline"
        if nearby_text_below_headline:
            return "subheadline"
        if len(text_content) <= 14 and bbox.w <= canvas_w * 0.35 and bbox.h <= canvas_h * 0.12:
            return "button"
        if len(text_content) > 40:
            return "body_text"
        return "body_text"

    # Non-text role inference.
    area_ratio = (bbox.w * bbox.h) / max(1.0, canvas_w * canvas_h)
    aspect = bbox.w / max(1.0, bbox.h)

    if layer_type == "vector_like":
        compact = min(bbox.w, bbox.h) / max(bbox.w, bbox.h, 1)
        if area_ratio <= 0.025 and compact >= 0.5:
            return "icon"
        if bbox.h <= canvas_h * 0.12 and 2.0 <= aspect <= 10.0 and bbox.w <= canvas_w * 0.5:
            return "button"
        if contains_text and area_ratio >= 0.05:
            return "card"
        if area_ratio >= 0.04:
            return "illustration"
        return "decoration"

    if layer_type == "image":
        # Very elongated strips = decorative bars/lines.
        if aspect > 15.0 or aspect < 1.0 / 15.0:
            return "decoration"
        if (
            bbox.w <= canvas_w * 0.25
            and bbox.h <= canvas_h * 0.12
            and bbox.y <= canvas_h * 0.2
            and not contains_text
        ):
            return "logo"
        if contains_text and area_ratio >= 0.04:
            return "card"
        # Short wide rect = button-shape.
        if (
            bbox.h <= canvas_h * 0.15
            and bbox.w <= canvas_w * 0.5
            and 2.0 <= aspect <= 10.0
        ):
            return "button"
        if bbox.w >= canvas_w * 0.4 and bbox.h >= canvas_h * 0.4:
            return "product_image"
        return "illustration"

    if layer_type == "unknown":
        compact = min(bbox.w, bbox.h) / max(bbox.w, bbox.h, 1)
        if aspect > 15.0 or aspect < 1.0 / 15.0:
            return "decoration"
        if area_ratio <= 0.025 and compact >= 0.5:
            return "icon"
        if bbox.h <= canvas_h * 0.15 and 2.0 <= aspect <= 10.0 and bbox.w <= canvas_w * 0.5:
            return "button"
        if contains_text:
            return "card"
        return "illustration"

    return "unknown"


def is_full_canvas(bbox: Box, canvas_w: int, canvas_h: int, tolerance: float = 0.02) -> bool:
    return (
        bbox.x <= canvas_w * tolerance
        and bbox.y <= canvas_h * tolerance
        and bbox.x + bbox.w >= canvas_w * (1.0 - tolerance)
        and bbox.y + bbox.h >= canvas_h * (1.0 - tolerance)
    )
