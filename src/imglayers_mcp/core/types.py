"""Shared dataclass types for the vision-driven decomposition pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..utils.bbox import Box

ElementKind = Literal["text", "image", "vector_like", "unknown"]


@dataclass
class RawLayer:
    """A pixel-cut layer produced from a vision element."""

    rgba: np.ndarray
    bbox: Box
    kind: ElementKind = "unknown"
    engine: str = "vision"
    confidence: float = 0.9
    z_index: int = 0
    debug: dict = field(default_factory=dict)


@dataclass
class VisionElement:
    """A single element in the layer tree, as provided by the vision model."""

    type: str  # "background" | "text" | "image" | "icon" | "button" | "card" | "badge" | "shape" | "decoration"
    bbox: Box
    name: str = ""
    color: str | None = None
    text_content: str | None = None
    font_size: float | None = None
    font_weight: int | None = None
    children: list[int] = field(default_factory=list)
