"""Bounding-box geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.w, "height": self.h}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "Box":
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            w=float(d.get("width", d.get("w", 0.0))),
            h=float(d.get("height", d.get("h", 0.0))),
        )

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "Box":
        return cls(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1))


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def overlap_ratio(inner: Box, outer: Box) -> float:
    """How much of `inner` is inside `outer` (0..1)."""
    x1 = max(inner.x, outer.x)
    y1 = max(inner.y, outer.y)
    x2 = min(inner.x2, outer.x2)
    y2 = min(inner.y2, outer.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inner.area <= 0:
        return 0.0
    return inter / inner.area


def alpha_bbox(alpha: np.ndarray, threshold: int = 8) -> Box | None:
    """Tight bounding box of nonzero alpha pixels."""
    if alpha.ndim != 2:
        raise ValueError("alpha must be 2D")
    mask = alpha > threshold
    if not mask.any():
        return None
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    y1, y2 = int(ys[0]), int(ys[-1]) + 1
    x1, x2 = int(xs[0]), int(xs[-1]) + 1
    return Box.from_xyxy(x1, y1, x2, y2)


def union_box(boxes: Iterable[Box]) -> Box | None:
    boxes = list(boxes)
    if not boxes:
        return None
    x1 = min(b.x for b in boxes)
    y1 = min(b.y for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    return Box.from_xyxy(x1, y1, x2, y2)
