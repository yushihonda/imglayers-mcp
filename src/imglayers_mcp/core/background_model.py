"""Background / fill classification (spec WS4).

Given a RGB region that is believed to be a container background, classify
its fill type and extract parameters:

  - solid         : single color
  - linear_gradient: direction + endpoint colors
  - radial_gradient: center + edge colors
  - texture_like  : non-uniform but non-photo (patterns, noise)
  - photo_like    : high color variety and edge density

Used to choose the correct inpaint strategy for hole-fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

FillKind = Literal["solid", "linear_gradient", "radial_gradient", "texture_like", "photo_like"]


@dataclass
class BackgroundModel:
    kind: FillKind
    color: np.ndarray | None = None  # for solid
    color_a: np.ndarray | None = None  # gradient start (top or left)
    color_b: np.ndarray | None = None  # gradient end (bottom or right)
    direction: str | None = None  # "vertical" | "horizontal" for linear
    center_color: np.ndarray | None = None  # radial center
    edge_color: np.ndarray | None = None  # radial edge
    confidence: float = 0.0
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "color": _c(self.color),
            "color_a": _c(self.color_a),
            "color_b": _c(self.color_b),
            "direction": self.direction,
            "center_color": _c(self.center_color),
            "edge_color": _c(self.edge_color),
            "confidence": round(self.confidence, 3),
            "stats": self.stats,
        }


def _c(v: np.ndarray | None) -> list | None:
    return None if v is None else [int(x) for x in v]


def classify_region(rgb: np.ndarray) -> BackgroundModel:
    """Classify a region into a background fill model."""
    if rgb.size == 0 or rgb.ndim != 3:
        return BackgroundModel(kind="solid", color=np.array([128, 128, 128], dtype=np.uint8), confidence=0.0)

    h, w = rgb.shape[:2]
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    overall_std = float(pixels.std(axis=0).mean())

    # Solid: nearly uniform.
    if overall_std < 3.0:
        color = np.median(pixels, axis=0).astype(np.uint8)
        return BackgroundModel(
            kind="solid", color=color, confidence=0.95,
            stats={"std": overall_std},
        )

    # Gradient tests (sample edge strips).
    top = pixels.reshape(h, w, 3)[: max(1, h // 10)].reshape(-1, 3).mean(axis=0)
    bot = pixels.reshape(h, w, 3)[-max(1, h // 10):].reshape(-1, 3).mean(axis=0)
    left = pixels.reshape(h, w, 3)[:, : max(1, w // 10)].reshape(-1, 3).mean(axis=0)
    right = pixels.reshape(h, w, 3)[:, -max(1, w // 10):].reshape(-1, 3).mean(axis=0)

    vert_diff = float(np.linalg.norm(top - bot))
    horz_diff = float(np.linalg.norm(left - right))

    def _row_monotonic(axis: int) -> bool:
        """Check if mean color is monotonically changing along `axis`."""
        arr = pixels.reshape(h, w, 3).mean(axis=2)  # H×W luminance
        line = arr.mean(axis=1 - axis)  # aggregate across the other axis
        diffs = np.diff(line)
        non_monotonic = (np.sign(diffs) != np.sign(diffs[np.abs(diffs) > 0.5].mean() if (np.abs(diffs) > 0.5).any() else 0)).mean()
        return non_monotonic < 0.25

    # Linear gradient.
    if vert_diff > 15 and vert_diff > horz_diff * 1.2 and overall_std < 60:
        if _row_monotonic(0):
            return BackgroundModel(
                kind="linear_gradient",
                color_a=top.astype(np.uint8),
                color_b=bot.astype(np.uint8),
                direction="vertical",
                confidence=0.8,
                stats={"std": overall_std, "endpoint_diff": vert_diff},
            )
    if horz_diff > 15 and horz_diff > vert_diff * 1.2 and overall_std < 60:
        if _row_monotonic(1):
            return BackgroundModel(
                kind="linear_gradient",
                color_a=left.astype(np.uint8),
                color_b=right.astype(np.uint8),
                direction="horizontal",
                confidence=0.8,
                stats={"std": overall_std, "endpoint_diff": horz_diff},
            )

    # Radial gradient test: center vs corners distance.
    cy, cx = h // 2, w // 2
    cs = max(1, min(h, w) // 8)
    center_patch = pixels.reshape(h, w, 3)[max(0, cy - cs):cy + cs, max(0, cx - cs):cx + cs].reshape(-1, 3).mean(axis=0)
    corners = np.stack([top, bot, left, right], axis=0).mean(axis=0)
    rad_diff = float(np.linalg.norm(center_patch - corners))
    if rad_diff > 15 and overall_std < 70:
        return BackgroundModel(
            kind="radial_gradient",
            center_color=center_patch.astype(np.uint8),
            edge_color=corners.astype(np.uint8),
            confidence=0.6,
            stats={"std": overall_std, "center_edge_diff": rad_diff},
        )

    # Texture vs photo: use edge density + color count.
    gray = pixels.reshape(h, w, 3).mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    if gx.size > 0 and gy.size > 0:
        edge_density = float(((gx[:-1] > 20) | (gy[:, :-1] > 20)).mean())
    else:
        edge_density = 0.0

    q = (pixels.astype(np.int32) // 24) * 24
    packed = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    unique_count = int(np.unique(packed).size)

    if edge_density > 0.2 and unique_count > 2000:
        # Photo-like.
        return BackgroundModel(
            kind="photo_like",
            color=np.median(pixels, axis=0).astype(np.uint8),
            confidence=0.5,
            stats={"std": overall_std, "edge_density": edge_density, "unique_colors": unique_count},
        )

    # Fallback: texture-like.
    return BackgroundModel(
        kind="texture_like",
        color=np.median(pixels, axis=0).astype(np.uint8),
        confidence=0.4,
        stats={"std": overall_std, "edge_density": edge_density, "unique_colors": unique_count},
    )


def fill_hole(model: BackgroundModel, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill `hole` in `rgb` using the given background model."""
    from .inpaint_utils import (
        fill_linear_gradient,
        fill_radial_gradient,
        fill_solid,
        fill_texture_tile,
    )

    if model.kind == "solid" and model.color is not None:
        return fill_solid(rgb, hole, model.color)
    if model.kind == "linear_gradient" and model.color_a is not None and model.color_b is not None:
        return fill_linear_gradient(
            rgb, hole,
            direction=model.direction or "vertical",
            color_a=model.color_a,
            color_b=model.color_b,
        )
    if model.kind == "radial_gradient" and model.center_color is not None and model.edge_color is not None:
        return fill_radial_gradient(
            rgb, hole,
            center_color=model.center_color,
            edge_color=model.edge_color,
        )
    # texture_like / photo_like / fallback.
    return fill_texture_tile(rgb, hole)
