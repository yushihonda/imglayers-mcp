"""Internal representation for per-mask outputs from any segmentation backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..utils.bbox import Box

MaskSource = Literal["sam2", "layerd", "manual", "grounded_sam2"]


@dataclass
class MaskCandidate:
    mask: np.ndarray
    bbox: Box
    area: int
    score: float = 0.0
    stability: float = 0.0
    predicted_iou: float = 0.0
    source: MaskSource = "sam2"
    checkpoint: str | None = None
    device: str | None = None
    crop_region: tuple[int, int, int, int] | None = None
    notes: list[str] = field(default_factory=list)

    def to_debug_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_dict(),
            "area": int(self.area),
            "score": float(self.score),
            "stability": float(self.stability),
            "predicted_iou": float(self.predicted_iou),
            "source": self.source,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "notes": self.notes,
        }
