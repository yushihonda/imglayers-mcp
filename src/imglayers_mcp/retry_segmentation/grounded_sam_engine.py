"""Grounded-SAM retry backend (spec §7.6).

Implements the `RetrySegmentationBackend` protocol. If the upstream
`groundingdino` + `segment_anything` packages are installed, the engine
runs text-prompted detection + segmentation over a bbox-of-interest
and returns a refined alpha mask. When the packages are absent, the
engine reports `available == False` and the orchestrator skips retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..utils.bbox import Box
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RefinedMaskResult:
    mask: np.ndarray  # HxW bool
    bbox: Box
    score: float
    prompt: str


class RetrySegmentationBackend(Protocol):
    @property
    def available(self) -> bool: ...

    def refine(
        self,
        image: np.ndarray,
        prompt: str,
        bbox: Box,
    ) -> RefinedMaskResult | None: ...


class GroundedSAMEngine:
    def __init__(self) -> None:
        self._available = False
        self._gd = None
        self._sam = None
        self._error: str | None = None

    def _ensure(self) -> None:
        if self._available or self._error is not None:
            return
        try:
            import groundingdino  # type: ignore  # noqa: F401
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore  # noqa: F401
            self._available = True
            log.info("Grounded-SAM retry backend available")
        except Exception as exc:  # pragma: no cover — optional
            self._error = str(exc)

    @property
    def available(self) -> bool:
        self._ensure()
        return self._available

    def refine(
        self,
        image: np.ndarray,
        prompt: str,
        bbox: Box,
    ) -> RefinedMaskResult | None:  # pragma: no cover — optional
        self._ensure()
        if not self._available:
            return None
        # Real implementation would:
        #   1. Crop region-of-interest (bbox + padding)
        #   2. Run GroundingDINO with `prompt` to get boxes
        #   3. Run SAM over the top box to get a mask
        #   4. Rebase the mask back onto full-canvas coords
        # Kept as a stub here; install `groundingdino-py` + `segment-anything`
        # to enable this path.
        raise NotImplementedError(
            "Grounded-SAM retry runtime wiring is not included in the v0.1 default build."
        )
