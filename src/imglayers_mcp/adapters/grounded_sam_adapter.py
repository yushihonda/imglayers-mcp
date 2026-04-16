"""Grounded-SAM adapter (spec WS5).

Heavy optional dependency. When `groundingdino` + `segment_anything` are
installed, this adapter performs text-prompted detection + mask generation
on a bounded region. When either is missing it reports `available == False`
and callers skip the tier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.bbox import Box
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RefinedMaskResult:
    mask: np.ndarray
    bbox: Box
    score: float
    prompt: str = ""


class GroundedSAMAdapter:
    """Optional backend. Kept as a stub body in v0.1; upgrade path documented."""

    def __init__(self) -> None:
        self._available = False
        self._error: str | None = None
        self._predictor = None  # SAM predictor
        self._gd_model = None  # GroundingDINO model

    def _ensure(self) -> None:
        if self._available or self._error is not None:
            return
        try:
            import groundingdino  # type: ignore  # noqa: F401
            from segment_anything import sam_model_registry  # type: ignore  # noqa: F401
            self._available = True
            log.info("Grounded-SAM adapter dependencies present")
        except Exception as exc:  # pragma: no cover — optional
            self._error = str(exc)
            log.info("Grounded-SAM unavailable: %s", exc)

    @property
    def available(self) -> bool:
        self._ensure()
        return self._available

    def refine(
        self,
        image: np.ndarray,
        *,
        prompt: str,
        bbox: Box,
    ) -> RefinedMaskResult | None:  # pragma: no cover — optional
        self._ensure()
        if not self._available:
            return None
        # To actually run:
        #   1. crop = image[bbox + padding]
        #   2. run GroundingDINO(crop, prompt) → detections
        #   3. take the top-1 box, feed to SAM predictor → mask
        #   4. rebase mask to full-canvas coordinates
        raise NotImplementedError(
            "Grounded-SAM runtime wiring is not bundled by default. "
            "Install `groundingdino-py` + `segment-anything` (+ model weights) "
            "and extend this method to enable heavy retry."
        )
