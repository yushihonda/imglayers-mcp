"""Stage 5: retry segmentation (spec §7.6).

Optional refinement backend (Grounded-SAM) for ambiguous layers.
Not shipped by default; the orchestrator detects availability at runtime.
"""

from .grounded_sam_engine import (
    GroundedSAMEngine,
    RefinedMaskResult,
    RetrySegmentationBackend,
)

__all__ = [
    "GroundedSAMEngine",
    "RefinedMaskResult",
    "RetrySegmentationBackend",
]
