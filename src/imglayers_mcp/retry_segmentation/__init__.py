"""Stage 5: retry segmentation (spec §7.6).

Optional refinement backend (Grounded-SAM) for ambiguous layers.
Not shipped by default; the orchestrator detects availability at runtime.
"""

from .grounded_sam_engine import (
    GroundedSAMEngine,
    RefinedMaskResult,
    RetrySegmentationBackend,
)
from .cv_refinement import refine_by_cc

# New v2 tier-ladder API lives in core/retry_segmentation.py.
# Import here so `from ..retry_segmentation import refine` still works.
from ..core.retry_segmentation import (
    refine as refine_tiered,
    refine_cc,
    refine_edge,
    refine_mask_morph,
)

__all__ = [
    "GroundedSAMEngine",
    "RefinedMaskResult",
    "RetrySegmentationBackend",
    "refine_by_cc",
    "refine_tiered",
    "refine_cc",
    "refine_edge",
    "refine_mask_morph",
]
