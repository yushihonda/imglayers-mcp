"""Stage 3: OCR extraction (spec §7.4)."""

from .paddleocr_engine import OCRLine, PaddleOCRAdapter as PaddleOCREngine
from .line_grouping import cluster_into_lines

__all__ = ["OCRLine", "PaddleOCREngine", "cluster_into_lines"]
