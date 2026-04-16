"""PaddleOCR adapter — optional dependency for text detection + recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.bbox import Box
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class OCRLine:
    text: str
    bbox: Box
    confidence: float
    language: str | None = None


class PaddleOCRAdapter:
    def __init__(self, lang: str = "japan") -> None:
        self._lang = lang
        self._ocr: Any = None
        self._available = False
        self._init_error: str | None = None

    def _ensure(self) -> None:
        if self._ocr is not None or self._init_error is not None:
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
            # Note: we keep doc_orientation_classify + doc_unwarping OFF because
            # they can subtly warp bboxes on flat design images (introducing
            # e.g. x=0 offsets). Turn them on only via env flag if needed.
            try:
                self._ocr = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    lang=self._lang,
                )
            except TypeError:
                try:
                    self._ocr = PaddleOCR(use_textline_orientation=True, lang=self._lang)
                except TypeError:
                    self._ocr = PaddleOCR(use_angle_cls=True, lang=self._lang, show_log=False)
            self._available = True
            log.info("PaddleOCR adapter ready (lang=%s)", self._lang)
        except Exception as exc:
            self._init_error = str(exc)
            log.info("PaddleOCR unavailable: %s", exc)

    @property
    def available(self) -> bool:
        self._ensure()
        return self._available

    def extract(self, rgb: np.ndarray) -> list[OCRLine]:
        self._ensure()
        if not self._available:
            return []
        try:
            try:
                result = self._ocr.predict(rgb)
            except (AttributeError, TypeError):
                result = self._ocr.ocr(rgb, cls=True)
        except Exception as exc:
            log.warning("PaddleOCR extract failed: %s", exc)
            return []
        raw_lines = _normalize(result, self._lang)
        merged = _merge_into_lines(raw_lines)
        return _pad_bboxes(merged)


def _normalize(result: Any, lang: str) -> list[OCRLine]:
    lines: list[OCRLine] = []
    if result is None:
        return lines
    if isinstance(result, list) and result and isinstance(result[0], dict):
        for page in result:
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or [1.0] * len(texts)
            polys = page.get("rec_polys") or page.get("dt_polys") or []
            for i, text in enumerate(texts):
                if not text:
                    continue
                poly = polys[i] if i < len(polys) else None
                lines.append(OCRLine(
                    text=str(text),
                    bbox=_poly_to_box(poly),
                    confidence=float(scores[i]) if i < len(scores) else 1.0,
                    language=lang,
                ))
        return lines
    if isinstance(result, list):
        for page in result:
            if page is None:
                continue
            for entry in page:
                try:
                    poly, recog = entry[0], entry[1]
                    text, score = recog[0], recog[1]
                except (IndexError, TypeError):
                    continue
                if not text:
                    continue
                lines.append(OCRLine(
                    text=str(text),
                    bbox=_poly_to_box(poly),
                    confidence=float(score),
                    language=lang,
                ))
    return lines


def _pad_bboxes(lines: list[OCRLine]) -> list[OCRLine]:
    """Pad OCR bboxes slightly to better match visual text bounding boxes.

    PaddleOCR returns bbox around the stroke pixels. Most design tools
    draw the text box with ~10-15% padding (ascender/descender space),
    so adding proportional padding brings our bboxes closer to what a
    designer considers the element's frame.
    """
    padded: list[OCRLine] = []
    for ln in lines:
        pad_x = ln.bbox.h * 0.08
        pad_y = ln.bbox.h * 0.15
        new_bbox = Box(
            x=max(0.0, ln.bbox.x - pad_x),
            y=max(0.0, ln.bbox.y - pad_y),
            w=ln.bbox.w + pad_x * 2,
            h=ln.bbox.h + pad_y * 2,
        )
        padded.append(OCRLine(text=ln.text, bbox=new_bbox, confidence=ln.confidence, language=ln.language))
    return padded


def _merge_into_lines(words: list[OCRLine]) -> list[OCRLine]:
    """Merge word-level OCR output into line-level entries.

    Two-pass clustering:
      1. Group words into horizontal rows (by y-center proximity)
      2. Within each row, merge adjacent words (by x-gap)
    """
    if not words:
        return []

    # Pass 1: sort by y, then group into rows.
    sorted_by_y = sorted(words, key=lambda w: w.bbox.y + w.bbox.h / 2)
    rows: list[list[OCRLine]] = []
    for w in sorted_by_y:
        w_cy = w.bbox.y + w.bbox.h / 2
        placed = False
        for row in rows:
            # Check against all words in the row, not just last (layout may vary).
            for ref in row:
                ref_cy = ref.bbox.y + ref.bbox.h / 2
                # Row tolerance: 50% of the smaller word's height.
                tol = min(ref.bbox.h, w.bbox.h) * 0.5
                if abs(ref_cy - w_cy) <= tol:
                    row.append(w)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            rows.append([w])

    # Pass 2: within each row, merge words that are horizontally adjacent.
    merged: list[OCRLine] = []
    for row in rows:
        row.sort(key=lambda w: w.bbox.x)
        # Chain words that are close in x.
        chains: list[list[OCRLine]] = []
        for w in row:
            if not chains:
                chains.append([w])
                continue
            prev = chains[-1][-1]
            gap = w.bbox.x - (prev.bbox.x + prev.bbox.w)
            # Allow up to ~3× space between words to keep them in same chain
            # (generous: handles wide letter-spacing, columns are split on bigger gaps).
            max_gap = max(prev.bbox.h, w.bbox.h) * 1.2
            if gap > max_gap:
                chains.append([w])
            else:
                chains[-1].append(w)

        for chain in chains:
            if len(chain) == 1:
                merged.append(chain[0])
            else:
                text = " ".join(c.text for c in chain)
                x1 = min(c.bbox.x for c in chain)
                y1 = min(c.bbox.y for c in chain)
                x2 = max(c.bbox.x + c.bbox.w for c in chain)
                y2 = max(c.bbox.y + c.bbox.h for c in chain)
                conf = sum(c.confidence for c in chain) / len(chain)
                merged.append(OCRLine(
                    text=text,
                    bbox=Box.from_xyxy(x1, y1, x2, y2),
                    confidence=conf,
                    language=chain[0].language,
                ))
    return merged


def _poly_to_box(poly: Any) -> Box:
    if poly is None:
        return Box(0, 0, 0, 0)
    pts = np.asarray(poly, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return Box(0, 0, 0, 0)
    x1 = float(pts[:, 0].min())
    y1 = float(pts[:, 1].min())
    x2 = float(pts[:, 0].max())
    y2 = float(pts[:, 1].max())
    return Box.from_xyxy(x1, y1, x2, y2)
