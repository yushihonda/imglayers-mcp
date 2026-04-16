"""Line-level grouping for OCR output (spec §7.4)."""

from __future__ import annotations

from ..utils.bbox import Box
from .paddleocr_engine import OCRLine


def cluster_into_lines(words: list[OCRLine]) -> list[OCRLine]:
    """Merge word-level OCR output into line-level entries by row proximity."""
    if not words:
        return []
    sorted_by_y = sorted(words, key=lambda w: w.bbox.y + w.bbox.h / 2)
    rows: list[list[OCRLine]] = []
    for w in sorted_by_y:
        w_cy = w.bbox.y + w.bbox.h / 2
        placed = False
        for row in rows:
            for ref in row:
                ref_cy = ref.bbox.y + ref.bbox.h / 2
                tol = min(ref.bbox.h, w.bbox.h) * 0.5
                if abs(ref_cy - w_cy) <= tol:
                    row.append(w)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            rows.append([w])

    merged: list[OCRLine] = []
    for row in rows:
        row.sort(key=lambda w: w.bbox.x)
        chains: list[list[OCRLine]] = []
        for w in row:
            if not chains:
                chains.append([w])
                continue
            prev = chains[-1][-1]
            gap = w.bbox.x - (prev.bbox.x + prev.bbox.w)
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
