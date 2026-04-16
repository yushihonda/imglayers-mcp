"""Rerender fitting: score a font candidate by rendering the text and
comparing with the original crop (spec §7.5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FitResult:
    score: float
    rendered_image: np.ndarray
    font_path: str | None
    estimated_size: float


class RerenderFitter:
    """Render `text` with a candidate font and score similarity to the crop.

    Simple but effective metrics:
      - dark-pixel overlap (IoU between rendered strokes and observed strokes)
      - width fit (rendered text width vs crop width)
    """

    def __init__(self) -> None:
        self._font_cache: dict[tuple[str, int, int], ImageFont.FreeTypeFont] = {}
        self._path_cache: dict[tuple[str, int], str | None] = {}

    def try_fit(
        self,
        text_crop: np.ndarray,
        text: str,
        family: str,
        *,
        weight: int = 400,
    ) -> FitResult | None:
        if text_crop.size == 0 or not text:
            return None
        h, w = text_crop.shape[:2]
        # Estimate target font size from crop height (~ cap height + descender).
        target_size = max(8, int(h * 0.85))
        font = self._load(family, weight, target_size)
        if font is None:
            return None

        # Rasterize the text.
        img = Image.new("L", (w, h), color=255)
        draw = ImageDraw.Draw(img)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (w - tw) // 2 - bbox[0]
            y = (h - th) // 2 - bbox[1]
            draw.text((x, y), text, fill=0, font=font)
        except Exception:
            return None
        rendered_arr = np.asarray(img)

        # Score: overlap between rendered strokes and observed dark pixels.
        score = _overlap_score(rendered_arr, text_crop)

        return FitResult(
            score=score,
            rendered_image=rendered_arr,
            font_path=self._path_cache.get((family, weight)),
            estimated_size=float(target_size),
        )

    def _load(self, family: str, weight: int, size: int) -> ImageFont.FreeTypeFont | None:
        key = (family, weight, size)
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        path = self._find_path(family, weight)
        if path is None:
            return None
        try:
            font = ImageFont.truetype(path, size)
            self._font_cache[key] = font
            return font
        except Exception:
            return None

    def _find_path(self, family: str, weight: int) -> str | None:
        """Locate a font file on disk. Cached. macOS/Linux/Win fallback search."""
        key = (family, weight)
        if key in self._path_cache:
            return self._path_cache[key]
        search_roots = [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path("/usr/share/fonts"),
            Path.home() / "Library/Fonts",
            Path("C:/Windows/Fonts"),
        ]
        lowered = family.lower().replace(" ", "").replace("-", "")
        weight_tokens = ["bold"] if weight >= 600 else ["regular", "medium", ""]
        found: str | None = None
        for root in search_roots:
            if not root.exists():
                continue
            for ext in ("*.ttc", "*.ttf", "*.otf"):
                for p in root.rglob(ext):
                    name = p.stem.lower().replace(" ", "").replace("-", "")
                    if lowered in name:
                        for w_token in weight_tokens:
                            if (w_token and w_token in name) or not w_token:
                                found = str(p)
                                break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        self._path_cache[key] = found
        return found


def _overlap_score(rendered_gray: np.ndarray, original: np.ndarray) -> float:
    """IoU between dark regions in rendered text and original text crop."""
    if original.ndim == 3:
        gray_orig = original[..., :3].astype(np.int32).sum(axis=2) / 3
    else:
        gray_orig = original.astype(np.float32)
    # Threshold: dark pixels are "on".
    rendered_mask = rendered_gray < 128
    orig_thresh = np.median(gray_orig)
    orig_mask = gray_orig < orig_thresh - 30
    if not rendered_mask.any() or not orig_mask.any():
        return 0.0
    inter = np.logical_and(rendered_mask, orig_mask).sum()
    union = np.logical_or(rendered_mask, orig_mask).sum()
    return float(inter / union) if union > 0 else 0.0
