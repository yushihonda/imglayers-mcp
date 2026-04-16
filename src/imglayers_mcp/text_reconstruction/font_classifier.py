"""Font classifier abstraction (spec §7.5).

Three backends:
  - KnownFontsBackend: rank user-supplied font list by rerender fit
  - HeuristicBackend: fallback using stroke width / x-height features
  - (DeepFontLikeBackend: placeholder; not implemented in v0.1)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class FontCandidate:
    family: str
    weight: int = 400
    score: float = 0.0
    path: str | None = None


@dataclass
class FontConfig:
    known_fonts: list[str]
    top_k: int = 5


class FontClassifier(Protocol):
    """Return ranked font candidates for a given text + image crop."""

    def rank_candidates(
        self,
        text_crop: np.ndarray,
        text: str,
        cfg: FontConfig,
    ) -> list[FontCandidate]:
        ...


class KnownFontsBackend:
    """Rank user-supplied fonts by rerender fit score (implemented by rerender_fitter)."""

    def __init__(self, fitter=None) -> None:
        from .rerender_fitter import RerenderFitter
        self._fitter = fitter or RerenderFitter()

    def rank_candidates(
        self,
        text_crop: np.ndarray,
        text: str,
        cfg: FontConfig,
    ) -> list[FontCandidate]:
        if not cfg.known_fonts:
            return []
        ranked: list[FontCandidate] = []
        for family in cfg.known_fonts:
            for weight in (400, 700):
                result = self._fitter.try_fit(text_crop, text, family, weight=weight)
                if result is None:
                    continue
                ranked.append(FontCandidate(
                    family=family, weight=weight, score=result.score, path=result.font_path,
                ))
        ranked.sort(key=lambda c: -c.score)
        return ranked[: cfg.top_k]


class HeuristicBackend:
    """Aspect-based fallback: guesses sans-serif vs serif from stroke variance."""

    def rank_candidates(
        self,
        text_crop: np.ndarray,
        text: str,
        cfg: FontConfig,
    ) -> list[FontCandidate]:
        if text_crop.size == 0:
            return [FontCandidate(family="sans-serif", weight=400, score=0.1)]
        gray = text_crop[..., :3].astype(np.int32).sum(axis=2) / 3 if text_crop.ndim == 3 else text_crop
        stroke_var = float(np.std(gray))
        # Heavy stroke variance → display/serif; flat → sans-serif.
        if stroke_var > 80:
            fam = "serif"
        else:
            fam = "sans-serif"
        return [FontCandidate(family=fam, weight=400, score=0.2)]
