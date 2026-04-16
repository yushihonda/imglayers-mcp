"""Estimate rendering order for mask candidates.

Rules (cheap but effective for flat design images):
  1. Background candidate always sits at z=0.
  2. Larger area => lower z-index (earlier in paint order).
  3. When two masks overlap heavily, the one whose mean color is more
     extreme than the ring gets the higher z-index.
  4. Text-like small masks float to the top.
"""

from __future__ import annotations

import numpy as np

from ..models.mask_candidate import MaskCandidate


def estimate(
    foreground: list[MaskCandidate],
    background: MaskCandidate | None,
    rgb: np.ndarray,
) -> list[tuple[int, MaskCandidate]]:
    """Return [(z_index, candidate), ...] ordered by z_index ascending."""
    zorder: list[tuple[int, MaskCandidate]] = []
    z = 0
    if background is not None:
        zorder.append((z, background))
        z += 1

    sorted_fg = sorted(foreground, key=lambda c: -c.area)

    def _rel_score(cand: MaskCandidate) -> float:
        small_bonus = 0.0
        if cand.area < 2000:
            small_bonus += 0.1
        aspect = cand.bbox.w / max(1.0, cand.bbox.h)
        if aspect > 3.0 and cand.bbox.h < 80:
            small_bonus += 0.2
        return small_bonus

    sorted_fg.sort(key=lambda c: (-c.area, -_rel_score(c)))
    for c in sorted_fg:
        zorder.append((z, c))
        z += 1
    return zorder
