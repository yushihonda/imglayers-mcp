"""Connected-component labeling for char-level text splitting."""

from __future__ import annotations

import numpy as np


def connected_components(mask: np.ndarray) -> list[dict]:
    """Flood-fill based CC labeling (8-connectivity). Returns per-component dicts
    with keys: mask, area, y1, y2, x1, x2.
    """
    if not mask.any():
        return []
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    coords = np.argwhere(mask)
    for y0, x0 in coords:
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        pts_y: list[int] = []
        pts_x: list[int] = []
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            if visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            pts_y.append(y)
            pts_x.append(x)
            stack.extend([
                (y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1),
                (y + 1, x + 1), (y + 1, x - 1), (y - 1, x + 1), (y - 1, x - 1),
            ])
        if not pts_y:
            continue
        y_arr = np.asarray(pts_y)
        x_arr = np.asarray(pts_x)
        comp_mask = np.zeros_like(mask, dtype=bool)
        comp_mask[y_arr, x_arr] = True
        comps.append({
            "mask": comp_mask,
            "area": int(y_arr.size),
            "y1": int(y_arr.min()),
            "y2": int(y_arr.max()) + 1,
            "x1": int(x_arr.min()),
            "x2": int(x_arr.max()) + 1,
        })
    return comps
