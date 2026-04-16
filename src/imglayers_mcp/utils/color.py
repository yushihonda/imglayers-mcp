"""Color helpers: hex conversion, palette extraction, dominant color."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = [int(max(0, min(255, c))) for c in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def dominant_colors(rgb_pixels: np.ndarray, k: int = 5) -> list[str]:
    """Quick quantized palette via uniform bucketing. No sklearn dep."""
    if rgb_pixels.size == 0:
        return []
    if rgb_pixels.ndim == 3:
        rgb_pixels = rgb_pixels.reshape(-1, rgb_pixels.shape[-1])
    rgb_pixels = rgb_pixels[:, :3].astype(np.int32)
    q = (rgb_pixels // 16) * 16 + 8
    packed = (q[:, 0].astype(np.int64) << 16) | (q[:, 1].astype(np.int64) << 8) | q[:, 2].astype(np.int64)
    unique, counts = np.unique(packed, return_counts=True)
    order = np.argsort(-counts)
    palette: list[str] = []
    for idx in order[:k]:
        p = int(unique[idx])
        r = (p >> 16) & 0xFF
        g = (p >> 8) & 0xFF
        b = p & 0xFF
        palette.append(rgb_to_hex((r, g, b)))
    return palette


def average_color(rgb_pixels: np.ndarray) -> str | None:
    if rgb_pixels.size == 0:
        return None
    flat = rgb_pixels.reshape(-1, rgb_pixels.shape[-1])[:, :3]
    mean = flat.mean(axis=0)
    return rgb_to_hex(mean.tolist())


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def extract_foreground_color(
    rgb_patch: np.ndarray, bg_hex: str | None = None
) -> str | None:
    """Pick the dominant non-background color from a patch.

    Strategy: bucket the patch into a coarse palette, then choose the bucket
    that is *furthest* from the supplied background color (or from the patch's
    own border average if bg is unknown). This recovers true text/foreground
    color even when the bbox is mostly background pixels.
    """
    if rgb_patch.size == 0:
        return None
    flat = rgb_patch.reshape(-1, 3).astype(np.int32)
    if bg_hex is not None:
        bg = np.array(hex_to_rgb(bg_hex), dtype=np.int32)
    else:
        h, w = rgb_patch.shape[:2]
        bw = max(1, min(h, w) // 8)
        border = np.concatenate(
            [
                rgb_patch[:bw].reshape(-1, 3),
                rgb_patch[-bw:].reshape(-1, 3),
                rgb_patch[:, :bw].reshape(-1, 3),
                rgb_patch[:, -bw:].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.int32)
        bg = border.mean(axis=0).round().astype(np.int32)

    q = (flat // 24) * 24 + 12
    packed = (q[:, 0].astype(np.int64) << 16) | (q[:, 1].astype(np.int64) << 8) | q[:, 2].astype(np.int64)
    unique, counts = np.unique(packed, return_counts=True)
    if unique.size == 0:
        return None
    cols = np.stack(
        [(unique >> 16) & 0xFF, (unique >> 8) & 0xFF, unique & 0xFF], axis=1
    ).astype(np.int32)
    dist_from_bg = np.linalg.norm(cols - bg, axis=1)
    # Score: must be *both* far from bg AND have non-trivial pixel mass.
    # Otherwise a single saturated outlier pixel wins. Weight by sqrt(count)
    # so common-but-distinct colors beat rare-but-extreme outliers.
    score = dist_from_bg * np.sqrt(counts.astype(np.float64))
    # Require some minimum distance to count as foreground at all.
    if dist_from_bg.max() < 24:
        return None
    winner = int(np.argmax(score))
    r, g, b = int(cols[winner, 0]), int(cols[winner, 1]), int(cols[winner, 2])
    return rgb_to_hex((r, g, b))


def detect_solid_or_gradient_fill(rgb_patch: np.ndarray) -> dict | None:
    """Classify a region as solid color, vertical/horizontal gradient, or None.

    Returns a dict matching the Fill schema (without the pydantic wrapper),
    or None if the region is too textured to call.
    """
    if rgb_patch.size == 0:
        return None
    h, w = rgb_patch.shape[:2]
    if h < 2 or w < 2:
        return None
    arr = rgb_patch.astype(np.float32)
    overall_std = arr.reshape(-1, 3).std(axis=0).mean()

    if overall_std < 4.0:
        # Practically uniform → solid fill.
        mean = arr.reshape(-1, 3).mean(axis=0)
        return {"type": "solid", "color": rgb_to_hex(mean.tolist())}

    # Test gradient by comparing edge means.
    top = arr[: max(1, h // 8)].reshape(-1, 3).mean(axis=0)
    bot = arr[-max(1, h // 8):].reshape(-1, 3).mean(axis=0)
    left = arr[:, : max(1, w // 8)].reshape(-1, 3).mean(axis=0)
    right = arr[:, -max(1, w // 8):].reshape(-1, 3).mean(axis=0)

    vert_diff = float(np.linalg.norm(top - bot))
    horz_diff = float(np.linalg.norm(left - right))

    # Sample row/col means and check monotonicity → gradient signature.
    if vert_diff > horz_diff and vert_diff > 30:
        rows = arr.mean(axis=1)  # (h, 3)
        diffs = np.diff(rows, axis=0).mean(axis=1)
        if (diffs >= -1).mean() > 0.9 or (diffs <= 1).mean() > 0.9:
            return {
                "type": "linear-gradient",
                "angle": 180.0,  # CSS: top→bottom
                "stops": [
                    {"offset": 0.0, "color": rgb_to_hex(top.tolist())},
                    {"offset": 1.0, "color": rgb_to_hex(bot.tolist())},
                ],
            }
    if horz_diff > vert_diff and horz_diff > 30:
        cols = arr.mean(axis=0)
        diffs = np.diff(cols, axis=0).mean(axis=1)
        if (diffs >= -1).mean() > 0.9 or (diffs <= 1).mean() > 0.9:
            return {
                "type": "linear-gradient",
                "angle": 90.0,  # CSS: left→right
                "stops": [
                    {"offset": 0.0, "color": rgb_to_hex(left.tolist())},
                    {"offset": 1.0, "color": rgb_to_hex(right.tolist())},
                ],
            }
    return None
