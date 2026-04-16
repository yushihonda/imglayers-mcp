"""LayerD adapter — CV-based image decomposition into RGBA layers.

Falls back to a pure-numpy CC pipeline when the upstream `layerd` package is
not installed. Accepts optional OCR text_boxes to protect text regions from
being split at stroke boundaries.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..core._cc import connected_components
from ..core.types import RawLayer
from ..utils.bbox import Box
from ..utils.color import average_color
from ..utils.logging import get_logger

log = get_logger(__name__)

ElementKind = Literal["text", "image", "vector_like", "unknown"]


class LayerDAdapter:
    def __init__(self, force_fallback: bool = False) -> None:
        self._impl = None
        self._available = False
        if not force_fallback:
            try:
                import layerd  # type: ignore
                self._impl = layerd
                self._available = True
                log.info("LayerD adapter: using upstream layerd package")
            except Exception as exc:
                log.info("LayerD adapter: upstream unavailable (%s); using CV fallback", exc)

    @property
    def available(self) -> bool:
        return self._available

    def decompose(
        self,
        rgba: np.ndarray,
        *,
        detail: str = "balanced",
        text_boxes: list[Box] | None = None,
    ) -> list[RawLayer]:
        return _cv_fallback(rgba, detail=detail, text_boxes=text_boxes)


def _cv_fallback(
    rgba: np.ndarray,
    *,
    detail: str = "balanced",
    text_boxes: list[Box] | None = None,
) -> list[RawLayer]:
    """Hybrid: OCR-guided text cut + color-segmented CC for non-text regions."""
    h, w = rgba.shape[:2]
    if rgba.shape[2] == 3:
        rgba = np.concatenate(
            [rgba, np.full(rgba.shape[:2] + (1,), 255, dtype=np.uint8)], axis=2
        )
    rgb = rgba[..., :3]

    bg_color_rgb = _estimate_background(rgb)
    diff = np.linalg.norm(rgb.astype(np.int16) - bg_color_rgb.astype(np.int16), axis=2)
    base_thr = {"fast": 48, "balanced": 32, "high": 22}.get(detail, 32)
    fg_mask = diff > base_thr

    layers: list[RawLayer] = []
    z = 1

    # Phase 1: text layers from OCR bboxes (one layer per line, alpha from fg_mask)
    text_region_mask = np.zeros((h, w), dtype=bool)
    tboxes = text_boxes or []
    for tbox in tboxes:
        x1 = max(0, int(round(tbox.x)))
        y1 = max(0, int(round(tbox.y)))
        x2 = min(w, int(round(tbox.x + tbox.w)))
        y2 = min(h, int(round(tbox.y + tbox.h)))
        if x2 <= x1 or y2 <= y1:
            continue
        text_region_mask[y1:y2, x1:x2] = True
        region_fg = fg_mask[y1:y2, x1:x2]
        sub_rgb = rgb[y1:y2, x1:x2]
        alpha = (region_fg.astype(np.uint8) * 255)[..., None]
        patch_rgba = np.concatenate(
            [np.where(region_fg[..., None], sub_rgb, 0).astype(np.uint8), alpha], axis=2,
        )
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = patch_rgba
        layers.append(RawLayer(
            rgba=canvas,
            bbox=Box(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            kind="text", engine="layerd-fallback", confidence=0.85, z_index=z,
            debug={"source": "ocr_bbox"},
        ))
        z += 1

    # Phase 2: non-text foreground via plain or color-segmented CC
    remainder_mask = fg_mask & ~_dilate_mask(text_region_mask, radius=3)
    fg_ratio = remainder_mask.sum() / max(1, h * w)
    if fg_ratio > 0.25:
        non_text = _color_segment_cc(rgb, remainder_mask, bg_color_rgb, detail, h, w, text_boxes=tboxes)
    else:
        non_text = _plain_cc(rgb, remainder_mask, detail, h, w, text_boxes=tboxes)
    for layer in non_text:
        layer.z_index = z
        z += 1
    layers.extend(non_text)

    # Phase 2b: soft-threshold pass to find near-bg-colored containers
    # (e.g. white cards on off-white background). Only keeps regions that
    # wrap at least one OCR text_box.
    soft_containers = _find_soft_containers(
        rgb, bg_color_rgb, text_region_mask, tboxes, h, w,
        primary_thr=base_thr, secondary_thr=max(3, base_thr // 5),
    )
    for layer in soft_containers:
        layer.z_index = z
        z += 1
    layers.extend(soft_containers)

    # Phase 3: background (erase foreground with soft threshold)
    soft_thr = max(8, base_thr // 3)
    soft_mask = diff > soft_thr
    bg_erase = _dilate_mask(fg_mask | text_region_mask | soft_mask, radius=2)
    bg_rgb = rgb.copy()
    bg_rgb[bg_erase] = bg_color_rgb
    bg_rgba = np.concatenate([bg_rgb, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
    layers.insert(0, RawLayer(
        rgba=bg_rgba, bbox=Box(0, 0, w, h), kind="image",
        engine="layerd-fallback", confidence=0.6, z_index=0,
        debug={"role_guess": "background", "bg_color": average_color(rgb)},
    ))

    return layers


def _estimate_background(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    bw = max(1, min(h, w) // 20)
    border = np.concatenate([
        rgb[:bw].reshape(-1, 3), rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3), rgb[:, -bw:].reshape(-1, 3),
    ], axis=0)
    q = (border // 16) * 16 + 8
    packed = (q[:, 0].astype(np.int64) << 16) | (q[:, 1].astype(np.int64) << 8) | q[:, 2].astype(np.int64)
    values, counts = np.unique(packed, return_counts=True)
    winner = int(values[int(np.argmax(counts))])
    return np.array([(winner >> 16) & 0xFF, (winner >> 8) & 0xFF, winner & 0xFF], dtype=np.uint8)


def _find_soft_containers(
    rgb: np.ndarray,
    bg_color: np.ndarray,
    text_region_mask: np.ndarray,
    text_boxes: list[Box],
    h: int,
    w: int,
    *,
    primary_thr: int,
    secondary_thr: int,
) -> list[RawLayer]:
    """Find low-contrast container regions (e.g. white cards on off-white bg).

    Use a soft threshold to build a mask, run CC, and keep only components
    that fully contain at least one OCR text_box. Returns them as image-kind
    RawLayers so they become candidate "card" roles later.
    """
    if secondary_thr >= primary_thr:
        return []
    if not text_boxes:
        return []

    diff = np.linalg.norm(rgb.astype(np.int16) - bg_color.astype(np.int16), axis=2)
    soft_mask = (diff > secondary_thr) & (diff <= primary_thr)
    # Exclude text regions from soft candidates (text handled by Phase 1).
    soft_mask = soft_mask & ~text_region_mask

    if not soft_mask.any():
        return []

    # Erode + dilate to clean up noise.
    soft_mask_clean = _dilate_mask(soft_mask, radius=1)
    comps = connected_components(soft_mask_clean)
    # Require containers to be reasonably large.
    min_area = (h * w) // 200
    comps = [c for c in comps if c["area"] >= min_area]

    layers: list[RawLayer] = []
    for comp in comps:
        y1, y2, x1, x2 = comp["y1"], comp["y2"], comp["x1"], comp["x2"]
        bbox = Box(x=float(x1), y=float(y1), w=float(x2 - x1), h=float(y2 - y1))
        # Accept only if it contains at least one text box.
        contains = False
        for tb in text_boxes:
            if (
                bbox.x - 2 <= tb.x
                and bbox.y - 2 <= tb.y
                and bbox.x + bbox.w + 2 >= tb.x + tb.w
                and bbox.y + bbox.h + 2 >= tb.y + tb.h
            ):
                contains = True
                break
        if not contains:
            continue

        # Cut the container region with soft alpha (binary here, but full
        # bbox filled so the card's interior is opaque).
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        canvas[y1:y2, x1:x2, :3] = rgb[y1:y2, x1:x2]
        canvas[y1:y2, x1:x2, 3] = 255
        layers.append(RawLayer(
            rgba=canvas, bbox=bbox, kind="image",
            engine="layerd-fallback", confidence=0.55, z_index=0,
            debug={"source": "soft_container", "text_count": sum(1 for tb in text_boxes
                if bbox.x - 2 <= tb.x and bbox.y - 2 <= tb.y
                and bbox.x + bbox.w + 2 >= tb.x + tb.w
                and bbox.y + bbox.h + 2 >= tb.y + tb.h)},
        ))
    return layers


def _plain_cc(
    rgb: np.ndarray, mask: np.ndarray, detail: str, h: int, w: int,
    *, text_boxes: list[Box] | None = None,
) -> list[RawLayer]:
    if not mask.any():
        return []
    comps = connected_components(mask)
    min_area = max(500, (h * w) // 1000)
    cap = {"fast": 6, "balanced": 12, "high": 20}.get(detail, 12)
    candidates = [c for c in comps if c["area"] >= min_area]
    candidates.sort(key=lambda c: -c["area"])
    candidates = candidates[:cap]
    return _comps_to_layers(candidates, rgb, h, w, text_boxes=text_boxes or [])


def _color_segment_cc(
    rgb: np.ndarray, mask: np.ndarray, bg_color: np.ndarray,
    detail: str, h: int, w: int, *, text_boxes: list[Box] | None = None,
) -> list[RawLayer]:
    if not mask.any():
        return []
    fg_pixels = rgb[mask].astype(np.int32)
    q = (fg_pixels // 32) * 32 + 16
    packed = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    unique, _, counts = np.unique(packed, return_inverse=True, return_counts=True)
    k = {"fast": 4, "balanced": 6, "high": 10}.get(detail, 6)
    top_k = np.argsort(-counts)[:k]
    top_colors = np.stack([
        (unique[top_k] >> 16) & 0xFF, (unique[top_k] >> 8) & 0xFF, unique[top_k] & 0xFF,
    ], axis=1).astype(np.int32)

    fg_coords = np.argwhere(mask)
    fg_rgb = rgb[mask].astype(np.int32)
    dists = np.linalg.norm(fg_rgb[:, None, :] - top_colors[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    min_area = max(500, (h * w) // 1000)
    cap = {"fast": 6, "balanced": 12, "high": 20}.get(detail, 12)
    all_candidates: list[dict] = []
    for ci in range(len(top_colors)):
        cluster_mask = np.zeros((h, w), dtype=bool)
        cluster_pixels = fg_coords[labels == ci]
        if len(cluster_pixels) == 0:
            continue
        cluster_mask[cluster_pixels[:, 0], cluster_pixels[:, 1]] = True
        comps = connected_components(cluster_mask)
        all_candidates.extend([c for c in comps if c["area"] >= min_area])
    all_candidates.sort(key=lambda c: -c["area"])
    all_candidates = all_candidates[:cap]
    return _comps_to_layers(all_candidates, rgb, h, w, text_boxes=text_boxes or [])


def _comps_to_layers(
    comps: list[dict], rgb: np.ndarray, h: int, w: int,
    *, text_boxes: list[Box] | None = None,
) -> list[RawLayer]:
    layers: list[RawLayer] = []
    text_boxes = text_boxes or []
    for comp in comps:
        y1, y2, x1, x2 = comp["y1"], comp["y2"], comp["x1"], comp["x2"]
        sub_mask = comp["mask"][y1:y2, x1:x2]
        sub_rgb = rgb[y1:y2, x1:x2]
        alpha = (sub_mask.astype(np.uint8) * 255)[..., None]
        patch_rgba = np.concatenate(
            [np.where(sub_mask[..., None], sub_rgb, 0).astype(np.uint8), alpha], axis=2,
        )
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = patch_rgba
        bbox = Box(x=float(x1), y=float(y1), w=float(x2 - x1), h=float(y2 - y1))

        # If this component wraps a text_box, it's a container candidate.
        # Fill the container's *full* bbox (including text holes) with the
        # dominant color — turning a punched-out shape into a solid shape.
        # Then, on top, we only keep the alpha mask's convex-hull region so
        # the container still has its real rounded shape (not a square).
        contained_text = [tb for tb in text_boxes if _bbox_contains(bbox, tb)]
        if contained_text and comp["area"] > 0:
            visible = rgb[comp["mask"]]
            if visible.size > 0:
                dom = _dominant_color(visible.reshape(-1, 3))
                # Build a "filled-in" alpha: CC mask dilated + text bboxes filled.
                filled_alpha = comp["mask"].copy()
                for tb in contained_text:
                    tx1 = max(0, int(round(tb.x)))
                    ty1 = max(0, int(round(tb.y)))
                    tx2 = min(w, int(round(tb.x + tb.w)))
                    ty2 = min(h, int(round(tb.y + tb.h)))
                    if tx2 > tx1 and ty2 > ty1:
                        filled_alpha[ty1:ty2, tx1:tx2] = True
                # Close small gaps between CC pixels and text region.
                filled_alpha = _dilate_mask(filled_alpha, radius=2)
                # Apply: where filled_alpha is True AND original CC was not set,
                # paint with dom color. Where original CC was set, keep original.
                orig_alpha = canvas[..., 3] > 0
                paint_region = filled_alpha & ~orig_alpha
                canvas[..., :3] = np.where(
                    paint_region[..., None], dom, canvas[..., :3]
                )
                canvas[..., 3] = np.where(filled_alpha, 255, canvas[..., 3])

        kind = _guess_kind(bbox, comp["area"])
        layers.append(RawLayer(
            rgba=canvas, bbox=bbox, kind=kind,
            engine="layerd-fallback", confidence=0.6, z_index=0,
            debug={"pixel_area": int(comp["area"]), "wraps_text": len(contained_text)},
        ))
    return layers


def _dominant_color(pixels: np.ndarray) -> np.ndarray:
    """Return the mode (most common) color via coarse quantization."""
    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    q = (pixels.astype(np.int32) // 8) * 8 + 4
    packed = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    unique, counts = np.unique(packed, return_counts=True)
    winner = int(unique[int(np.argmax(counts))])
    r = (winner >> 16) & 0xFF
    g = (winner >> 8) & 0xFF
    b = winner & 0xFF
    # Within the winning bucket, take the mean of actual pixels for smoothness.
    in_bucket = packed == winner
    exact = pixels[in_bucket].mean(axis=0).round().astype(np.uint8)
    return exact


def _bbox_contains(outer: Box, inner: Box, tol: float = 2.0) -> bool:
    return (
        outer.x - tol <= inner.x
        and outer.y - tol <= inner.y
        and outer.x + outer.w + tol >= inner.x + inner.w
        and outer.y + outer.h + tol >= inner.y + inner.h
    )


def _guess_kind(bbox: Box, area: int) -> ElementKind:
    w, h = bbox.w, bbox.h
    if w <= 0 or h <= 0:
        return "unknown"
    density = area / max(1.0, w * h)
    if density > 0.85 and min(w, h) >= 32:
        return "image"
    if density < 0.35:
        return "vector_like"
    return "unknown"


def _dilate_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        out[1:] |= out[:-1]
        out[:-1] |= out[1:]
        out[:, 1:] |= out[:, :-1]
        out[:, :-1] |= out[:, 1:]
    return out
