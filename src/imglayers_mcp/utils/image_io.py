"""Image IO and transforms — thin wrappers around Pillow/numpy."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image


def resolve_input_uri(input_uri: str) -> Path:
    parsed = urlparse(input_uri)
    if parsed.scheme in ("", "file"):
        raw = parsed.path if parsed.scheme == "file" else input_uri
        return Path(unquote(raw)).expanduser().resolve()
    raise ValueError(f"unsupported input_uri scheme: {parsed.scheme!r}")


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGBA")
        return np.asarray(im).copy()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im).copy()


def save_png(array: np.ndarray, path: Path) -> None:
    if array.ndim == 2:
        im = Image.fromarray(array, mode="L")
    elif array.shape[2] == 4:
        im = Image.fromarray(array, mode="RGBA")
    elif array.shape[2] == 3:
        im = Image.fromarray(array, mode="RGB")
    else:
        raise ValueError(f"unsupported array shape: {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG", optimize=True)


def resize_to_max_side(rgba: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Return (resized_image, scale_applied). scale = new/original."""
    h, w = rgba.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgba, 1.0
    scale = max_side / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    mode = "RGBA" if rgba.shape[2] == 4 else "RGB"
    im = Image.fromarray(rgba, mode=mode).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(im).copy(), scale


def composite_over(background: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite overlay (RGBA) on top of background (RGB or RGBA)."""
    if overlay_rgba.shape[2] != 4:
        raise ValueError("overlay must be RGBA")
    fg = overlay_rgba.astype(np.float32) / 255.0
    if background.shape[2] == 3:
        bg = np.concatenate(
            [background.astype(np.float32) / 255.0, np.ones(background.shape[:2] + (1,), dtype=np.float32)],
            axis=2,
        )
    else:
        bg = background.astype(np.float32) / 255.0

    alpha = fg[..., 3:4]
    out_rgb = fg[..., :3] * alpha + bg[..., :3] * (1.0 - alpha)
    out_alpha = alpha + bg[..., 3:4] * (1.0 - alpha)
    out = np.concatenate([out_rgb, out_alpha], axis=2)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
