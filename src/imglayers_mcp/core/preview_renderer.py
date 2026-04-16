"""Render a preview by compositing layers per manifest ordering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..storage.paths import ProjectPaths


def render_preview(
    manifest: dict,
    project_paths: ProjectPaths,
    output_name: str = "preview.png",
) -> Path:
    canvas_w = int(manifest["canvas"]["width"])
    canvas_h = int(manifest["canvas"]["height"])
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas[..., 3] = 255  # opaque white-ish default

    # Fill with canvas background if provided.
    bg_hex = manifest["canvas"].get("background")
    if bg_hex:
        rgb = _hex_to_rgb(bg_hex)
        canvas[..., :3] = rgb

    layers = sorted(manifest.get("layers", []), key=lambda l: l.get("zIndex", l.get("z_index", 0)))
    for layer in layers:
        asset = layer.get("asset")
        if not asset:
            continue
        rel = asset.get("path")
        if not rel:
            continue
        layer_path = project_paths.dir / rel
        if not layer_path.exists():
            continue
        overlay = np.asarray(Image.open(layer_path).convert("RGBA"))
        if overlay.shape[:2] != (canvas_h, canvas_w):
            # Paste via Pillow to handle differing sizes (unexpected but safe).
            pil_canvas = Image.fromarray(canvas, mode="RGBA")
            pil_canvas.alpha_composite(
                Image.fromarray(overlay, mode="RGBA"),
                (int(layer["bbox"]["x"]), int(layer["bbox"]["y"])),
            )
            canvas = np.asarray(pil_canvas).copy()
        else:
            canvas = _alpha_composite(canvas, overlay)

    out_path = project_paths.preview_dir / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGBA").save(out_path, format="PNG", optimize=True)
    return out_path


def _alpha_composite(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    bg_f = bg.astype(np.float32) / 255.0
    fg_f = fg.astype(np.float32) / 255.0
    a = fg_f[..., 3:4]
    out_rgb = fg_f[..., :3] * a + bg_f[..., :3] * (1 - a)
    out_a = a + bg_f[..., 3:4] * (1 - a)
    return (np.clip(np.concatenate([out_rgb, out_a], axis=2), 0, 1) * 255 + 0.5).astype(np.uint8)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
