"""Annotated preview: original image + bbox outlines + labels per layer.

This is a *diagnostic* preview meant for humans (and AIs reading it back)
to understand how an image was decomposed — distinct from the plain
compositing preview produced by preview_renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from ..storage.paths import ProjectPaths


# Deterministic color palette keyed on semanticRole so a role always
# gets the same color across projects — helps the eye.
ROLE_COLORS: dict[str, tuple[int, int, int]] = {
    "background": (160, 160, 160),
    "headline": (220, 50, 50),
    "subheadline": (230, 120, 40),
    "body_text": (240, 180, 40),
    "button": (40, 180, 90),
    "card": (60, 140, 210),
    "icon": (140, 90, 210),
    "logo": (200, 60, 180),
    "product_image": (20, 130, 200),
    "illustration": (20, 170, 170),
    "decoration": (150, 150, 200),
    "unknown": (110, 110, 110),
}


def render_annotated_preview(
    manifest: dict,
    project_paths: ProjectPaths,
    *,
    output_name: str = "preview_annotated.png",
    show_labels: bool = True,
    include_background: bool = False,
    only_ids: Iterable[str] | None = None,
    stroke_width: int = 3,
) -> Path:
    """Draw bboxes + labels on top of the current composited preview."""
    base_path = project_paths.preview_path
    if not base_path.exists():
        # Fall back to original if plain preview is missing.
        base_path = project_paths.original_path
    base = Image.open(base_path).convert("RGBA")
    canvas_w = int(manifest["canvas"]["width"])
    canvas_h = int(manifest["canvas"]["height"])
    if base.size != (canvas_w, canvas_h):
        base = base.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(max(10, min(24, canvas_h // 40)))

    allowed = set(only_ids) if only_ids else None
    layers = sorted(manifest.get("layers", []), key=lambda l: l.get("zIndex", l.get("z_index", 0)))

    for layer in layers:
        if allowed is not None and layer["id"] not in allowed:
            continue
        role = layer.get("semanticRole") or layer.get("semantic_role") or "unknown"
        if not include_background and role == "background":
            continue
        bbox = layer["bbox"]
        x1 = int(round(bbox["x"]))
        y1 = int(round(bbox["y"]))
        x2 = int(round(bbox["x"] + bbox["width"]))
        y2 = int(round(bbox["y"] + bbox["height"]))
        color = ROLE_COLORS.get(role, (110, 110, 110))

        # Translucent fill + solid stroke.
        fill = (*color, 40)
        stroke = (*color, 230)
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=stroke, width=stroke_width)

        if show_labels:
            label = f"{layer['id']} · {role}"
            text = layer.get("text") or {}
            if text.get("content"):
                snippet = text["content"].replace("\n", " ")
                if len(snippet) > 24:
                    snippet = snippet[:22] + "…"
                label = f"{label} · “{snippet}”"
            _draw_label(draw, (x1, y1), label, color, font, canvas_w)

    out = project_paths.preview_dir / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).save(out, format="PNG", optimize=True)
    return out


def render_layer_grid(
    manifest: dict,
    project_paths: ProjectPaths,
    *,
    output_name: str = "preview_grid.png",
    columns: int = 3,
    cell_padding: int = 16,
    include_background: bool = True,
) -> Path:
    """Render one thumbnail per layer in a grid for visual inspection."""
    layers = sorted(manifest.get("layers", []), key=lambda l: l.get("zIndex", l.get("z_index", 0)))
    if not include_background:
        layers = [l for l in layers if (l.get("semanticRole") or l.get("semantic_role")) != "background"]
    if not layers:
        # Empty grid — still emit a tiny placeholder.
        Image.new("RGBA", (64, 64), (240, 240, 240, 255)).save(
            project_paths.preview_dir / output_name
        )
        return project_paths.preview_dir / output_name

    canvas_w = int(manifest["canvas"]["width"])
    canvas_h = int(manifest["canvas"]["height"])
    # Target thumbnail size — fit grid to ~1200px wide.
    target_grid_w = min(1600, max(600, canvas_w))
    thumb_w = max(120, (target_grid_w - cell_padding * (columns + 1)) // columns)
    scale = thumb_w / max(1, canvas_w)
    thumb_h = max(1, int(round(canvas_h * scale)))
    label_h = 28
    cell_w = thumb_w + cell_padding
    cell_h = thumb_h + label_h + cell_padding

    rows = (len(layers) + columns - 1) // columns
    grid_w = columns * cell_w + cell_padding
    grid_h = rows * cell_h + cell_padding
    grid = Image.new("RGBA", (grid_w, grid_h), (245, 245, 245, 255))
    draw = ImageDraw.Draw(grid)
    font = _load_font(14)

    # Checker tile to visualize alpha.
    checker = _make_checker(thumb_w, thumb_h, tile=16)

    for idx, layer in enumerate(layers):
        row = idx // columns
        col = idx % columns
        x0 = cell_padding + col * cell_w
        y0 = cell_padding + row * cell_h

        asset = layer.get("asset")
        tile = checker.copy()
        if asset:
            asset_path = project_paths.dir / asset["path"]
            if asset_path.exists():
                im = Image.open(asset_path).convert("RGBA").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                tile = Image.alpha_composite(tile, im)
        grid.paste(tile, (x0, y0))

        role = layer.get("semanticRole") or layer.get("semantic_role") or "unknown"
        color = ROLE_COLORS.get(role, (110, 110, 110))
        # Label strip under the thumbnail.
        draw.rectangle(
            [x0, y0 + thumb_h, x0 + thumb_w, y0 + thumb_h + label_h],
            fill=(*color, 220),
        )
        label = f"{layer['id']} · {role}"
        draw.text((x0 + 6, y0 + thumb_h + 6), label, fill=(255, 255, 255, 255), font=font)

    out = project_paths.preview_dir / output_name
    grid.save(out, format="PNG", optimize=True)
    return out


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    anchor: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    canvas_w: int,
) -> None:
    x, y = anchor
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 8
    th = bbox[3] - bbox[1] + 6
    # Prefer above; flip to below if we'd go above the canvas.
    bx = max(0, min(canvas_w - tw, x))
    by = y - th - 2
    if by < 0:
        by = y + 2
    draw.rectangle([bx, by, bx + tw, by + th], fill=(*color, 220))
    draw.text((bx + 4, by + 2), text, fill=(255, 255, 255, 255), font=font)


def _make_checker(w: int, h: int, tile: int = 16) -> Image.Image:
    im = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    d = ImageDraw.Draw(im)
    for yy in range(0, h, tile):
        for xx in range(0, w, tile):
            if ((xx // tile) + (yy // tile)) % 2 == 0:
                d.rectangle([xx, yy, xx + tile, yy + tile], fill=(225, 225, 225, 255))
    return im
