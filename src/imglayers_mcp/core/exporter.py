"""Derived-output exporters: SVG + PSD + codegen-plan."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from ..storage.paths import ProjectPaths
from .codegen_planner import build_codegen_plan


def export_svg(manifest: dict, project_paths: ProjectPaths, name: str = "export.svg") -> Path:
    w = int(manifest["canvas"]["width"])
    h = int(manifest["canvas"]["height"])
    lines: list[str] = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
    ]
    bg = manifest["canvas"].get("background")
    if bg:
        lines.append(f'  <rect width="{w}" height="{h}" fill="{escape(bg)}"/>')

    for layer in sorted(manifest.get("layers", []), key=lambda l: l.get("zIndex", l.get("z_index", 0))):
        bbox = layer["bbox"]
        lid = layer["id"]
        role = layer.get("semanticRole") or layer.get("semantic_role")
        asset = layer.get("asset")
        text = layer.get("text")

        group_open = f'  <g id="{escape(lid)}" data-role="{escape(str(role))}">'
        group_close = "  </g>"
        lines.append(group_open)
        if text and text.get("content"):
            style = layer.get("styleHints") or {}
            font_size = style.get("fontSize") or max(12, bbox["height"] * 0.75)
            color = style.get("textColor") or "#000000"
            weight = style.get("fontWeight") or 500
            ty = bbox["y"] + font_size
            lines.append(
                f'    <text x="{bbox["x"]}" y="{ty}" '
                f'font-size="{font_size}" fill="{escape(color)}" '
                f'font-weight="{weight}" font-family="sans-serif" '
                f'xml:space="preserve">{escape(text["content"])}</text>'
            )
        elif asset:
            href = escape(asset["path"])
            lines.append(
                f'    <image href="{href}" x="{bbox["x"]}" y="{bbox["y"]}" '
                f'width="{bbox["width"]}" height="{bbox["height"]}"/>'
            )
        lines.append(group_close)

    lines.append("</svg>")
    out = project_paths.export_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_psd(manifest: dict, project_paths: ProjectPaths, name: str = "export.psd") -> Path | None:
    """Export a PSD if `psd-tools` is installed; else None."""
    try:
        from psd_tools import PSDImage  # type: ignore
        from psd_tools.api.layers import Group, PixelLayer  # type: ignore  # noqa: F401
    except Exception:
        return None

    try:
        from PIL import Image  # type: ignore
        w = int(manifest["canvas"]["width"])
        h = int(manifest["canvas"]["height"])
        psd = PSDImage.new(mode="RGBA", size=(w, h))
        for layer in sorted(manifest.get("layers", []), key=lambda l: l.get("zIndex", l.get("z_index", 0))):
            asset = layer.get("asset")
            if not asset:
                continue
            layer_path = project_paths.dir / asset["path"]
            if not layer_path.exists():
                continue
            im = Image.open(layer_path).convert("RGBA")
            try:
                pl = PixelLayer.frompil(im, psd, layer["id"])  # type: ignore[attr-defined]
            except Exception:
                continue
            psd.append(pl)
        out = project_paths.export_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        psd.save(out)
        return out
    except Exception:
        return None


def export_codegen_plan(manifest: dict, project_paths: ProjectPaths, target: str = "generic", name: str = "codegen-plan.json") -> Path:
    plan = build_codegen_plan(manifest, target=target)
    out = project_paths.export_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
