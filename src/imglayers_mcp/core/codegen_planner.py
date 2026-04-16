"""Rule-based codegen plan generator (spec §6.7, §12.2)."""

from __future__ import annotations

from typing import Any


def build_codegen_plan(manifest: dict, *, target: str = "generic") -> dict[str, Any]:
    layers = sorted(
        manifest.get("layers", []),
        key=lambda l: l.get("zIndex", l.get("z_index", 0)),
    )
    canvas = manifest.get("canvas", {}) or {}
    canvas_w = int(canvas.get("width", 0)) or 1
    canvas_h = int(canvas.get("height", 0)) or 1

    nodes: list[dict[str, Any]] = []
    container_children: list[str] = []
    background_node: dict[str, Any] | None = None

    for l in layers:
        role = l.get("semanticRole") or l.get("semantic_role") or "unknown"
        lid = l["id"]
        bbox = l.get("bbox") or {}
        style_hints = l.get("styleHints") or {}

        # Express geometry as both absolute pixels and percentages so the
        # generated code can choose between fixed-canvas and responsive layouts.
        layout = _layout_for(bbox, canvas_w, canvas_h)

        node: dict[str, Any] = {
            "id": lid,
            "component": _role_to_component(role, target),
            "layout": layout,
        }

        text = l.get("text")
        if text and text.get("content"):
            node["props"] = {"text": text["content"]}
            if role == "headline":
                node["props"]["level"] = 1
            if role == "subheadline":
                node["props"]["level"] = 2
            if role == "button":
                node["props"]["variant"] = "primary"
            node["textStyle"] = _text_style_block(style_hints)
        else:
            node["fill"] = _fill_block(style_hints, l)
            if l.get("asset", {}).get("path"):
                node["asset"] = l["asset"]["path"]

        if role == "background":
            background_node = node
            continue
        nodes.append(node)
        container_children.append(lid)

    root_id = "RootSection"
    if any(l.get("semanticRole") == "headline" for l in layers):
        root_id = "HeroSection"

    # Build group nodes from manifest groups. Each group wraps its child layer
    # IDs and carries the full-line text so codegen can emit a single text
    # component if it prefers line-level over char-level.
    group_nodes: list[dict[str, Any]] = []
    groups = manifest.get("groups", [])
    node_by_id = {n["id"]: n for n in nodes}
    for g in groups:
        gid = g.get("id") or g.get("name", "")
        child_ids = g.get("layerIds") or g.get("layer_ids") or []
        child_layouts = [node_by_id[cid]["layout"] for cid in child_ids if cid in node_by_id]
        child_text_styles = [node_by_id[cid].get("textStyle") for cid in child_ids if cid in node_by_id]
        group_node: dict[str, Any] = {
            "id": gid,
            "component": "TextLine",
            "text": g.get("name", ""),
            "children": child_ids,
        }
        if child_text_styles and child_text_styles[0]:
            group_node["textStyle"] = child_text_styles[0]
        if child_layouts:
            xs = [cl["x"] for cl in child_layouts]
            ys = [cl["y"] for cl in child_layouts]
            x2s = [cl["x"] + cl["width"] for cl in child_layouts]
            y2s = [cl["y"] + cl["height"] for cl in child_layouts]
            group_node["layout"] = _layout_for(
                {"x": min(xs), "y": min(ys), "width": max(x2s) - min(xs), "height": max(y2s) - min(ys)},
                canvas_w, canvas_h,
            )
        group_nodes.append(group_node)

    plan: dict[str, Any] = {
        "target": target,
        "rootComponent": root_id,
        "layout": "absolute",
        "canvas": {"width": canvas_w, "height": canvas_h},
        "background": background_node,
        "nodes": nodes,
        "groups": group_nodes,
        "components": [
            {"id": "root_container", "kind": "container", "children": container_children}
        ],
    }
    return plan


def _layout_for(bbox: dict, canvas_w: int, canvas_h: int) -> dict[str, Any]:
    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    w = float(bbox.get("width", 0))
    h = float(bbox.get("height", 0))
    return {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "xPct": round(x / canvas_w * 100, 3),
        "yPct": round(y / canvas_h * 100, 3),
        "widthPct": round(w / canvas_w * 100, 3),
        "heightPct": round(h / canvas_h * 100, 3),
    }


def _text_style_block(style_hints: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("fontSize", "fontWeight", "fontFamilyGuess", "fontStyle",
              "textColor", "textAlign", "lineHeight", "letterSpacing"):
        if style_hints.get(k) is not None:
            out[k] = style_hints[k]
    return out


def _fill_block(style_hints: dict, layer: dict) -> dict[str, Any] | None:
    fill = style_hints.get("fill")
    if fill:
        return fill
    # Image-typed layers without a structured fill fall back to image asset reference.
    asset = layer.get("asset", {}).get("path")
    if asset:
        return {"type": "image", "src": asset}
    return None


def _role_to_component(role: str, target: str) -> str:
    mapping = {
        "headline": "Heading",
        "subheadline": "Heading",
        "body_text": "Paragraph",
        "button": "Button",
        "card": "Card",
        "icon": "Icon",
        "logo": "Logo",
        "product_image": "Image",
        "illustration": "Image",
        "decoration": "Decoration",
        "background": "Background",
    }
    if target == "html":
        html_map = {
            "headline": "h1",
            "subheadline": "h2",
            "body_text": "p",
            "button": "button",
            "card": "section",
            "icon": "span",
            "logo": "img",
            "product_image": "img",
            "illustration": "img",
            "decoration": "div",
            "background": "div",
        }
        return html_map.get(role, "div")
    return mapping.get(role, "Box")
