"""Build the canonical manifest.json (spec §8)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from ..models.manifest import (
    AssetIndex,
    BBox,
    CanvasInfo,
    CodegenHints,
    ExportIndex,
    Fill,
    GradientStop,
    LayerAsset,
    LayerGroup,
    LayerNode,
    Manifest,
    PipelineInfo,
    Provenance,
    RetryState,
    SourceInfo,
    StatsInfo,
    StylePayload,
    TextLine,
    TextPayload,
    WarningItem,
)
from ..storage.paths import ProjectPaths
from ..utils.bbox import Box
from ..utils.color import (
    average_color,
    detect_solid_or_gradient_fill,
    dominant_colors,
    extract_foreground_color,
)
from .merger import PromotedLayer, TextGroupInfo
from .naming import NameAllocator, infer_semantic_role, is_full_canvas


def build_manifest(
    *,
    project_paths: ProjectPaths,
    source: SourceInfo,
    canvas: CanvasInfo,
    pipeline: PipelineInfo,
    original_rgb: np.ndarray,
    promoted_layers: list[PromotedLayer],
    text_groups: list[TextGroupInfo] | None = None,
    warnings: Iterable[WarningItem] = (),
    exports: ExportIndex | None = None,
    style_overrides: dict[int, dict] | None = None,
) -> Manifest:
    style_overrides = style_overrides or {}
    allocator = NameAllocator()

    # Rank text layers by font-size (taller bbox ≈ larger text).
    text_layers_sorted = sorted(
        [l for l in promoted_layers if l.kind == "text"],
        key=lambda l: -l.bbox.h,
    )
    text_rank: dict[int, int] = {id(l): r for r, l in enumerate(text_layers_sorted)}
    headline_box: Box | None = text_layers_sorted[0].bbox if text_layers_sorted else None

    # Area rank across all layers for fallback heuristics.
    area_sorted = sorted(promoted_layers, key=lambda l: -(l.bbox.w * l.bbox.h))
    area_rank = {id(l): r for r, l in enumerate(area_sorted)}

    layer_nodes: list[LayerNode] = []
    stats = {"text": 0, "image": 0, "vector_like": 0, "unknown": 0}

    # Precompute text bboxes for contains_text check.
    text_bboxes = [l.bbox for l in promoted_layers if l.kind == "text"]

    # Map promoted layer index → style override (from text_reconstruction).
    promoted_idx: dict[int, int] = {id(pl): i for i, pl in enumerate(promoted_layers)}

    for layer in promoted_layers:
        # Prefer vision-provided semantic hint (button/card/badge/icon) when available.
        hint = getattr(layer, "semantic_hint", None)
        hint_role = _map_vision_type_to_role(hint) if hint else None
        if hint_role:
            role = hint_role
        else:
            # Check if this non-text layer wraps any text element.
            contains_text = False
            if layer.kind != "text":
                for tb in text_bboxes:
                    if _contains(layer.bbox, tb):
                        contains_text = True
                        break
            role = infer_semantic_role(
                layer_type=layer.kind,  # type: ignore[arg-type]
                bbox=layer.bbox,
                canvas_w=canvas.width,
                canvas_h=canvas.height,
                text_content=layer.text_content,
                area_rank=area_rank[id(layer)],
                text_size_rank=text_rank.get(id(layer)),
                nearby_text_below_headline=_is_below(layer.bbox, headline_box)
                if headline_box is not None
                else False,
                is_full_canvas=is_full_canvas(layer.bbox, canvas.width, canvas.height),
                contains_text=contains_text,
            )
        layer_id = allocator.allocate(role)
        asset_rel = f"layers/{layer_id}.png"
        asset_abs = project_paths.layer_path(layer_id)

        # WS4: container hole-fill via background model.
        rgba_to_save = layer.rgba
        if role in ("card", "button", "badge") and layer.kind != "text" and text_bboxes:
            rgba_to_save = _inpaint_text_holes_v2(
                layer.rgba, layer.bbox, text_bboxes, canvas.width, canvas.height,
                original_rgb=original_rgb,
            )

        _save_layer_png(rgba_to_save, asset_abs)

        has_alpha = bool(layer.rgba.shape[-1] == 4 and (layer.rgba[..., 3] < 255).any())

        text_payload = None
        style_hints = None
        codegen_hints: CodegenHints | None = None
        if layer.kind == "text" and layer.text_content:
            text_payload = TextPayload(
                content=layer.text_content,
                language=layer.text_language,
                confidence=layer.text_confidence,
                lines=[TextLine(text=l["text"], bbox=BBox(**l["bbox"])) for l in layer.text_lines]
                if layer.text_lines
                else None,
            )
            style_hints = _infer_text_style(
                layer, original_rgb, canvas_bg=canvas.background, canvas_w=canvas.width
            )
            codegen_hints = _text_codegen_hints(role)
        elif layer.kind in ("image", "vector_like"):
            style_hints = _infer_region_style(
                layer, original_rgb, is_background=role == "background"
            )
            if role in ("button", "card"):
                codegen_hints = CodegenHints(
                    component_candidate={"button": "Button", "card": "Card"}[role],
                    container_likely=(role == "card"),
                )

        # Apply style_overrides from text_reconstruction when present.
        override = style_overrides.get(promoted_idx.get(id(layer), -1))
        if override and style_hints is not None:
            if override.get("font_family"):
                style_hints.font_family = override["font_family"]
            if override.get("font_candidates"):
                style_hints.font_candidates = override["font_candidates"]
            if override.get("reconstruction_confidence") is not None:
                style_hints.reconstruction_confidence = override["reconstruction_confidence"]
            if override.get("font_weight"):
                style_hints.font_weight = override["font_weight"]
            if override.get("color"):
                style_hints.color = override["color"]
            if override.get("text_align"):
                style_hints.text_align = override["text_align"]

        engines_sorted = sorted(set(layer.engines))
        engine_used = "layerd"
        for e in engines_sorted:
            if e.startswith("sam2"):
                engine_used = "sam2"
                break
            if e.startswith("vision"):
                engine_used = "vision"
                break
            if e.startswith("layerd"):
                engine_used = "layerd"

        debug = dict(layer.debug or {}) if hasattr(layer, "debug") and layer.debug else {}
        mask_quality = debug.get("mask_quality")
        alpha_edge_quality = debug.get("alpha_edge_quality")

        layer_nodes.append(
            LayerNode(
                id=layer_id,
                name=role,
                type=_to_layer_type(layer.kind),  # type: ignore[arg-type]
                semantic_role=role,  # type: ignore[arg-type]
                bbox=BBox(**layer.bbox.to_dict()),
                z_index=layer.z_index,
                visible=True,
                locked=False,
                editable=True,
                opacity=1.0,
                asset=LayerAsset(path=asset_rel, format="png", has_alpha=has_alpha),
                text=text_payload,
                style=style_hints,
                provenance=Provenance(engines=engines_sorted),
                confidence=layer.confidence,
                engine_used=engine_used,
                mask_quality=float(mask_quality) if mask_quality is not None else None,
                alpha_edge_quality=float(alpha_edge_quality) if alpha_edge_quality is not None else None,
                codegen_hints=codegen_hints,
            )
        )
        key = layer.kind if layer.kind in stats else "unknown"
        stats[key] += 1

    # low_confidence_layers is filled by verifier later; initial estimate here.
    low_conf = sum(1 for n in layer_nodes if n.confidence < 0.55)
    stats_info = StatsInfo(
        total_layers=len(layer_nodes),
        text_layers=stats["text"],
        image_layers=stats["image"],
        vector_like_layers=stats["vector_like"],
        low_confidence_layers=low_conf,
    )

    # Build groups from text_groups (char→line hierarchy).
    group_nodes: list[LayerGroup] = []
    if text_groups:
        tag_to_id = {pl.group_tag: lid for pl, lid in zip(promoted_layers, [n.id for n in layer_nodes]) if pl.group_tag}
        for tg in text_groups:
            child_ids = [tag_to_id[ct] for ct in tg.child_tags if ct in tag_to_id]
            if not child_ids:
                continue
            group_id = allocator.allocate("text_group")
            group_nodes.append(
                LayerGroup(id=group_id, name=tg.text, layer_ids=child_ids)
            )
            child_id_set = set(child_ids)
            for node in layer_nodes:
                if node.id in child_id_set:
                    node.children = [group_id]

    # Build container hierarchy: assign each layer to its smallest enclosing
    # container (card/button/badge/image with significantly larger bbox).
    _build_container_hierarchy(layer_nodes, group_nodes, allocator)

    manifest = Manifest(
        version="0.1.0",
        project_id=project_paths.project_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        source=source,
        canvas=canvas,
        pipeline=pipeline,
        stats=stats_info,
        assets=AssetIndex(original="meta/original.png", preview="preview/preview.png", layers_dir="layers"),
        layers=layer_nodes,
        groups=group_nodes,
        warnings=list(warnings),
        exports=exports or ExportIndex(manifest="manifest.json"),
    )
    return manifest


def _inpaint_text_holes_v2(
    rgba: np.ndarray,
    container_bbox: Box,
    text_bboxes: list[Box],
    canvas_w: int,
    canvas_h: int,
    original_rgb: np.ndarray,
) -> np.ndarray:
    """Background-model-aware hole fill (WS4).

    Classifies the container's bg (solid / linear / radial / texture / photo)
    and uses the matching fill strategy. Falls back to texture-tile on
    low-confidence classifications.
    """
    from .background_model import classify_region, fill_hole

    h, w = rgba.shape[:2]
    out = rgba.copy()

    # Crop the container region from the ORIGINAL (full source) so the bg
    # classifier sees the actual surrounding colors — the layer's alpha-cut
    # rgba can be mostly transparent.
    cx1 = max(0, int(round(container_bbox.x)))
    cy1 = max(0, int(round(container_bbox.y)))
    cx2 = min(w, int(round(container_bbox.x + container_bbox.w)))
    cy2 = min(h, int(round(container_bbox.y + container_bbox.h)))
    if cx2 <= cx1 or cy2 <= cy1:
        return out

    container_rgb = original_rgb[cy1:cy2, cx1:cx2]

    # Build a "clean bg mask" for classification: container pixels that
    # are NOT inside a text bbox.
    mask_bg = np.ones(container_rgb.shape[:2], dtype=bool)
    for tb in text_bboxes:
        if not _contains(container_bbox, tb):
            continue
        tx1 = max(0, int(round(tb.x - container_bbox.x)))
        ty1 = max(0, int(round(tb.y - container_bbox.y)))
        tx2 = min(container_rgb.shape[1], int(round(tb.x + tb.w - container_bbox.x)))
        ty2 = min(container_rgb.shape[0], int(round(tb.y + tb.h - container_bbox.y)))
        if tx2 > tx1 and ty2 > ty1:
            mask_bg[ty1:ty2, tx1:tx2] = False

    if not mask_bg.any():
        return out

    # Build classification region by flattening (background pixels only).
    bg_pixels_flat = container_rgb[mask_bg]
    # We pass a fabricated 2D view: reshape to 1xN for the classifier.
    # classify_region expects HxWx3, so give it the full container with bg
    # pixels; text regions will leak but we compensate by ignoring std
    # contributions (classifier uses medians / edges on the sample).
    sample = container_rgb.copy()
    # Replace text-region with median of bg pixels so classifier isn't fooled.
    med = np.median(bg_pixels_flat.reshape(-1, 3), axis=0).astype(np.uint8)
    sample[~mask_bg] = med
    model = classify_region(sample)

    # Fill each text bbox using the model.
    for tb in text_bboxes:
        if not _contains(container_bbox, tb):
            continue
        tx1 = max(0, int(round(tb.x)))
        ty1 = max(0, int(round(tb.y)))
        tx2 = min(w, int(round(tb.x + tb.w)))
        ty2 = min(h, int(round(tb.y + tb.h)))
        if tx2 <= tx1 or ty2 <= ty1:
            continue
        hole_in_container = np.zeros_like(mask_bg)
        hole_in_container[
            max(0, ty1 - cy1):min(mask_bg.shape[0], ty2 - cy1),
            max(0, tx1 - cx1):min(mask_bg.shape[1], tx2 - cx1),
        ] = True
        filled_container = fill_hole(model, container_rgb, hole_in_container)
        out[ty1:ty2, tx1:tx2, :3] = filled_container[
            max(0, ty1 - cy1):min(filled_container.shape[0], ty2 - cy1),
            max(0, tx1 - cx1):min(filled_container.shape[1], tx2 - cx1),
        ]
        out[ty1:ty2, tx1:tx2, 3] = 255

    return out


def _inpaint_text_holes(
    rgba: np.ndarray,
    container_bbox: Box,
    text_bboxes: list[Box],
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Replace text-region pixels in a container's RGBA with the surrounding fill.

    For each text bbox contained in the container, sample the perimeter just
    outside the text bbox (but still inside the container), take the median
    color, and paint that color over the text region — keeping the container's
    alpha intact. This turns a "button with punched-out text" into a clean
    button shape that can hold any text rendered on top.
    """
    h, w = rgba.shape[:2]
    out = rgba.copy()

    for tb in text_bboxes:
        if not _contains(container_bbox, tb):
            continue
        tx1 = max(0, int(round(tb.x)))
        ty1 = max(0, int(round(tb.y)))
        tx2 = min(w, int(round(tb.x + tb.w)))
        ty2 = min(h, int(round(tb.y + tb.h)))
        if tx2 <= tx1 or ty2 <= ty1:
            continue

        # Sample a ring just outside the text bbox, clipped to the container.
        cb_x1 = max(0, int(round(container_bbox.x)))
        cb_y1 = max(0, int(round(container_bbox.y)))
        cb_x2 = min(w, int(round(container_bbox.x + container_bbox.w)))
        cb_y2 = min(h, int(round(container_bbox.y + container_bbox.h)))
        pad = max(4, int(tb.h * 0.1))
        sx1 = max(cb_x1, tx1 - pad)
        sy1 = max(cb_y1, ty1 - pad)
        sx2 = min(cb_x2, tx2 + pad)
        sy2 = min(cb_y2, ty2 + pad)
        if sx2 <= sx1 or sy2 <= sy1:
            continue

        # Ring = outside box minus inside box.
        ring_pixels: list[np.ndarray] = []
        # Top & bottom stripes of the padded ring.
        if sy1 < ty1:
            ring_pixels.append(out[sy1:ty1, sx1:sx2, :3].reshape(-1, 3))
        if ty2 < sy2:
            ring_pixels.append(out[ty2:sy2, sx1:sx2, :3].reshape(-1, 3))
        # Left & right stripes of the middle band.
        if sx1 < tx1:
            ring_pixels.append(out[ty1:ty2, sx1:tx1, :3].reshape(-1, 3))
        if tx2 < sx2:
            ring_pixels.append(out[ty1:ty2, tx2:sx2, :3].reshape(-1, 3))
        if not ring_pixels:
            continue
        ring = np.concatenate(ring_pixels, axis=0)
        # Filter out fully transparent pixels by also looking at alpha in ring
        # (same slicing, but on alpha channel).
        # For simplicity use median of all ring samples.
        if ring.size == 0:
            continue
        median_color = np.median(ring, axis=0).astype(np.uint8)

        # Paint the text region with median color. Keep alpha from container.
        region_alpha = out[ty1:ty2, tx1:tx2, 3]
        # Only paint where the container was actually present (alpha > 0).
        mask = region_alpha > 0
        if not mask.any():
            # Text bbox area had no container pixels — force-fill anyway so
            # the rendered container shape has no visible hole.
            out[ty1:ty2, tx1:tx2, :3] = median_color
            out[ty1:ty2, tx1:tx2, 3] = 255
        else:
            out[ty1:ty2, tx1:tx2, :3] = np.where(
                mask[..., None], median_color, out[ty1:ty2, tx1:tx2, :3]
            )
            # Fill the text-hole alpha gaps so the container is solid.
            out[ty1:ty2, tx1:tx2, 3] = 255

    return out


def _contains(outer: Box, inner: Box, tolerance: float = 2.0) -> bool:
    """Return True if `outer` bbox encloses `inner` (with small tolerance)."""
    return (
        outer.x - tolerance <= inner.x
        and outer.y - tolerance <= inner.y
        and outer.x + outer.w + tolerance >= inner.x + inner.w
        and outer.y + outer.h + tolerance >= inner.y + inner.h
    )


def _map_vision_type_to_role(vtype: str | None) -> str | None:
    """Map vision-LLM element types to semantic roles."""
    if not vtype:
        return None
    mapping = {
        "button": "button",
        "card": "card",
        "badge": "button",  # badges are clickable-like
        "icon": "icon",
        "image": "illustration",
        "shape": "decoration",
        "decoration": "decoration",
    }
    return mapping.get(vtype)


def _build_container_hierarchy(
    layer_nodes: list, group_nodes: list, allocator
) -> None:
    """Assign each layer to its smallest enclosing container layer.

    A layer A is a container of B if:
      - A's bbox fully contains B's bbox (with small tolerance)
      - A is significantly larger (bbox area > 1.5× B's area)
      - A is not the background (full canvas)
      - A is non-text OR kind is "button"/"card"/"badge"/"image"

    Updates each child node's `children` field to reference the container.
    Appends container groups to `group_nodes`.
    """
    # Parent candidates: non-background, non-text-only elements that could
    # visually wrap other elements.
    containable_roles = {"card", "button", "badge", "illustration", "product_image"}
    candidates = []
    for node in layer_nodes:
        bb = node.bbox
        if bb.width <= 0 or bb.height <= 0:
            continue
        is_bg = (
            node.semantic_role == "background"
            or (bb.width >= 900 and bb.height >= 900 and node.type != "text")
        )
        if is_bg:
            continue
        role = node.semantic_role or ""
        if role in containable_roles or (node.type != "text" and role not in {"decoration", "icon"}):
            candidates.append(node)

    # Track parent assignments: child_id → parent_id
    parent_of: dict[str, str] = {}

    for child in layer_nodes:
        if child.semantic_role == "background":
            continue
        cb = child.bbox
        if cb.width <= 0 or cb.height <= 0:
            continue
        child_area = cb.width * cb.height

        # Find the smallest container that fully contains this child.
        best_parent: str | None = None
        best_parent_area = float("inf")
        for parent in candidates:
            if parent.id == child.id:
                continue
            pb = parent.bbox
            p_area = pb.width * pb.height
            if p_area < child_area * 1.5:
                continue
            # Containment check with 2px tolerance.
            if (
                pb.x <= cb.x + 2
                and pb.y <= cb.y + 2
                and pb.x + pb.width >= cb.x + cb.width - 2
                and pb.y + pb.height >= cb.y + cb.height - 2
            ):
                if p_area < best_parent_area:
                    best_parent_area = p_area
                    best_parent = parent.id

        if best_parent is not None:
            parent_of[child.id] = best_parent

    # Build container groups from parent_of mapping.
    parent_children: dict[str, list[str]] = {}
    for child_id, parent_id in parent_of.items():
        parent_children.setdefault(parent_id, []).append(child_id)

    node_by_id = {n.id: n for n in layer_nodes}
    for parent_id, child_ids in parent_children.items():
        parent_node = node_by_id[parent_id]
        group_id = allocator.allocate("container_group")
        group_nodes.append(LayerGroup(
            id=group_id,
            name=f"{parent_node.name or parent_id} container",
            layer_ids=[parent_id] + child_ids,
        ))
        # Update children refs on child layers.
        for cid in child_ids:
            child_node = node_by_id[cid]
            existing = child_node.children or []
            if group_id not in existing:
                child_node.children = existing + [group_id]


def _save_layer_png(rgba: np.ndarray, path) -> None:
    from ..utils.image_io import save_png

    save_png(rgba, path)


def _to_layer_type(kind: str) -> str:
    return kind if kind in ("text", "image", "vector_like", "unknown") else "unknown"


def _is_below(candidate: Box, anchor: Box | None) -> bool:
    if anchor is None:
        return False
    return candidate.y >= anchor.y + anchor.h * 0.6 and candidate.y - (anchor.y + anchor.h) < anchor.h * 2.5


def _infer_text_style(
    layer: PromotedLayer,
    rgb: np.ndarray,
    *,
    canvas_bg: str | None = None,
    canvas_w: int | None = None,
) -> StylePayload:
    # Font size proxy: bbox height (≈ cap-height + descender).
    font_size = max(8.0, layer.bbox.h * 0.9)
    patch_rgb = _slice_rgb(rgb, layer.bbox)

    # Foreground color = bucket farthest from background (more accurate than mean,
    # which gets pulled toward whatever fills the bbox padding).
    text_color = (
        extract_foreground_color(patch_rgb, bg_hex=canvas_bg)
        or average_color(patch_rgb)
        if patch_rgb.size
        else None
    )

    # Weight estimate from stroke density inside the bbox.
    weight = _estimate_font_weight(patch_rgb, text_color)

    # Alignment: where the bbox center sits relative to canvas center.
    text_align: str | None = None
    if canvas_w:
        cx = layer.bbox.x + layer.bbox.w / 2
        rel = cx / canvas_w
        if rel < 0.4:
            text_align = "left"
        elif rel > 0.6:
            text_align = "right"
        else:
            text_align = "center"

    # Line height: if multiple OCR lines, derive from line spacing.
    line_height: float | None = None
    if layer.text_lines and len(layer.text_lines) >= 2:
        ys = sorted(ln["bbox"]["y"] for ln in layer.text_lines)
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        if gaps:
            line_height = round(sum(gaps) / len(gaps) / max(1.0, font_size), 2)

    palette = dominant_colors(patch_rgb, k=3) if patch_rgb.size else []
    return StylePayload(
        font_size=round(font_size, 1),
        font_weight=weight,
        text_color=text_color,
        text_align=text_align,  # type: ignore[arg-type]
        line_height=line_height,
        dominant_colors=palette or None,
    )


def _infer_region_style(
    layer: PromotedLayer, rgb: np.ndarray, *, is_background: bool = False
) -> StylePayload | None:
    patch = _slice_rgb(rgb, layer.bbox)
    if patch.size == 0:
        return None
    # Background fill: sample only the outer border strip — the interior is
    # dominated by foreground content and will defeat solid/gradient detection.
    fill_dict = None
    if is_background:
        fill_dict = detect_solid_or_gradient_fill(_border_strip(patch))
    fill = None
    if fill_dict is not None:
        if fill_dict["type"] == "solid":
            fill = Fill(type="solid", color=fill_dict["color"])
        else:
            fill = Fill(
                type=fill_dict["type"],
                angle=fill_dict.get("angle"),
                stops=[GradientStop(**s) for s in fill_dict.get("stops", [])],
            )
    return StylePayload(
        dominant_colors=dominant_colors(patch, k=4) or None,
        fill=fill,
    )


def _estimate_font_weight(patch_rgb: np.ndarray, text_color: str | None) -> int:
    """Crude weight estimate from how much of the bbox matches the text color."""
    if patch_rgb.size == 0 or text_color is None:
        return 500
    from ..utils.color import hex_to_rgb

    target = np.array(hex_to_rgb(text_color), dtype=np.int32)
    diff = np.linalg.norm(patch_rgb.astype(np.int32) - target, axis=2)
    density = float((diff < 48).mean())
    # Stroke density buckets — empirical thresholds for display-size text.
    if density >= 0.45:
        return 900
    if density >= 0.30:
        return 700
    if density >= 0.18:
        return 600
    return 400


def _text_codegen_hints(role: str) -> CodegenHints:
    if role == "headline":
        return CodegenHints(component_candidate="Heading", text_role="title")
    if role == "subheadline":
        return CodegenHints(component_candidate="Heading", text_role="title")
    if role == "button":
        return CodegenHints(component_candidate="Button", text_role="button-label")
    if role == "body_text":
        return CodegenHints(component_candidate="Paragraph", text_role="paragraph")
    return CodegenHints(component_candidate="Text", text_role="caption")


def _border_strip(rgb: np.ndarray) -> np.ndarray:
    """Return the outer border of an image as a single tall strip.

    Stacks the four edges (top/bottom/left/right) into one (n, k, 3) array so
    detect_solid_or_gradient_fill can reason about uniformity vs. gradient on
    just the background-likely pixels.
    """
    h, w = rgb.shape[:2]
    if h < 4 or w < 4:
        return rgb
    bw = max(1, min(h, w) // 12)
    top = rgb[:bw]  # (bw, w, 3)
    bot = rgb[-bw:]
    # Reshape side strips to match width=bw orientation, then concat vertically.
    left = rgb[:, :bw].transpose(1, 0, 2)  # (bw, h, 3)
    right = rgb[:, -bw:].transpose(1, 0, 2)
    # Pad to common width.
    max_w = max(top.shape[1], left.shape[1])
    def _pad(a: np.ndarray) -> np.ndarray:
        if a.shape[1] == max_w:
            return a
        pad = np.zeros((a.shape[0], max_w - a.shape[1], 3), dtype=a.dtype)
        return np.concatenate([a, pad], axis=1)
    return np.concatenate([_pad(top), _pad(bot), _pad(left), _pad(right)], axis=0)


def _slice_rgb(rgb: np.ndarray, bbox: Box) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(w, int(round(bbox.x + bbox.w)))
    y2 = min(h, int(round(bbox.y + bbox.h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=rgb.dtype)
    return rgb[y1:y2, x1:x2]
