"""Convert vision-derived RawLayers into PromotedLayers for the manifest.

Vision-only pipeline: the vision model already provides correct bboxes and
text classification. The merger's remaining job is to apply text_granularity
(line/block/char) and produce TextGroupInfo for grouping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.bbox import Box, union_box
from .types import RawLayer


@dataclass(eq=False)
class PromotedLayer:
    rgba: np.ndarray
    bbox: Box
    kind: str
    engines: list[str]
    confidence: float
    z_index: int
    text_content: str | None = None
    text_language: str | None = None
    text_confidence: float | None = None
    text_lines: list[dict] = field(default_factory=list)
    group_tag: str | None = None
    # Hint from vision LLM: "button", "card", "badge", "icon", etc.
    # When present, naming.py prefers this over heuristic role inference.
    semantic_hint: str | None = None


@dataclass
class TextGroupInfo:
    tag: str
    text: str
    bbox: Box
    language: str | None = None
    child_tags: list[str] = field(default_factory=list)


@dataclass
class MergeResult:
    layers: list[PromotedLayer]
    text_groups: list[TextGroupInfo]


def merge(
    raw_layers: list[RawLayer],
    *,
    canvas_w: int,
    canvas_h: int,
    text_granularity: str = "line",
    text_contents: dict[int, str] | None = None,
) -> MergeResult:
    """Convert RawLayers to PromotedLayers.

    text_contents: maps raw_layer index -> text string (from vision model).
    text_granularity:
      - "line": one layer per text element (default)
      - "block": cluster nearby text lines into paragraph groups
      - "char": split each text line into per-character layers
    """
    promoted: list[PromotedLayer] = []
    text_groups: list[TextGroupInfo] = []
    text_contents = text_contents or {}

    text_indices: list[int] = []
    nontext_indices: list[int] = []
    for i, raw in enumerate(raw_layers):
        if raw.kind == "text":
            text_indices.append(i)
        else:
            nontext_indices.append(i)

    # Dedup non-text layers by IoU, but exclude the background (full canvas).
    # iou_thr=0.85 is conservative: only near-identical pairs get merged.
    dedup_candidates: list[int] = []
    bg_indices: list[int] = []
    for i in nontext_indices:
        bb = raw_layers[i].bbox
        is_full = bb.w >= canvas_w * 0.95 and bb.h >= canvas_h * 0.95
        if is_full:
            bg_indices.append(i)
        else:
            dedup_candidates.append(i)
    dedup_nontext = bg_indices + _dedup_bbox(raw_layers, dedup_candidates, iou_thr=0.85)

    for i in dedup_nontext:
        raw = raw_layers[i]
        hint = (raw.debug or {}).get("vision_type")
        promoted.append(PromotedLayer(
            rgba=raw.rgba, bbox=raw.bbox, kind=raw.kind,
            engines=[raw.engine], confidence=raw.confidence, z_index=raw.z_index,
            semantic_hint=hint,
        ))

    if text_granularity == "char":
        _emit_char(raw_layers, text_indices, text_contents, promoted, text_groups, canvas_w, canvas_h)
    elif text_granularity == "block":
        _emit_block(raw_layers, text_indices, text_contents, promoted, text_groups)
    else:
        _emit_line(raw_layers, text_indices, text_contents, promoted, text_groups)

    sorted_layers = _sort_z(promoted, canvas_w, canvas_h)
    return MergeResult(layers=sorted_layers, text_groups=text_groups)


def _emit_line(
    raw_layers: list[RawLayer],
    text_indices: list[int],
    text_contents: dict[int, str],
    promoted: list[PromotedLayer],
    text_groups: list[TextGroupInfo],
) -> None:
    """Emit one layer per text RawLayer. Cluster lines into block groups."""
    layer_tags: dict[int, str] = {}
    for idx, ri in enumerate(text_indices):
        raw = raw_layers[ri]
        tag = f"line_{idx}"
        layer_tags[ri] = tag
        text = text_contents.get(ri, "")
        promoted.append(PromotedLayer(
            rgba=raw.rgba, bbox=raw.bbox, kind="text",
            engines=[raw.engine], confidence=raw.confidence, z_index=raw.z_index,
            text_content=text, text_content_confidence=None,
            text_lines=[{"text": text, "bbox": raw.bbox.to_dict()}] if text else [],
            group_tag=tag,
        ) if False else PromotedLayer(
            rgba=raw.rgba, bbox=raw.bbox, kind="text",
            engines=[raw.engine], confidence=raw.confidence, z_index=raw.z_index,
            text_content=text,
            text_lines=[{"text": text, "bbox": raw.bbox.to_dict()}] if text else [],
            group_tag=tag,
        ))

    # Block-level clustering
    clusters = _cluster_text_lines([(ri, raw_layers[ri].bbox, text_contents.get(ri, "")) for ri in text_indices])
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        child_tags = [layer_tags[ri] for ri, _, _ in cluster if ri in layer_tags]
        bbox = union_box([b for _, b, _ in cluster]) or cluster[0][1]
        text_groups.append(TextGroupInfo(
            tag=f"textblock_{len(text_groups)}",
            text="\n".join(t for _, _, t in cluster),
            bbox=bbox,
            child_tags=child_tags,
        ))


def _emit_block(
    raw_layers: list[RawLayer],
    text_indices: list[int],
    text_contents: dict[int, str],
    promoted: list[PromotedLayer],
    text_groups: list[TextGroupInfo],
) -> None:
    """Composite nearby text lines into single paragraph layers."""
    clusters = _cluster_text_lines([(ri, raw_layers[ri].bbox, text_contents.get(ri, "")) for ri in text_indices])
    for cluster in clusters:
        indices = [ri for ri, _, _ in cluster]
        if len(indices) == 1:
            ri = indices[0]
            raw = raw_layers[ri]
            text = text_contents.get(ri, "")
            promoted.append(PromotedLayer(
                rgba=raw.rgba, bbox=raw.bbox, kind="text",
                engines=[raw.engine], confidence=raw.confidence, z_index=raw.z_index,
                text_content=text,
                text_lines=[{"text": text, "bbox": raw.bbox.to_dict()}] if text else [],
            ))
            continue
        # Composite multiple lines into one layer.
        canvas_h, canvas_w = raw_layers[indices[0]].rgba.shape[:2]
        rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        for ri in indices:
            m = raw_layers[ri].rgba[..., 3] > 0
            rgba[m] = raw_layers[ri].rgba[m]
        bbox = union_box([raw_layers[ri].bbox for ri in indices]) or raw_layers[indices[0]].bbox
        content = "\n".join(text_contents.get(ri, "") for ri in indices)
        promoted.append(PromotedLayer(
            rgba=rgba, bbox=bbox, kind="text",
            engines=[raw_layers[indices[0]].engine], confidence=0.9,
            z_index=raw_layers[indices[0]].z_index,
            text_content=content,
            text_lines=[{"text": text_contents.get(ri, ""), "bbox": raw_layers[ri].bbox.to_dict()} for ri in indices],
        ))


def _emit_char(
    raw_layers: list[RawLayer],
    text_indices: list[int],
    text_contents: dict[int, str],
    promoted: list[PromotedLayer],
    text_groups: list[TextGroupInfo],
    canvas_w: int,
    canvas_h: int,
) -> None:
    """Split each text RawLayer into per-character layers via CC."""
    from ._cc import connected_components

    global_tag_idx = 0
    for ri in text_indices:
        raw = raw_layers[ri]
        text = text_contents.get(ri, "")
        chars = [ch for ch in text if not ch.isspace()]
        n_chars = len(chars)

        alpha = raw.rgba[..., 3]
        cc_mask = alpha > 0
        comps = connected_components(cc_mask)
        min_area = max(8, int(raw.bbox.w * raw.bbox.h) // 500)
        comps = [c for c in comps if c["area"] >= min_area]
        comps.sort(key=lambda c: c["x1"])

        sub_raws: list[RawLayer] = []
        for comp in comps:
            y1, y2, x1, x2 = comp["y1"], comp["y2"], comp["x1"], comp["x2"]
            sub_canvas = np.zeros_like(raw.rgba)
            mask = comp["mask"]
            sub_canvas[mask] = raw.rgba[mask]
            sub_raws.append(RawLayer(
                rgba=sub_canvas,
                bbox=Box(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                kind="text", engine=raw.engine, confidence=raw.confidence, z_index=raw.z_index,
            ))

        indices = list(range(len(sub_raws)))
        indices = _merge_vertical_parts(indices, sub_raws, line_bbox=raw.bbox)

        if len(indices) > n_chars and n_chars > 0:
            areas = [sub_raws[i].bbox.area for i in indices]
            median_area = float(sorted(areas)[len(areas) // 2])
            threshold = median_area * 0.15
            major = [i for i in indices if sub_raws[i].bbox.area >= threshold or
                     (min(sub_raws[i].bbox.w, sub_raws[i].bbox.h) / max(sub_raws[i].bbox.w, sub_raws[i].bbox.h, 1) >= 0.5)]
            minor = [i for i in indices if i not in major]
            for mi in minor:
                if not major:
                    major.append(mi)
                    continue
                mx = sub_raws[mi].bbox.x + sub_raws[mi].bbox.w / 2
                nearest = min(major, key=lambda j: abs(sub_raws[j].bbox.x + sub_raws[j].bbox.w / 2 - mx))
                _merge_cc_into(sub_raws, mi, nearest)
            indices = sorted(major, key=lambda i: sub_raws[i].bbox.x)

        child_tags: list[str] = []
        for i_idx, si in enumerate(indices):
            sr = sub_raws[si]
            tag = f"char_{global_tag_idx}"
            child_tags.append(tag)
            ch = chars[i_idx] if i_idx < len(chars) else ""
            promoted.append(PromotedLayer(
                rgba=sr.rgba, bbox=sr.bbox, kind="text",
                engines=[sr.engine], confidence=sr.confidence, z_index=sr.z_index,
                text_content=ch,
                text_lines=[{"text": ch, "bbox": sr.bbox.to_dict()}],
                group_tag=tag,
            ))
            global_tag_idx += 1

        text_groups.append(TextGroupInfo(
            tag=f"textline_{len(text_groups)}", text=text, bbox=raw.bbox,
            child_tags=child_tags,
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _dedup_bbox(
    raw_layers: list[RawLayer], indices: list[int], iou_thr: float = 0.7,
) -> list[int]:
    """Remove duplicate bboxes; prefer the larger of near-duplicate pairs."""
    kept: list[int] = []
    sorted_indices = sorted(indices, key=lambda i: -raw_layers[i].bbox.area)
    for i in sorted_indices:
        dup = False
        bi = raw_layers[i].bbox
        for j in kept:
            bj = raw_layers[j].bbox
            if _bbox_iou(bi, bj) > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(i)
    return kept


def _cluster_text_lines(entries: list[tuple[int, Box, str]]) -> list[list[tuple[int, Box, str]]]:
    """Group text lines that are vertically close with strong horizontal overlap.

    Stricter thresholds than simple clustering: we require 60%+ horizontal
    overlap and a vertical gap under 30% of the reference line's height,
    so unrelated text blocks (e.g. heading and CTA far apart) don't merge.
    """
    if not entries:
        return []
    sorted_entries = sorted(entries, key=lambda e: (e[1].y, e[1].x))
    groups: list[list[tuple[int, Box, str]]] = []
    for entry in sorted_entries:
        _, bbox, _ = entry
        placed = False
        for g in groups:
            _, ref_bbox, _ = g[-1]
            vgap = bbox.y - (ref_bbox.y + ref_bbox.h)
            h_overlap = min(bbox.x + bbox.w, ref_bbox.x + ref_bbox.w) - max(bbox.x, ref_bbox.x)
            min_w = min(bbox.w, ref_bbox.w)
            ref_h = max(1.0, ref_bbox.h)
            # Require substantial horizontal overlap relative to the narrower line.
            if (
                vgap <= ref_h * 0.4
                and vgap >= -ref_h * 0.3
                and min_w > 0
                and h_overlap / min_w >= 0.5
            ):
                g.append(entry)
                placed = True
                break
        if not placed:
            groups.append([entry])
    return groups


def _merge_vertical_parts(ccs: list[int], sub_raws: list[RawLayer], *, line_bbox: Box | None = None) -> list[int]:
    if len(ccs) <= 1:
        return ccs
    remaining = list(ccs)
    areas = [sub_raws[ci].bbox.area for ci in remaining]
    median_area = float(sorted(areas)[len(areas) // 2]) if areas else 1
    dots: list[int] = []
    for ci in remaining:
        b = sub_raws[ci].bbox
        if b.w <= 0 or b.h <= 0:
            continue
        aspect = min(b.w, b.h) / max(b.w, b.h)
        if aspect >= 0.6 and b.area < median_area * 0.4:
            dots.append(ci)

    for dot_ci in dots:
        db = sub_raws[dot_ci].bbox
        dot_cx = db.x + db.w / 2
        best_ci: int | None = None
        best_gap = float("inf")
        for ci in remaining:
            if ci == dot_ci:
                continue
            sb = sub_raws[ci].bbox
            if dot_cx < sb.x or dot_cx > sb.x + sb.w:
                continue
            gap = max(0.0, max(db.y, sb.y) - min(db.y + db.h, sb.y + sb.h))
            max_h = max(db.h, sb.h)
            if max_h > 0 and gap / max_h >= 0.6:
                continue
            if line_bbox is not None:
                merged_y1 = min(db.y, sb.y)
                merged_y2 = max(db.y + db.h, sb.y + sb.h)
                margin = max(30, line_bbox.h * 0.2)
                if merged_y1 < line_bbox.y - margin or merged_y2 > line_bbox.y + line_bbox.h + margin:
                    continue
            if gap < best_gap:
                best_gap = gap
                best_ci = ci
        if best_ci is not None:
            _merge_cc_into(sub_raws, dot_ci, best_ci)
            remaining.remove(dot_ci)

    return sorted(remaining, key=lambda ci: sub_raws[ci].bbox.x)


def _merge_cc_into(sub_raws: list[RawLayer], src: int, dst: int) -> None:
    m_rgba = sub_raws[src].rgba
    n_rgba = sub_raws[dst].rgba
    mask = m_rgba[..., 3] > 0
    n_rgba[mask] = m_rgba[mask]
    nb = sub_raws[dst].bbox
    mb = sub_raws[src].bbox
    nx1 = min(nb.x, mb.x)
    ny1 = min(nb.y, mb.y)
    nx2 = max(nb.x + nb.w, mb.x + mb.w)
    ny2 = max(nb.y + nb.h, mb.y + mb.h)
    sub_raws[dst].bbox = Box(x=nx1, y=ny1, w=nx2 - nx1, h=ny2 - ny1)


def _sort_z(layers: list[PromotedLayer], canvas_w: int, canvas_h: int) -> list[PromotedLayer]:
    def is_bg(l: PromotedLayer) -> bool:
        return (
            l.bbox.w >= canvas_w * 0.95
            and l.bbox.h >= canvas_h * 0.95
            and l.kind != "text"
        )

    backgrounds = [l for l in layers if is_bg(l)]
    texts = [l for l in layers if l.kind == "text"]
    bg_ids = {id(l) for l in backgrounds}
    text_ids = {id(l) for l in texts}
    others = [l for l in layers if id(l) not in bg_ids and id(l) not in text_ids]
    others.sort(key=lambda l: -(l.bbox.w * l.bbox.h))
    texts.sort(key=lambda l: (l.bbox.y, l.bbox.x))

    ordered = backgrounds + others + texts
    for idx, layer in enumerate(ordered):
        layer.z_index = idx
    return ordered
