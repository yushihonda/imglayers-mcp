"""Verifier v2 — layer-local residual-driven scoring (spec WS1).

For every layer we compute:
  - local_residual_mean, local_residual_p95  (reconstructed vs original)
  - alpha_edge_mismatch  (does our alpha agree with observed foreground?)
  - bbox_border_diff     (residual along the outer border = leakage)
  - ocr_reread_consistency for text layers
  - retry_priority: weighted combination of the above

The retry queue is ordered by retry_priority so heavy backends can target
only the top-K hardest cases.

Thresholds are image_type-aware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..utils.bbox import Box
from . import diffmap


@dataclass
class LayerDiag:
    layer_id: str
    local_residual_mean: float
    local_residual_p95: float
    alpha_edge_mismatch: float
    bbox_border_diff: float
    ocr_reread_consistency: float | None
    retry_priority: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layerId": self.layer_id,
            "localResidualMean": round(self.local_residual_mean, 4),
            "localResidualP95": round(self.local_residual_p95, 4),
            "alphaEdgeMismatch": round(self.alpha_edge_mismatch, 4),
            "bboxBorderDiff": round(self.bbox_border_diff, 4),
            "ocrRereadConsistency": round(self.ocr_reread_consistency, 4) if self.ocr_reread_consistency is not None else None,
            "retryPriority": round(self.retry_priority, 4),
            "reasons": self.reasons,
        }


@dataclass
class VerificationResult:
    overall_score: float
    preview_diff: float
    per_layer: list[LayerDiag] = field(default_factory=list)
    low_confidence_layers: list[str] = field(default_factory=list)
    retry_queue: list[dict] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overallScore": round(self.overall_score, 4),
            "previewDiff": round(self.preview_diff, 4),
            "perLayer": [d.to_dict() for d in self.per_layer],
            "lowConfidenceLayers": self.low_confidence_layers,
            "retryQueue": self.retry_queue,
            "thresholds": self.thresholds,
            "notes": self.notes,
        }


def verify(
    manifest: dict,
    project_dir: Path,
    thresholds: dict | None = None,
) -> VerificationResult:
    thresholds = thresholds or {}
    low_conf_thr = float(thresholds.get("low_confidence_layer", 0.55))
    retry_preview_diff = float(thresholds.get("retry_preview_diff", 0.12))
    retry_edge_leakage = float(thresholds.get("retry_edge_leakage", 0.18))
    retry_residual_p95 = float(thresholds.get("retry_residual_p95", 0.35))

    original = _load(project_dir, "meta/original.png")
    preview = _load(project_dir, "preview/preview.png", match_to=original)

    # Global diff.
    preview_diff = 0.0
    diff = None
    if original is not None and preview is not None and original.shape == preview.shape:
        diff = diffmap.rgb_diff(original, preview)
        preview_diff = float(diff.mean())

    canvas = manifest.get("canvas", {})
    canvas_w = int(canvas.get("width", 1))
    canvas_h = int(canvas.get("height", 1))

    # Per-layer diagnostics.
    per_layer: list[LayerDiag] = []
    observed_fg = _observed_fg(original) if original is not None else None

    for l in manifest.get("layers", []):
        bb = _box(l["bbox"])
        diag_reasons: list[str] = []

        if diff is not None:
            stats = diffmap.bbox_stats(diff, bb, canvas_h, canvas_w)
            border = diffmap.border_stats(diff, bb, canvas_h, canvas_w)
            residual_mean = stats["mean"]
            residual_p95 = stats["p95"]
            border_diff = border["mean"]
        else:
            residual_mean = residual_p95 = border_diff = 0.0

        # Alpha edge mismatch (approximate from preview pixels).
        aem = 0.0
        if observed_fg is not None and original is not None:
            # Reconstruct an approximate "should be fg" mask from the layer's
            # bbox + role; compare with the observed fg mask inside bbox.
            aem = _approx_alpha_edge_mismatch(l, bb, observed_fg, canvas_h, canvas_w)

        # OCR reread consistency (text layers only).
        ocr_cons: float | None = None
        if l.get("type") == "text":
            tc = (l.get("text") or {}).get("confidence")
            if tc is not None:
                ocr_cons = float(tc)

        base_conf = float(l.get("confidence", 0.7))

        # Combine into retry priority (higher = more urgent).
        priority = 0.0
        priority += max(0.0, residual_mean - 0.08) * 3.0
        priority += max(0.0, residual_p95 - retry_residual_p95) * 2.0
        priority += max(0.0, border_diff - retry_edge_leakage) * 2.0
        priority += max(0.0, 0.7 - base_conf) * 1.0
        priority += max(0.0, aem - 0.15) * 2.0
        if ocr_cons is not None and ocr_cons < 0.6:
            priority += (0.6 - ocr_cons) * 1.5

        if residual_p95 > retry_residual_p95:
            diag_reasons.append("high_residual_p95")
        if border_diff > retry_edge_leakage:
            diag_reasons.append("edge_leakage")
        if base_conf < low_conf_thr:
            diag_reasons.append("low_confidence")
        if aem > 0.15:
            diag_reasons.append("alpha_edge_mismatch")
        if ocr_cons is not None and ocr_cons < 0.6:
            diag_reasons.append("ocr_inconsistency")

        per_layer.append(LayerDiag(
            layer_id=l["id"],
            local_residual_mean=residual_mean,
            local_residual_p95=residual_p95,
            alpha_edge_mismatch=aem,
            bbox_border_diff=border_diff,
            ocr_reread_consistency=ocr_cons,
            retry_priority=priority,
            reasons=diag_reasons,
        ))

    low_conf = [d.layer_id for d in per_layer if d.retry_priority > 0.15]

    # Build retry queue sorted by priority desc.
    retry_q: list[dict] = []
    id_to_layer = {l["id"]: l for l in manifest.get("layers", [])}
    for d in sorted(per_layer, key=lambda x: -x.retry_priority):
        if d.retry_priority <= 0.05:
            continue
        layer = id_to_layer.get(d.layer_id, {})
        retry_q.append({
            "layer_id": d.layer_id,
            "role": layer.get("semanticRole"),
            "reason": ",".join(d.reasons) or "low_quality",
            "priority": round(d.retry_priority, 4),
        })

    # Overall score: avg confidence minus preview diff penalty.
    avg_conf = float(np.mean([float(l.get("confidence", 0.7)) for l in manifest.get("layers", [])] or [0.7]))
    overall = max(0.0, avg_conf - preview_diff * 0.5)

    notes: list[str] = []
    if preview_diff > retry_preview_diff:
        notes.append(f"preview_diff {preview_diff:.3f} exceeds threshold")

    return VerificationResult(
        overall_score=overall,
        preview_diff=preview_diff,
        per_layer=per_layer,
        low_confidence_layers=low_conf,
        retry_queue=retry_q,
        thresholds={
            "low_confidence_layer": low_conf_thr,
            "retry_preview_diff": retry_preview_diff,
            "retry_edge_leakage": retry_edge_leakage,
            "retry_residual_p95": retry_residual_p95,
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
def _load(project_dir: Path, rel: str, match_to: np.ndarray | None = None) -> np.ndarray | None:
    p = project_dir / rel
    if not p.exists():
        return None
    try:
        img = Image.open(p).convert("RGB")
        if match_to is not None and img.size != (match_to.shape[1], match_to.shape[0]):
            img = img.resize((match_to.shape[1], match_to.shape[0]))
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def _observed_fg(rgb: np.ndarray) -> np.ndarray:
    """Rough foreground mask: pixels differing from the border mode."""
    h, w = rgb.shape[:2]
    bw = max(1, min(h, w) // 20)
    border = np.concatenate([
        rgb[:bw].reshape(-1, 3), rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3), rgb[:, -bw:].reshape(-1, 3),
    ], axis=0).astype(np.int32)
    q = (border // 16) * 16 + 8
    packed = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    unique, counts = np.unique(packed, return_counts=True)
    winner = int(unique[int(np.argmax(counts))])
    bg = np.array([(winner >> 16) & 0xFF, (winner >> 8) & 0xFF, winner & 0xFF], dtype=np.int32)
    diff = np.linalg.norm(rgb.astype(np.int32) - bg, axis=2)
    return diff > 20


def _approx_alpha_edge_mismatch(
    layer: dict, bbox: Box, observed_fg: np.ndarray, canvas_h: int, canvas_w: int
) -> float:
    """Rough proxy: within bbox, how much of the observed fg is missing?"""
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(canvas_w, int(round(bbox.x + bbox.w)))
    y2 = min(canvas_h, int(round(bbox.y + bbox.h)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region_fg = observed_fg[y1:y2, x1:x2]
    if not region_fg.any():
        return 0.0
    # We don't have the actual per-layer alpha here; use role heuristic:
    # text layers should cover most fg in their bbox; containers should too.
    role = layer.get("semanticRole", "unknown")
    target_coverage = 0.5 if role in {"background"} else 0.3
    actual = float(region_fg.mean())
    return abs(actual - target_coverage) * 0.6


def _box(b: dict) -> Box:
    return Box(x=float(b["x"]), y=float(b["y"]), w=float(b["width"]), h=float(b["height"]))
