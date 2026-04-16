"""Evaluation metrics v2 (spec WS6).

Implements:
  - preview_diff_l1: L1 diff between reconstructed preview and original
  - alpha_iou: AlphaIoU over matched layer pairs
  - rgb_l1: RGB L1 averaged over canvas
  - layers_edit_dist: edit distance on (role, rough-bbox) sequences
  - style_fit_score_mean: mean of reconstruction confidences
  - ocr_reread_consistency_mean: mean OCR reread ratio for text layers
  - retry_success_rate: fraction of retry queue that improved
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class EvalMetrics:
    preview_diff: float
    alpha_iou: float
    rgb_l1: float
    layers_edit_dist: float
    style_fit_score_mean: float
    ocr_reread_consistency_mean: float
    retry_success_rate: float
    per_case: dict

    def to_dict(self) -> dict:
        return {
            "preview_diff": round(self.preview_diff, 4),
            "alpha_iou": round(self.alpha_iou, 4),
            "rgb_l1": round(self.rgb_l1, 4),
            "layers_edit_dist": round(self.layers_edit_dist, 4),
            "style_fit_score_mean": round(self.style_fit_score_mean, 4),
            "ocr_reread_consistency_mean": round(self.ocr_reread_consistency_mean, 4),
            "retry_success_rate": round(self.retry_success_rate, 4),
            "per_case": self.per_case,
        }


def compute_preview_diff(project_dir: Path) -> float:
    op = project_dir / "meta" / "original.png"
    pp = project_dir / "preview" / "preview.png"
    if not op.exists() or not pp.exists():
        return 0.0
    a = np.asarray(Image.open(op).convert("RGB"), dtype=np.int32)
    b_img = Image.open(pp).convert("RGB")
    if b_img.size != (a.shape[1], a.shape[0]):
        b_img = b_img.resize((a.shape[1], a.shape[0]))
    b = np.asarray(b_img, dtype=np.int32)
    return float(np.abs(a - b).mean() / 255.0)


def compute_rgb_l1(project_dir: Path) -> float:
    return compute_preview_diff(project_dir)  # alias


def compute_alpha_iou(manifest: dict, gt: dict) -> float:
    """Mean IoU between best-matched GT ↔ detected bboxes.
    Simplified when full alpha masks aren't available for GT.
    """
    gt_boxes = [(_box(l["bbox"]), l.get("text", "")) for l in gt.get("layers", [])]
    det_boxes = [(_box(l["bbox"]), (l.get("text") or {}).get("content", "")) for l in manifest.get("layers", [])]
    if not gt_boxes or not det_boxes:
        return 0.0
    used = set()
    ious: list[float] = []
    for gb, gt_text in gt_boxes:
        best = 0.0
        best_j = -1
        for j, (db, _) in enumerate(det_boxes):
            if j in used:
                continue
            v = _iou(gb, db)
            if v > best:
                best = v
                best_j = j
        if best_j >= 0:
            used.add(best_j)
        ious.append(best)
    return float(sum(ious) / len(ious))


def compute_layers_edit_dist(manifest: dict, gt: dict) -> float:
    """Edit distance on (role, rough-position-bucket) tokens, normalized."""
    def tokens(layers, scale_w, scale_h):
        out = []
        for l in layers:
            role = l.get("semanticRole") or l.get("role") or "unknown"
            bb = l["bbox"]
            col = int(bb["x"] / max(1, scale_w) * 4)
            row = int(bb["y"] / max(1, scale_h) * 4)
            out.append(f"{role}@{row},{col}")
        return out

    gt_w = gt["canvas"]["width"]
    gt_h = gt["canvas"]["height"]
    man_w = manifest["canvas"]["width"]
    man_h = manifest["canvas"]["height"]

    gt_tokens = tokens(gt.get("layers", []), gt_w, gt_h)
    det_tokens = tokens(manifest.get("layers", []), man_w, man_h)
    if not gt_tokens and not det_tokens:
        return 0.0
    d = _levenshtein(gt_tokens, det_tokens)
    denom = max(len(gt_tokens), len(det_tokens), 1)
    return float(d / denom)


def compute_style_fit_score(manifest: dict) -> float:
    scores = []
    for l in manifest.get("layers", []):
        style = l.get("style") or l.get("styleHints") or {}
        if "reconstructionConfidence" in style:
            scores.append(float(style["reconstructionConfidence"]))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def compute_ocr_reread_consistency(manifest: dict) -> float:
    """Approximation: average text confidence across text layers."""
    scores = []
    for l in manifest.get("layers", []):
        t = l.get("text") or {}
        if t.get("content") and t.get("confidence") is not None:
            scores.append(float(t["confidence"]))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def compute_retry_success_rate(project_dir: Path) -> float:
    log = project_dir / "debug" / "retry_log.json"
    if not log.exists():
        return 0.0
    import json
    try:
        data = json.loads(log.read_text())
    except Exception:
        return 0.0
    entries = data.get("entries", [])
    if not entries:
        return 0.0
    improved = sum(1 for e in entries if e.get("improved"))
    return float(improved / len(entries))


def _box(b: dict):
    from ..utils.bbox import Box
    return Box(x=float(b["x"]), y=float(b["y"]), w=float(b["width"]), h=float(b["height"]))


def _iou(a, b) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _levenshtein(a, b) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[-1]
