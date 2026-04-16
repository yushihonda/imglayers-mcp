"""Eval scorer: compare manifest against ground truth.

Metrics:
  - text_recall: fraction of GT text strings found in detected layers
  - text_precision: fraction of detected texts matching GT
  - text_f1: harmonic mean
  - bbox_iou_mean: average IoU of best-matched pairs
  - role_accuracy: fraction of GT layers with correctly-inferred role
  - layer_count_ratio: detected / expected (closer to 1.0 is better)
  - overall_score: weighted combination
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
GT_DIR = ROOT / "groundtruth"


def iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def norm_text(s: str) -> str:
    return "".join(s.lower().split())


def score_case(gt: dict, manifest: dict) -> dict[str, float]:
    canvas = manifest.get("canvas", {}) or {}
    canvas_w = int(canvas.get("width", 1))
    canvas_h = int(canvas.get("height", 1))
    gt_w = gt["canvas"]["width"]
    gt_h = gt["canvas"]["height"]
    sx = canvas_w / gt_w
    sy = canvas_h / gt_h

    gt_layers = []
    for l in gt["layers"]:
        b = l["bbox"]
        gt_layers.append({
            "role": l["role"],
            "bbox": {"x": b["x"] * sx, "y": b["y"] * sy, "width": b["width"] * sx, "height": b["height"] * sy},
            "text": l.get("text", ""),
        })

    det_layers = []
    for l in manifest.get("layers", []):
        b = l["bbox"]
        det_layers.append({
            "role": l.get("semanticRole", "unknown"),
            "bbox": b,
            "text": (l.get("text") or {}).get("content", "") or "",
        })

    # Text matching
    gt_texts = [norm_text(l["text"]) for l in gt_layers if l["text"]]
    det_texts = [norm_text(l["text"]) for l in det_layers if l["text"]]
    gt_text_set = set(gt_texts)
    det_text_set = set(det_texts)

    # Recall: fraction of GT texts found (substring match in any detected)
    found = 0
    det_concat = " | ".join(det_texts)
    for gt_t in gt_texts:
        if gt_t and gt_t in det_concat:
            found += 1
    text_recall = found / len(gt_texts) if gt_texts else 1.0

    # Precision: detected texts that match any GT (substring in GT)
    matched = 0
    gt_concat = " | ".join(gt_texts)
    for d_t in det_texts:
        if d_t and d_t in gt_concat:
            matched += 1
    text_precision = matched / len(det_texts) if det_texts else 1.0
    text_f1 = (2 * text_precision * text_recall / (text_precision + text_recall)) if (text_precision + text_recall) > 0 else 0.0

    # Bbox matching (Hungarian-lite: greedy best match)
    ious: list[float] = []
    used_det = set()
    for gt_l in gt_layers:
        best_iou = 0.0
        best_idx = -1
        for i, d in enumerate(det_layers):
            if i in used_det:
                continue
            score = iou(gt_l["bbox"], d["bbox"])
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_idx >= 0:
            used_det.add(best_idx)
        ious.append(best_iou)
    bbox_iou_mean = sum(ious) / len(ious) if ious else 0.0

    # Role accuracy (for matched pairs with IoU > 0.3)
    role_correct = 0
    role_total = 0
    used_det = set()
    for gt_l in gt_layers:
        best_iou = 0.0
        best_idx = -1
        for i, d in enumerate(det_layers):
            if i in used_det:
                continue
            s = iou(gt_l["bbox"], d["bbox"])
            if s > best_iou:
                best_iou = s
                best_idx = i
        if best_idx >= 0 and best_iou > 0.3:
            used_det.add(best_idx)
            role_total += 1
            if _role_compatible(gt_l["role"], det_layers[best_idx]["role"]):
                role_correct += 1
    role_accuracy = role_correct / role_total if role_total > 0 else 0.0

    # Layer count ratio (penalize both over- and under-detection)
    gt_n = len(gt_layers)
    det_n = len(det_layers)
    if gt_n == 0:
        layer_count_ratio = 1.0
    else:
        ratio = det_n / gt_n
        # 1.0 is perfect, 0.5 or 2.0 give 0.5
        layer_count_ratio = 1.0 - min(1.0, abs(ratio - 1.0) / 2.0)

    overall = (
        text_f1 * 0.35
        + bbox_iou_mean * 0.25
        + role_accuracy * 0.25
        + layer_count_ratio * 0.15
    )

    return {
        "text_recall": text_recall,
        "text_precision": text_precision,
        "text_f1": text_f1,
        "bbox_iou_mean": bbox_iou_mean,
        "role_accuracy": role_accuracy,
        "layer_count_ratio": layer_count_ratio,
        "overall": overall,
        "gt_n": gt_n,
        "det_n": det_n,
    }


def _role_compatible(gt_role: str, det_role: str) -> bool:
    """Consider equivalent roles as correct matches."""
    if gt_role == det_role:
        return True
    # Treat similar text roles as interchangeable.
    text_group = {"headline", "subheadline", "body_text"}
    if gt_role in text_group and det_role in text_group:
        return True
    img_group = {"illustration", "product_image", "image"}
    if gt_role in img_group and det_role in img_group:
        return True
    return False


def run_all(project_scores: dict[str, dict]) -> dict:
    """Aggregate per-case scores into totals."""
    if not project_scores:
        return {}
    keys = ["text_f1", "text_recall", "text_precision", "bbox_iou_mean",
            "role_accuracy", "layer_count_ratio", "overall"]
    summary = {k: sum(s[k] for s in project_scores.values()) / len(project_scores) for k in keys}
    summary["cases"] = len(project_scores)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests_dir", help="Directory of manifest JSON files (same names as GT)")
    ap.add_argument("--out", default=None, help="Write per-case scores as JSON")
    args = ap.parse_args()

    manifests_dir = Path(args.manifests_dir)
    scores: dict[str, dict] = {}
    for gt_file in sorted(GT_DIR.glob("*.json")):
        name = gt_file.stem
        manifest_file = manifests_dir / f"{name}.json"
        if not manifest_file.exists():
            print(f"  {name}: SKIP (no manifest)")
            continue
        gt = json.loads(gt_file.read_text())
        manifest = json.loads(manifest_file.read_text())
        s = score_case(gt, manifest)
        scores[name] = s
        print(f"  {name}: overall={s['overall']:.3f}  text_f1={s['text_f1']:.2f}  iou={s['bbox_iou_mean']:.2f}  role={s['role_accuracy']:.2f}  count={s['det_n']}/{s['gt_n']}")

    print()
    summary = run_all(scores)
    print(f"TOTAL (avg over {summary['cases']} cases):")
    print(f"  overall:         {summary['overall']:.3f}")
    print(f"  text_f1:         {summary['text_f1']:.3f}  (recall={summary['text_recall']:.3f}, precision={summary['text_precision']:.3f})")
    print(f"  bbox_iou_mean:   {summary['bbox_iou_mean']:.3f}")
    print(f"  role_accuracy:   {summary['role_accuracy']:.3f}")
    print(f"  layer_count:     {summary['layer_count_ratio']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps({"per_case": scores, "summary": summary}, indent=2))
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
