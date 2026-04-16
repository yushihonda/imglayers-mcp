"""Eval metrics v2 — compute the extended metric set per case.

Usage:
  python eval/metrics.py <run_tag>   # reads eval/runs/<tag>/ + project dirs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent
GT_DIR = ROOT / "groundtruth"
RUNS_DIR = ROOT / "runs"


def collect_metrics(tag: str) -> dict:
    from imglayers_mcp.core import metrics as M
    run_dir = RUNS_DIR / tag
    per_case: dict[str, dict] = {}

    for gt_file in sorted(GT_DIR.glob("*.json")):
        name = gt_file.stem
        manifest_path = run_dir / f"{name}.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        gt = json.loads(gt_file.read_text())

        project_id = manifest.get("projectId")
        project_dir = (
            Path("projects") / project_id
            if project_id else None
        )

        row = {
            "alpha_iou": M.compute_alpha_iou(manifest, gt),
            "rgb_l1": 0.0,
            "preview_diff": 0.0,
            "layers_edit_dist": M.compute_layers_edit_dist(manifest, gt),
            "style_fit_score_mean": M.compute_style_fit_score(manifest),
            "ocr_reread_consistency_mean": M.compute_ocr_reread_consistency(manifest),
            "retry_success_rate": 0.0,
            "image_type": manifest.get("pipeline", {}).get("imageType", "unknown"),
        }
        if project_dir and project_dir.exists():
            row["preview_diff"] = M.compute_preview_diff(project_dir)
            row["rgb_l1"] = M.compute_rgb_l1(project_dir)
            row["retry_success_rate"] = M.compute_retry_success_rate(project_dir)

        per_case[name] = row

    # Aggregate.
    if not per_case:
        return {"per_case": {}, "summary": {}, "by_type": {}}
    keys = [k for k in next(iter(per_case.values())).keys() if k != "image_type"]
    summary = {k: sum(c[k] for c in per_case.values()) / len(per_case) for k in keys}

    # Per image-type breakdown.
    by_type: dict[str, dict] = {}
    for name, row in per_case.items():
        t = row["image_type"]
        by_type.setdefault(t, {"_cases": []})["_cases"].append(name)
    for t, v in by_type.items():
        cases = v["_cases"]
        v["count"] = len(cases)
        for k in keys:
            v[k] = sum(per_case[c][k] for c in cases) / len(cases)

    return {"per_case": per_case, "summary": summary, "by_type": by_type}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = collect_metrics(args.tag)
    summary = data["summary"]
    by_type = data["by_type"]

    print("=" * 60)
    print(f"Metrics v2 — {args.tag}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:32s}: {v:.4f}")
    print()
    for t, v in sorted(by_type.items()):
        print(f"[{t}] ({v['count']} cases)")
        for k in summary:
            print(f"  {k:30s}: {v[k]:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
