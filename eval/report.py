"""Generate a side-by-side ablation report for multiple eval runs.

Usage:
  python eval/report.py tag1 tag2 tag3
  python eval/report.py --out eval/report.md baseline v7-stage-driven v8-phase-a v9-all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.metrics import collect_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows: list[tuple[str, dict]] = []
    for tag in args.tags:
        data = collect_metrics(tag)
        rows.append((tag, data["summary"]))

    if not rows:
        print("No data.")
        return

    keys = list(rows[0][1].keys())

    lines: list[str] = []
    lines.append("# Ablation report\n")
    lines.append("| metric | " + " | ".join(tag for tag, _ in rows) + " |")
    lines.append("|---|" + "|".join(["---"] * len(rows)) + "|")
    for k in keys:
        row_vals = [f"{r[1].get(k, 0):.4f}" for r in rows]
        lines.append(f"| **{k}** | " + " | ".join(row_vals) + " |")

    # Per-type sections.
    all_by_type: list[tuple[str, dict]] = []
    for tag in args.tags:
        all_by_type.append((tag, collect_metrics(tag)["by_type"]))

    lines.append("\n## By image_type\n")
    types = set()
    for tag, bt in all_by_type:
        types.update(bt.keys())
    for t in sorted(types):
        lines.append(f"### {t}")
        lines.append("| metric | " + " | ".join(tag for tag, _ in all_by_type) + " |")
        lines.append("|---|" + "|".join(["---"] * len(all_by_type)) + "|")
        for k in keys:
            vals = [
                f"{bt[t][k]:.4f}" if t in bt else "-"
                for _, bt in all_by_type
            ]
            lines.append(f"| {k} | " + " | ".join(vals) + " |")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
