"""Run the current decompose_image pipeline on all eval images.

Usage:
  python eval/run.py <tag>

Writes manifests to eval/runs/<tag>/<name>.json
Then: python eval/score.py eval/runs/<tag>/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
IMG_DIR = ROOT / "images"
RUNS_DIR = ROOT / "runs"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="Run tag (e.g. 'baseline', 'ocr-preprocess')")
    ap.add_argument("--granularity", default="line")
    ap.add_argument("--detail", default="high")
    ap.add_argument("--engine", default="hybrid", choices=["layerd", "sam2", "hybrid"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = RUNS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    from imglayers_mcp.server import build_runtime
    tools, router, orchestrator, store = build_runtime()

    for img_file in sorted(IMG_DIR.glob("*.png")):
        name = img_file.stem
        print(f"  {name}...", end=" ", flush=True)
        try:
            result = orchestrator.decompose(
                input_uri=str(img_file),
                detail_level=args.detail,
                text_granularity=args.granularity,
                enable_ocr=True,
                open_in_browser=False,
                engine=args.engine,
                device_preference=args.device,
            )
            # Copy manifest to runs dir
            manifest_path = result.project_paths.manifest_path
            shutil.copy(manifest_path, out_dir / f"{name}.json")
            print(f"OK ({result.manifest['stats']['totalLayers']} layers)")
        except Exception as exc:
            print(f"FAIL: {exc}")

    print(f"\nManifests written to {out_dir}")
    print(f"Next: python eval/score.py {out_dir}")


if __name__ == "__main__":
    main()
