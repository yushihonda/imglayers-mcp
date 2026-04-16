"""Run decompose_image + enrich_with_vision (semantic labels from Claude).

Reads pre-written enrichment JSON files from eval/enrichments/<name>.json
and applies them after decompose_image.

Each enrichment file has the shape:
{
  "layer_updates": [ {"layer_id": "headline_001", "semantic_role": "headline"}, ... ],
  "groups": [...]
}
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
ENRICH_DIR = ROOT / "enrichments"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="Run tag")
    ap.add_argument("--open-in-browser", action="store_true", help="Open index.html for each case after enriching")
    ap.add_argument("--open-gallery", action="store_true", help="Generate and open a gallery page")
    args = ap.parse_args()

    out_dir = RUNS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    from imglayers_mcp.server import build_runtime
    from imglayers_mcp.tools.enrich_with_vision import enrich_with_vision
    from imglayers_mcp.core.html_viewer import render_viewer_html

    tools, router, orchestrator, store = build_runtime()
    viewers: list[tuple[str, Path]] = []

    for img_file in sorted(IMG_DIR.glob("*.png")):
        name = img_file.stem
        print(f"  {name}...", end=" ", flush=True)
        try:
            # Step 1: CV+OCR decomposition
            result = orchestrator.decompose(
                input_uri=str(img_file),
                detail_level="high",
                text_granularity="line",
                enable_ocr=True,
                open_in_browser=False,
            )
            pid = result.manifest["projectId"]

            # Step 2: enrich with pre-authored semantic labels
            enrich_file = ENRICH_DIR / f"{name}.json"
            if enrich_file.exists():
                enrich_data = json.loads(enrich_file.read_text())
                enrich_with_vision(store, {
                    "project_id": pid,
                    "layer_updates": enrich_data.get("layer_updates", []),
                    "groups": enrich_data.get("groups", []),
                })
                # Re-render viewer with enriched manifest.
                manifest = json.loads(result.project_paths.manifest_path.read_text())
                render_viewer_html(manifest, result.project_paths)

            # Step 3: copy final manifest
            manifest_path = result.project_paths.manifest_path
            shutil.copy(manifest_path, out_dir / f"{name}.json")

            viewer_path = result.project_paths.preview_dir / "index.html"
            if viewer_path.exists():
                viewers.append((name, viewer_path))
                if args.open_in_browser:
                    import webbrowser
                    webbrowser.open(viewer_path.resolve().as_uri())

            print("OK")
        except Exception as exc:
            print(f"FAIL: {exc}")

    print(f"\nManifests written to {out_dir}")

    if args.open_gallery:
        gallery = build_gallery(out_dir, viewers)
        print(f"Gallery: {gallery}")
        import webbrowser
        webbrowser.open(gallery.as_uri())


def build_gallery(out_dir: Path, viewers: list[tuple[str, Path]]) -> Path:
    """Emit a gallery HTML that links to every per-case viewer."""
    items = []
    for name, path in viewers:
        # Load scoring info if available.
        rel_viewer = path.resolve().as_uri()
        rel_image = (Path(__file__).parent / "images" / f"{name}.png").resolve().as_uri()
        items.append(
            f'<div class="case">'
            f'<h3>{name}</h3>'
            f'<a href="{rel_viewer}" target="_blank"><img src="{rel_image}" alt="{name}"/></a>'
            f'<p><a href="{rel_viewer}" target="_blank">Open viewer →</a></p>'
            f'</div>'
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>imglayers eval gallery</title>
<style>
  body {{ margin: 0; background: #1e1e1e; color: #ddd; font: 14px -apple-system,BlinkMacSystemFont,sans-serif; }}
  header {{ padding: 16px 24px; border-bottom: 1px solid #333; }}
  h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; padding: 24px; }}
  .case {{ background: #252526; border: 1px solid #333; border-radius: 6px; padding: 12px; }}
  .case h3 {{ font-size: 13px; margin: 0 0 8px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .case img {{ width: 100%; max-height: 200px; object-fit: contain; background: #111; border-radius: 4px; display: block; }}
  .case a {{ color: #0d99ff; text-decoration: none; font-size: 12px; }}
  .case a:hover {{ text-decoration: underline; }}
  .case p {{ margin: 8px 0 0; }}
</style>
</head>
<body>
<header><h1>imglayers eval gallery — {out_dir.name}</h1></header>
<div class="grid">
{"".join(items)}
</div>
</body>
</html>
"""
    gallery_path = out_dir / "gallery.html"
    gallery_path.write_text(html, encoding="utf-8")
    return gallery_path


if __name__ == "__main__":
    main()
