# imglayers-mcp

MCP server that decomposes **design images** (banners, LP heroes, UI mocks, posters, illustrations) into editable layers and a structured `manifest.json`.

The **source of truth** is `manifest.json`. Layer assets are PNG files referenced from the manifest. HTML / React / SVG / PSD are derived outputs.

> **Scope (v0.1):** optimized for graphic designs, illustrations, and stylized visuals. Photo-realistic imagery is out of scope.

## Install

```bash
pip install -e .
# Optional engines:
pip install -e '.[ocr]'     # PaddleOCR for text promotion
pip install -e '.[layerd]'  # LayerD extras
pip install -e '.[qwen]'    # Qwen-Image-Layered (GPU recommended)
```

## Run

```bash
imglayers-mcp
```

Starts an MCP stdio server exposing the tools:

- `decompose_image`
- `apply_vision_analysis`
- `enrich_with_vision`
- `get_manifest`
- `list_layers`
- `get_layer`
- `render_preview`
- `render_annotated_preview`
- `export_project`
- `derive_codegen_plan`

Projects are written under `$IMGLAYERS_PROJECT_ROOT` (default: `./projects/`).

## Pipeline (spec v0.1)

```
Input image
 ├─ Stage 0  Source-aware routing
 ├─ Stage 1  Image type classification
 │            → ui_mock / banner / poster / illustration / photo_mixed / scan_capture
 ├─ Stage 2  Base decomposition  (LayerD)
 ├─ Stage 3  Text extraction      (PaddleOCR)
 ├─ Stage 4  Text reconstruction  (font candidates + rerender fit)
 ├─ Stage 5  Retry segmentation   (optional Grounded-SAM)
 ├─ Stage 6  Manifest building    (semantic roles + hierarchy)
 └─ Stage 7  Verification         (confidence + retry queue)
```

## Design principles

1. **`manifest.json` is source of truth** — PNGs, SVG, PSD, codegen plans are derived.
2. **Text is reconstructed, not cut out** — OCR gives content + coordinates; a font classifier ranks candidate families; rerender fitting picks the best match.
3. **Vision LLM is optional, post-hoc** — used for semantic role enrichment, not mask generation. The core deterministic pipeline works without it.
4. **Confidence + retry** — every layer gets a confidence score; low-confidence items land in a retry queue consumed by Grounded-SAM when available.

## Project layout (at runtime)

```
projects/proj_<id>/
├─ manifest.json
├─ meta/
│  ├─ original.png
│  └─ image_type.json
├─ layers/*.png
├─ text/               # debug crops for text reconstruction
├─ ocr/ocr.json
├─ preview/
│  ├─ preview.png
│  ├─ preview_annotated.png
│  ├─ preview_grid.png
│  └─ index.html       # Figma-style interactive viewer
├─ debug/
│  └─ verification.json
└─ exports/
   ├─ export.svg
   ├─ export.psd
   └─ codegen-plan.json
```

See [implementation spec v0.1](./docs) for the full architecture.
