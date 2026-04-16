# imglayers-mcp

MCP server that decomposes **design images** (banners, LP heroes, UI mocks, posters, illustrations) into editable layers and a structured `manifest.json`.

The **source of truth** is `manifest.json`. Layer assets are PNG files referenced from the manifest. HTML / React / SVG / PSD are derived outputs.

> **Scope (v0.1):** optimized for graphic designs, illustrations, and stylized visuals. Photo-realistic imagery is out of scope.

## Install

```bash
# 1. Install the package and all required dependencies
#    (torch, torchvision, opencv-python, scikit-image, hydra-core, iopath,
#     paddleocr, sam2 from facebookresearch/sam2)
pip install -e .

# 2. Download SAM2 checkpoints into ./weights/sam2/
#    (override location with SAM2_WEIGHTS_DIR)
imglayers-download-weights                # small + base_plus (~560 MB)
imglayers-download-weights --all          # all four checkpoints (~1.2 GB)
imglayers-download-weights --only tiny    # just the smallest one

# 3. Optional extras
pip install -e '.[export]'         # psd-tools for PSD export
pip install -e '.[grounded_sam2]'  # Grounded-SAM-2 semantic retry
```

### Checkpoint selection

SAM2 picks a checkpoint based on the resolved runtime device:

| Device | Default | Recommended |
|---|---|---|
| CUDA  | `base_plus` | `large` for HQ |
| MPS (Apple Silicon) | `small` | `tiny` when memory is tight |
| CPU   | `tiny`  | debug only |

Override explicitly:
```json
{"engine": "hybrid", "sam2_checkpoint": "base_plus", "device_preference": "cuda"}
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

## Engines

`decompose_image` accepts `engine={layerd,sam2,hybrid}` plus `device_preference={auto,cuda,mps,cpu}` and `sam2_checkpoint={auto,tiny,small,base_plus,large}`.

- **layerd** — CV connected components + PaddleOCR. Fast, offline, best for flat banners / UI mocks / posters.
- **sam2** — SAM2 automatic mask generation + alpha refinement. Best for illustrations, photo-mixed designs, complex boundaries. Requires `sam2` extra + weights.
- **hybrid** — routes per image type (`illustration`/`photo_mixed` → sam2 first, others → layerd first) with cross-engine retry for low-confidence layers.

Cross-engine retry kicks in when the verifier flags a layer with `alpha_edge_mismatch` or `high_residual_p95`, and escalates to SAM2 prompt-mode refinement when available.

## Pipeline

```
Input image
 ├─ Stage 0  Source-aware routing
 ├─ Stage 1  Image type classification
 │            → ui_mock / banner / poster / illustration / photo_mixed / scan_capture
 ├─ Stage 2  Base decomposition  (LayerD or SAM2, per engine/hybrid route)
 ├─ Stage 3  Text extraction      (PaddleOCR)
 ├─ Stage 4  Text reconstruction  (font candidates + rerender fit, 2-pass)
 ├─ Stage 5  Retry segmentation   (cc/edge/morph/SAM2-prompt/Grounded-SAM)
 ├─ Stage 6  Manifest building    (semantic roles + hierarchy + engineUsed)
 └─ Stage 7  Verification         (per-layer residuals + retry queue)
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
