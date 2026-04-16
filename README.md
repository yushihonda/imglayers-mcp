# imglayers-mcp

MCP server that decomposes images into editable layers and a structured `manifest.json`.

The **source of truth** is `manifest.json`. Layer assets are PNG files referenced from the manifest. HTML/React/SVG/PSD are derived outputs.

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
- `get_manifest`
- `list_layers`
- `get_layer`
- `render_preview`
- `export_project`
- `derive_codegen_plan`

Projects are written under `$IMGLAYERS_PROJECT_ROOT` (default: `./projects/`).

## Design

See [implementation spec v0.1](./docs) — layered architecture with adapters for LayerD / PaddleOCR / Qwen-Image-Layered, a rule-based engine selector, and a manifest builder that treats OCR-promoted text as first-class layers.
