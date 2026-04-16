"""decompose_image — CV+OCR pipeline. LayerD for structure, PaddleOCR for text."""

from __future__ import annotations

from typing import Any

from ..core.orchestrator import Orchestrator


def decompose_image(orchestrator: Orchestrator, raw: dict[str, Any]) -> dict[str, Any]:
    result = orchestrator.decompose(
        input_uri=raw["input_uri"],
        detail_level=raw.get("detail_level", "balanced"),
        max_side=raw.get("max_side", 2048),
        text_granularity=raw.get("text_granularity", "line"),
        enable_ocr=raw.get("enable_ocr", True),
        export_formats=list(raw.get("export_formats", ["manifest"])),
        open_in_browser=raw.get("open_in_browser", True),
    )
    m = result.manifest
    stats = m["stats"]
    pid = m["projectId"]

    def _uri(path) -> str | None:
        if path is None:
            return None
        rel = path.relative_to(result.project_paths.dir).as_posix()
        return f"project://{pid}/{rel}"

    return {
        "project_id": pid,
        "manifest_uri": f"project://{pid}/manifest",
        "preview_uri": f"project://{pid}/preview",
        "annotated_preview_uri": _uri(result.annotated_preview_path),
        "grid_preview_uri": _uri(result.grid_preview_path),
        "viewer_uri": _uri(result.viewer_html_path),
        "viewer_path": (
            result.viewer_html_path.resolve().as_uri()
            if result.viewer_html_path is not None else None
        ),
        "stats": {
            "total_layers": stats["totalLayers"],
            "text_layers": stats["textLayers"],
            "image_layers": stats["imageLayers"],
            "vector_like_layers": stats["vectorLikeLayers"],
        },
        "warnings": m.get("warnings", []),
    }
