"""apply_vision_analysis — decompose an image using a pre-computed element list.

The caller (a vision-capable LLM such as Claude Code) analyzes the image and
supplies a structured list of elements with bboxes. This tool cuts pixel data
from each bbox and builds the manifest + preview + exports.
"""

from __future__ import annotations

from typing import Any

from ..core.orchestrator import Orchestrator
from ..core.types import VisionElement
from ..utils.bbox import Box


def apply_vision_analysis(orchestrator: Orchestrator, raw: dict[str, Any]) -> dict[str, Any]:
    input_uri = raw["input_uri"]
    elements = _parse_elements(raw["elements"])
    text_granularity = raw.get("text_granularity", "line")
    export_formats = raw.get("export_formats", ["manifest"])
    open_in_browser = raw.get("open_in_browser", True)

    result = orchestrator.decompose_from_vision(
        input_uri=input_uri,
        elements=elements,
        text_granularity=text_granularity,
        export_formats=list(export_formats),
        open_in_browser=open_in_browser,
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


def _parse_elements(raw_elements: list[dict]) -> list[VisionElement]:
    out: list[VisionElement] = []
    for el in raw_elements:
        bb = el.get("bbox", {})
        out.append(VisionElement(
            type=el.get("type", "unknown"),
            bbox=Box(
                x=float(bb.get("x", 0)),
                y=float(bb.get("y", 0)),
                w=float(bb.get("width", bb.get("w", 0))),
                h=float(bb.get("height", bb.get("h", 0))),
            ),
            name=el.get("name", ""),
            color=el.get("color"),
            text_content=el.get("text_content") or el.get("text"),
            font_size=el.get("font_size"),
            font_weight=el.get("font_weight"),
            children=el.get("children", []),
        ))
    return out
