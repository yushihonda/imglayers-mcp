from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from ..core.annotated_preview import render_annotated_preview, render_layer_grid
from ..storage.projects import ProjectStore


class _AnnotatedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_id: str
    style: Literal["overlay", "grid"] = "overlay"
    output_name: Optional[str] = None
    show_labels: bool = True
    include_background: bool = False
    only_ids: Optional[list[str]] = None
    stroke_width: int = 3
    columns: int = 3


def render_annotated_preview_tool(store: ProjectStore, raw: dict[str, Any]) -> dict[str, Any]:
    req = _AnnotatedRequest.model_validate(raw)
    paths = store.get(req.project_id)
    manifest = store.load_manifest(req.project_id)

    if req.style == "overlay":
        out = render_annotated_preview(
            manifest,
            paths,
            output_name=req.output_name or "preview_annotated.png",
            show_labels=req.show_labels,
            include_background=req.include_background,
            only_ids=req.only_ids,
            stroke_width=req.stroke_width,
        )
    else:
        out = render_layer_grid(
            manifest,
            paths,
            output_name=req.output_name or "preview_grid.png",
            columns=req.columns,
            include_background=req.include_background,
        )

    rel = out.relative_to(paths.dir).as_posix()
    return {
        "preview_uri": f"project://{req.project_id}/{rel}",
        "style": req.style,
        "path": rel,
    }
