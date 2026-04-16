from __future__ import annotations

from typing import Any

from ..core.preview_renderer import render_preview as _render
from ..models.requests import RenderPreviewRequest
from ..storage.projects import ProjectStore


def render_preview(store: ProjectStore, raw: dict[str, Any]) -> dict[str, Any]:
    req = RenderPreviewRequest.model_validate(raw)
    paths = store.get(req.project_id)
    manifest = store.load_manifest(req.project_id)
    out_path = _render(manifest, paths, output_name=req.output_name)
    rel = out_path.relative_to(paths.dir)
    return {"preview_uri": f"project://{req.project_id}/{rel.as_posix()}"}
