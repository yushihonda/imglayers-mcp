from __future__ import annotations

from typing import Any

from ..core.exporter import export_codegen_plan, export_psd, export_svg
from ..models.requests import ExportProjectRequest
from ..storage.projects import ProjectStore


def export_project(store: ProjectStore, raw: dict[str, Any]) -> dict[str, Any]:
    req = ExportProjectRequest.model_validate(raw)
    paths = store.get(req.project_id)
    manifest = store.load_manifest(req.project_id)
    out: dict[str, str | None] = {}

    for fmt in req.formats:
        if fmt == "manifest":
            out["manifest"] = f"project://{req.project_id}/manifest"
        elif fmt == "svg":
            p = export_svg(manifest, paths)
            out["svg"] = f"project://{req.project_id}/{p.relative_to(paths.dir).as_posix()}"
        elif fmt == "psd":
            p = export_psd(manifest, paths)
            out["psd"] = (
                f"project://{req.project_id}/{p.relative_to(paths.dir).as_posix()}" if p else None
            )
        elif fmt == "codegen-plan":
            p = export_codegen_plan(manifest, paths)
            out["codegen_plan"] = f"project://{req.project_id}/{p.relative_to(paths.dir).as_posix()}"
        elif fmt == "preview":
            out["preview"] = f"project://{req.project_id}/preview"
    return {"exports": out}
