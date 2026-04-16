from __future__ import annotations

from typing import Any

from ..core.codegen_planner import build_codegen_plan
from ..models.requests import DeriveCodegenPlanRequest
from ..storage.projects import ProjectStore


def derive_codegen_plan(store: ProjectStore, raw: dict[str, Any]) -> dict[str, Any]:
    req = DeriveCodegenPlanRequest.model_validate(raw)
    manifest = store.load_manifest(req.project_id)
    plan = build_codegen_plan(manifest, target=req.target)
    return {"plan": plan}
