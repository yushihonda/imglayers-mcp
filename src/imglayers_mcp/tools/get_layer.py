from __future__ import annotations

from typing import Any

from ..models.requests import GetLayerRequest
from ..storage.projects import ProjectStore


def get_layer(store: ProjectStore, raw: dict[str, Any]) -> dict[str, Any]:
    req = GetLayerRequest.model_validate(raw)
    manifest = store.load_manifest(req.project_id)
    for layer in manifest.get("layers", []):
        if layer["id"] == req.layer_id:
            out = dict(layer)
            if not req.include_asset_path and out.get("asset") is not None:
                out = {**out, "asset": None}
            return {"layer": out}
    raise LookupError(f"layer {req.layer_id!r} not found in project {req.project_id!r}")
