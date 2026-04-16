from __future__ import annotations

from typing import Any

from ..models.requests import ListLayersRequest
from ..storage.projects import ProjectStore


def list_layers(store: ProjectStore, raw: dict[str, Any]) -> dict[str, Any]:
    req = ListLayersRequest.model_validate(raw)
    manifest = store.load_manifest(req.project_id)
    layers = manifest.get("layers", [])
    f = req.filter
    if f is not None:
        if f.type is not None:
            layers = [l for l in layers if l.get("type") == f.type]
        if f.semantic_role is not None:
            layers = [l for l in layers if l.get("semanticRole") == f.semantic_role]
        if f.editable_only:
            layers = [l for l in layers if l.get("editable", True)]

    summaries = [
        {
            "id": l["id"],
            "name": l["name"],
            "type": l["type"],
            "semantic_role": l.get("semanticRole"),
            "bbox": l["bbox"],
            "z_index": l.get("zIndex", 0),
            "editable": l.get("editable", True),
        }
        for l in layers
    ]
    return {"layers": summaries}
