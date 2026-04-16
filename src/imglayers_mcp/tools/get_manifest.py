from __future__ import annotations

from typing import Any

from ..models.requests import GetManifestRequest
from ..storage.projects import ProjectStore


def get_manifest(store: ProjectStore, raw: dict[str, Any]) -> dict[str, Any]:
    req = GetManifestRequest.model_validate(raw)
    manifest = store.load_manifest(req.project_id)
    return {"manifest": manifest}
