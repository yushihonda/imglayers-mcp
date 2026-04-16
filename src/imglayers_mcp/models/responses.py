"""Response payload models for MCP tool outputs."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


class DecomposeStats(_Base):
    total_layers: int
    text_layers: int
    image_layers: int
    vector_like_layers: int


class DecomposeImageResponse(_Base):
    project_id: str
    engine_selected: str
    manifest_uri: str
    preview_uri: str
    annotated_preview_uri: Optional[str] = None
    grid_preview_uri: Optional[str] = None
    viewer_uri: Optional[str] = None
    viewer_path: Optional[str] = None
    stats: DecomposeStats
    warnings: list[dict[str, Any]] = []


class LayerSummary(_Base):
    id: str
    name: str
    type: str
    semantic_role: str
    bbox: dict[str, float]
    z_index: int
    editable: bool


class ListLayersResponse(_Base):
    layers: list[LayerSummary]


class GetLayerResponse(_Base):
    layer: dict[str, Any]


class GetManifestResponse(_Base):
    manifest: dict[str, Any]


class RenderPreviewResponse(_Base):
    preview_uri: str


class ExportProjectResponse(_Base):
    exports: dict[str, Optional[str]]


class DeriveCodegenPlanResponse(_Base):
    plan: dict[str, Any]
