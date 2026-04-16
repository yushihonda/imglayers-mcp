"""Request payload models for MCP tool inputs."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ExportFormat = Literal["manifest", "svg", "psd", "codegen-plan", "preview"]
TextGranularity = Literal["char", "line", "block"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class GetManifestRequest(_Base):
    project_id: str


class LayerFilter(_Base):
    type: Optional[Literal["text", "image", "vector_like", "unknown"]] = None
    semantic_role: Optional[str] = None
    editable_only: bool = False


class ListLayersRequest(_Base):
    project_id: str
    filter: Optional[LayerFilter] = None


class GetLayerRequest(_Base):
    project_id: str
    layer_id: str
    include_asset_path: bool = True


class RenderPreviewRequest(_Base):
    project_id: str
    output_name: str = "preview.png"


class ExportProjectRequest(_Base):
    project_id: str
    formats: list[ExportFormat] = Field(default_factory=lambda: ["manifest"])


class DeriveCodegenPlanRequest(_Base):
    project_id: str
    target: Literal["react", "html", "generic"] = "generic"
