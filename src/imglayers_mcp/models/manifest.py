"""Pydantic models mirroring the manifest.json schema (spec §8)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LayerType = Literal["text", "image", "vector_like", "group", "unknown"]
SemanticRole = Literal[
    "background",
    "headline",
    "subheadline",
    "body_text",
    "button",
    "card",
    "icon",
    "logo",
    "product_image",
    "illustration",
    "decoration",
    "unknown",
]
EngineRequested = Literal["layerd", "vision-external"]
EngineSelected = Literal["layerd", "vision-external"]
DetailLevel = Literal["fast", "balanced", "high"]
Severity = Literal["info", "warn", "error"]


class Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class BBox(Base):
    x: float
    y: float
    width: float
    height: float


class SourceInfo(Base):
    input_uri: str = Field(alias="inputUri")
    original_file_name: str = Field(alias="originalFileName")
    mime_type: str = Field(alias="mimeType")
    sha256: Optional[str] = None
    width: int
    height: int


class CanvasInfo(Base):
    width: int
    height: int
    background: Optional[str] = None


class PipelineInfo(Base):
    engine_requested: EngineRequested = Field(alias="engineRequested")
    engine_selected: EngineSelected = Field(alias="engineSelected")
    enable_ocr: bool = Field(alias="enableOCR")
    detail_level: DetailLevel = Field(alias="detailLevel")
    timings_ms: Optional[dict[str, float]] = Field(default=None, alias="timingsMs")
    engine_candidates: Optional[list[str]] = Field(default=None, alias="engineCandidates")


class StatsInfo(Base):
    total_layers: int = Field(alias="totalLayers")
    text_layers: int = Field(alias="textLayers")
    image_layers: int = Field(alias="imageLayers")
    vector_like_layers: int = Field(alias="vectorLikeLayers")
    unknown_layers: int = Field(alias="unknownLayers")


class AssetIndex(Base):
    original: str
    preview: str
    layers_dir: str = Field(alias="layersDir")


class LayerAsset(Base):
    path: str
    format: Literal["png"] = "png"
    has_alpha: bool = Field(alias="hasAlpha")


class TextLine(Base):
    text: str
    bbox: BBox


class TextPayload(Base):
    content: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    lines: Optional[list[TextLine]] = None


class GradientStop(Base):
    offset: float
    color: str


class Fill(Base):
    type: Literal["solid", "linear-gradient", "radial-gradient", "image"]
    color: Optional[str] = None
    angle: Optional[float] = None
    stops: Optional[list[GradientStop]] = None


class StyleHints(Base):
    font_size: Optional[float] = Field(default=None, alias="fontSize")
    font_weight: Optional[int] = Field(default=None, alias="fontWeight")
    font_family_guess: Optional[str] = Field(default=None, alias="fontFamilyGuess")
    font_style: Optional[Literal["normal", "italic"]] = Field(default=None, alias="fontStyle")
    text_color: Optional[str] = Field(default=None, alias="textColor")
    text_align: Optional[Literal["left", "center", "right"]] = Field(default=None, alias="textAlign")
    line_height: Optional[float] = Field(default=None, alias="lineHeight")
    letter_spacing: Optional[float] = Field(default=None, alias="letterSpacing")
    dominant_colors: Optional[list[str]] = Field(default=None, alias="dominantColors")
    border_radius_guess: Optional[float] = Field(default=None, alias="borderRadiusGuess")
    fill: Optional[Fill] = None


class Provenance(Base):
    engines: list[str]
    confidence: float
    notes: Optional[list[str]] = None


class CodegenHints(Base):
    component_candidate: Optional[str] = Field(default=None, alias="componentCandidate")
    container_likely: Optional[bool] = Field(default=None, alias="containerLikely")
    text_role: Optional[Literal["title", "paragraph", "button-label", "caption"]] = Field(
        default=None, alias="textRole"
    )


class LayerNode(Base):
    id: str
    name: str
    type: LayerType
    semantic_role: SemanticRole = Field(alias="semanticRole")
    bbox: BBox
    z_index: int = Field(alias="zIndex")
    visible: bool = True
    locked: bool = False
    editable: bool = True
    opacity: float = 1.0
    blend_mode: Optional[str] = Field(default=None, alias="blendMode")
    asset: Optional[LayerAsset] = None
    text: Optional[TextPayload] = None
    style_hints: Optional[StyleHints] = Field(default=None, alias="styleHints")
    provenance: Provenance
    codegen_hints: Optional[CodegenHints] = Field(default=None, alias="codegenHints")
    children: Optional[list[str]] = None


class LayerGroup(Base):
    id: str
    name: str
    layer_ids: list[str] = Field(alias="layerIds")


class WarningItem(Base):
    code: str
    message: str
    severity: Severity = "warn"


class ExportIndex(Base):
    manifest: Optional[str] = None
    svg: Optional[str] = None
    psd: Optional[str] = None
    codegen_plan: Optional[str] = Field(default=None, alias="codegenPlan")


class Manifest(Base):
    version: str = "0.1.0"
    project_id: str = Field(alias="projectId")
    created_at: str = Field(alias="createdAt")
    source: SourceInfo
    canvas: CanvasInfo
    pipeline: PipelineInfo
    stats: StatsInfo
    assets: AssetIndex
    layers: list[LayerNode]
    groups: list[LayerGroup] = Field(default_factory=list)
    warnings: list[WarningItem] = Field(default_factory=list)
    exports: ExportIndex = Field(default_factory=ExportIndex)

    def to_json_dict(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)
