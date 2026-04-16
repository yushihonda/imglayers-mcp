"""Pydantic models mirroring the manifest.json schema (spec v0.1 §8)."""

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
ImageType = Literal["ui_mock", "banner", "poster", "illustration", "photo_mixed", "scan_capture"]
DetailLevel = Literal["fast", "balanced", "high"]
Severity = Literal["info", "warn", "error"]
RetryBackend = Literal["grounded-sam"]
FontClassifierKind = Literal["known-fonts", "deepfont-like", "heuristic"]
VisionReviewMode = Literal["disabled", "optional"]


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


class PipelineEngines(Base):
    decomposition: Literal["layerd", "sam2", "hybrid"] = "layerd"
    ocr: Literal["paddleocr", "disabled"] = "paddleocr"
    retry_segmentation: Optional[Literal["grounded-sam", "sam2", "cc-refine"]] = Field(default=None, alias="retrySegmentation")
    font_classifier: Optional[FontClassifierKind] = Field(default=None, alias="fontClassifier")
    vision_review: VisionReviewMode = Field(default="disabled", alias="visionReview")


class PipelinePreprocessing(Base):
    orientation_correction: bool = Field(default=False, alias="orientationCorrection")
    unwarping: bool = False


class PipelineInfo(Base):
    image_type: ImageType = Field(alias="imageType")
    engines: PipelineEngines
    detail_level: DetailLevel = Field(alias="detailLevel")
    preprocessing: PipelinePreprocessing
    timings_ms: Optional[dict[str, float]] = Field(default=None, alias="timingsMs")
    engine_requested: Optional[str] = Field(default=None, alias="engineRequested")
    engine_initial: Optional[str] = Field(default=None, alias="engineInitial")
    cross_engine_retry_used: bool = Field(default=False, alias="crossEngineRetryUsed")
    device_used: Optional[str] = Field(default=None, alias="deviceUsed")
    sam2_checkpoint: Optional[str] = Field(default=None, alias="sam2Checkpoint")


class StatsInfo(Base):
    total_layers: int = Field(alias="totalLayers")
    text_layers: int = Field(alias="textLayers")
    image_layers: int = Field(alias="imageLayers")
    vector_like_layers: int = Field(alias="vectorLikeLayers")
    low_confidence_layers: int = Field(default=0, alias="lowConfidenceLayers")


class AssetIndex(Base):
    original: str
    preview: str
    layers_dir: str = Field(alias="layersDir")


class LayerAsset(Base):
    path: str
    format: Literal["png"] = "png"
    has_alpha: bool = Field(alias="hasAlpha")
    original_path: Optional[str] = Field(default=None, alias="originalPath")


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


class StylePayload(Base):
    """Text + region style. Spec §8 `StylePayload`."""
    font_family: Optional[str] = Field(default=None, alias="fontFamily")
    font_candidates: Optional[list[str]] = Field(default=None, alias="fontCandidates")
    font_weight: Optional[int] = Field(default=None, alias="fontWeight")
    font_size: Optional[float] = Field(default=None, alias="fontSize")
    font_style: Optional[Literal["normal", "italic"]] = Field(default=None, alias="fontStyle")
    line_height: Optional[float] = Field(default=None, alias="lineHeight")
    letter_spacing: Optional[float] = Field(default=None, alias="letterSpacing")
    color: Optional[str] = None
    text_align: Optional[Literal["left", "center", "right"]] = Field(default=None, alias="textAlign")
    reconstruction_confidence: Optional[float] = Field(default=None, alias="reconstructionConfidence")
    dominant_colors: Optional[list[str]] = Field(default=None, alias="dominantColors")
    border_radius_guess: Optional[float] = Field(default=None, alias="borderRadiusGuess")
    fill: Optional[Fill] = None


# Legacy alias kept for backwards compatibility.
StyleHints = StylePayload


class Provenance(Base):
    engines: list[str]
    notes: Optional[list[str]] = None


class CodegenHints(Base):
    component_candidate: Optional[str] = Field(default=None, alias="componentCandidate")
    container_likely: Optional[bool] = Field(default=None, alias="containerLikely")
    text_role: Optional[Literal["title", "paragraph", "button-label", "caption"]] = Field(
        default=None, alias="textRole"
    )


class RetryState(Base):
    attempted: bool = False
    backend: Optional[RetryBackend] = None
    reason: Optional[str] = None
    improved: Optional[bool] = None


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
    style: Optional[StylePayload] = None
    style_hints: Optional[StylePayload] = Field(default=None, alias="styleHints")
    provenance: Provenance
    confidence: float = 0.7
    engine_used: Optional[str] = Field(default=None, alias="engineUsed")
    mask_quality: Optional[float] = Field(default=None, alias="maskQuality")
    alpha_edge_quality: Optional[float] = Field(default=None, alias="alphaEdgeQuality")
    preferred_retry_engine: Optional[str] = Field(default=None, alias="preferredRetryEngine")
    failure_signals: Optional[list[str]] = Field(default=None, alias="failureSignals")
    retry_state: Optional[RetryState] = Field(default=None, alias="retryState")
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
    preview: Optional[str] = None
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
