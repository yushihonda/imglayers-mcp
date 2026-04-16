"""Verification result models (WS1)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class LayerVerification(_Base):
    layer_id: str = Field(alias="layerId")
    local_residual_mean: float = Field(alias="localResidualMean")
    local_residual_p95: float = Field(alias="localResidualP95")
    alpha_edge_mismatch: float = Field(alias="alphaEdgeMismatch")
    bbox_border_diff: float = Field(alias="bboxBorderDiff")
    ocr_reread_consistency: Optional[float] = Field(default=None, alias="ocrRereadConsistency")
    retry_priority: float = Field(alias="retryPriority")
    reasons: list[str] = Field(default_factory=list)


class VerificationReport(_Base):
    overall_score: float = Field(alias="overallScore")
    preview_diff: float = Field(alias="previewDiff")
    per_layer: list[LayerVerification] = Field(default_factory=list, alias="perLayer")
    low_confidence_layers: list[str] = Field(default_factory=list, alias="lowConfidenceLayers")
    retry_queue: list[dict] = Field(default_factory=list, alias="retryQueue")
    thresholds: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
