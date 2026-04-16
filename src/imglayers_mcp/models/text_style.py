"""Text style & reconstruction models (WS2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class TextStyleV2(_Base):
    font_family: Optional[str] = Field(default=None, alias="fontFamily")
    font_candidates: list[str] = Field(default_factory=list, alias="fontCandidates")
    font_weight: int = Field(default=400, alias="fontWeight")
    font_size: float = Field(default=12.0, alias="fontSize")
    color: Optional[str] = None
    text_align: Optional[str] = Field(default=None, alias="textAlign")
    fit_score_pass1: Optional[float] = Field(default=None, alias="fitScorePass1")
    fit_score_pass2: Optional[float] = Field(default=None, alias="fitScorePass2")
    style_confidence: Optional[float] = Field(default=None, alias="styleConfidence")
    pass2_updated: bool = Field(default=False, alias="pass2Updated")
