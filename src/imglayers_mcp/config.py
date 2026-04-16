"""Runtime configuration (spec §12)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EngineConfig:
    decomposition: str = "layerd"
    retry_segmentation: str | None = None  # "grounded-sam" when available
    vision_review: str = "disabled"  # "disabled" | "optional"


@dataclass(frozen=True)
class OCRConfig:
    backend: str = "paddleocr"
    lang: str = "japan"
    orientation_correction: str = "auto"  # "auto" | "on" | "off"
    unwarping: str = "auto"


@dataclass(frozen=True)
class TextReconstructionConfig:
    enabled: bool = True
    font_mode: str = "known-fonts"  # "known-fonts" | "deepfont-like" | "heuristic"
    known_fonts: tuple[str, ...] = (
        "Noto Sans JP",
        "Hiragino Sans",
        "Helvetica",
        "Inter",
        "Roboto",
        "Arial",
    )
    rerender_fit: bool = True
    top_k_candidates: int = 5


@dataclass(frozen=True)
class Thresholds:
    text_promote_confidence: float = 0.75
    low_confidence_layer: float = 0.55
    retry_preview_diff: float = 0.12
    retry_edge_leakage: float = 0.18


@dataclass(frozen=True)
class OutputConfig:
    export_svg: bool = True
    export_psd: bool = True
    save_debug_assets: bool = True


@dataclass(frozen=True)
class Config:
    project_root: Path
    default_max_side: int = 2048
    engine: EngineConfig = field(default_factory=EngineConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    text_reconstruction: TextReconstructionConfig = field(default_factory=TextReconstructionConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_env(cls) -> "Config":
        root = Path(os.environ.get("IMGLAYERS_PROJECT_ROOT", "./projects")).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=root,
            default_max_side=int(os.environ.get("IMGLAYERS_MAX_SIDE", "2048")),
        )
