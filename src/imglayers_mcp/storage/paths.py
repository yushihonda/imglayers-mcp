"""Path helpers for per-project storage layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    project_id: str

    @property
    def dir(self) -> Path:
        return self.root / self.project_id

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def meta_dir(self) -> Path:
        return self.dir / "meta"

    @property
    def preview_dir(self) -> Path:
        return self.dir / "preview"

    @property
    def layers_dir(self) -> Path:
        return self.dir / "layers"

    @property
    def ocr_dir(self) -> Path:
        return self.dir / "ocr"

    @property
    def debug_dir(self) -> Path:
        return self.dir / "debug"

    @property
    def exports_dir(self) -> Path:
        return self.dir / "exports"

    @property
    def text_dir(self) -> Path:
        return self.dir / "text"

    @property
    def original_path(self) -> Path:
        return self.meta_dir / "original.png"

    @property
    def preflight_path(self) -> Path:
        return self.meta_dir / "preflight.json"

    @property
    def engine_selection_path(self) -> Path:
        return self.meta_dir / "engine-selection.json"

    @property
    def preview_path(self) -> Path:
        return self.preview_dir / "preview.png"

    @property
    def ocr_raw_path(self) -> Path:
        return self.ocr_dir / "ocr.json"

    def ensure(self) -> None:
        for p in (
            self.dir,
            self.meta_dir,
            self.preview_dir,
            self.layers_dir,
            self.ocr_dir,
            self.debug_dir,
            self.exports_dir,
            self.text_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def layer_path(self, layer_id: str) -> Path:
        return self.layers_dir / f"{layer_id}.png"

    def export_path(self, name: str) -> Path:
        return self.exports_dir / name
