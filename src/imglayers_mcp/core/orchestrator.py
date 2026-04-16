"""Stage-driven orchestrator (spec §5).

Pipeline stages:
  0. Source-aware routing (metadata vs raster)
  1. Image type classification
  2. Base decomposition (LayerD)
  3. OCR extraction (PaddleOCR)
  4. Text reconstruction (font classifier + rerender fit)
  5. Retry segmentation (optional, Grounded-SAM)
  6. Manifest building
  7. Verification + confidence scoring
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..decomposition import LayerDEngine
from ..models.manifest import (
    CanvasInfo,
    ExportIndex,
    PipelineEngines,
    PipelineInfo,
    PipelinePreprocessing,
    SourceInfo,
    WarningItem,
)
from ..ocr import OCRLine, PaddleOCREngine, cluster_into_lines
from ..retry_segmentation import GroundedSAMEngine
from ..storage.paths import ProjectPaths
from ..storage.projects import ProjectStore
from ..text_reconstruction import (
    FontCandidate,
    HeuristicBackend,
    KnownFontsBackend,
    RerenderFitter,
)
from ..utils.bbox import Box
from ..utils.color import average_color
from ..utils.hash_utils import sha256_file
from ..utils.image_io import (
    guess_mime,
    load_rgba,
    resize_to_max_side,
    resolve_input_uri,
    save_png,
)
from ..utils.logging import get_logger
from .annotated_preview import render_annotated_preview, render_layer_grid
from .exporter import export_codegen_plan, export_psd, export_svg
from .html_viewer import render_viewer_html
from .image_type_classifier import ImageTypeResult, classify
from .manifest_builder import build_manifest
from .merger import merge
from .preview_renderer import render_preview
from .types import RawLayer, VisionElement
from .verifier import verify

log = get_logger(__name__)


@dataclass
class DecomposeResult:
    project_paths: ProjectPaths
    manifest: dict
    warnings: list[WarningItem]
    preview_path: Path
    annotated_preview_path: Path | None = None
    grid_preview_path: Path | None = None
    viewer_html_path: Path | None = None
    image_type: str = "unknown"
    verification_score: float = 0.0


class Orchestrator:
    def __init__(self, config: Config, store: ProjectStore) -> None:
        self.config = config
        self.store = store
        self.layerd = LayerDEngine()
        self.ocr = PaddleOCREngine()
        self.retry_backend = GroundedSAMEngine()
        self._font_fitter = RerenderFitter() if config.text_reconstruction.rerender_fit else None

    # ------------------------------------------------------------------
    # Path A: CV + OCR automatic decomposition (spec §5 main pipeline)
    # ------------------------------------------------------------------
    def decompose(
        self,
        *,
        input_uri: str,
        detail_level: str = "balanced",
        max_side: int | None = None,
        text_granularity: str = "line",
        enable_ocr: bool = True,
        export_formats: list[str] | None = None,
        open_in_browser: bool = False,
    ) -> DecomposeResult:
        timings: dict[str, float] = {}
        warnings: list[WarningItem] = []
        rgba, paths, rgba_full, src_path = self._prepare(
            input_uri, max_side or self.config.default_max_side
        )
        h, w = rgba.shape[:2]

        # ---- Stage 1: image type classification
        t = time.perf_counter()
        ocr_boxes_for_classifier: list = []
        it_result: ImageTypeResult = classify(rgba, ocr_boxes_for_classifier)
        timings["image_type_ms"] = (time.perf_counter() - t) * 1000
        self._save_debug(paths, "meta/image_type.json", it_result.to_dict())

        # ---- Preprocessing decision
        preprocess = self._decide_preprocessing(it_result.image_type)

        # ---- Stage 3: OCR extraction
        ocr_lines: list[OCRLine] = []
        if enable_ocr and self.ocr.available:
            t = time.perf_counter()
            try:
                ocr_lines = self.ocr.extract(rgba[..., :3])
            except Exception as exc:
                warnings.append(WarningItem(code="OCR_FAILED", message=str(exc), severity="warn"))
            timings["ocr_ms"] = (time.perf_counter() - t) * 1000
            # Save raw OCR for debug.
            self._save_debug(paths, "ocr/ocr.json", {
                "lines": [
                    {"text": l.text, "bbox": l.bbox.to_dict(), "confidence": l.confidence}
                    for l in ocr_lines
                ],
            })
        elif enable_ocr and not self.ocr.available:
            warnings.append(WarningItem(
                code="OCR_UNAVAILABLE",
                message="PaddleOCR not installed",
                severity="info",
            ))

        # ---- Stage 2: Base decomposition
        ocr_bboxes = [ln.bbox for ln in ocr_lines] if ocr_lines else None
        t = time.perf_counter()
        raw_layers = self.layerd.decompose(rgba, detail=detail_level, text_boxes=ocr_bboxes)
        timings["decompose_ms"] = (time.perf_counter() - t) * 1000

        # Associate OCR text with corresponding raw text layers (IoU).
        text_contents: dict[int, str] = {}
        for i, raw in enumerate(raw_layers):
            if raw.kind != "text":
                continue
            best = None
            best_iou = 0.0
            for ln in ocr_lines:
                iou = _iou(raw.bbox, ln.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best = ln
            if best and best_iou > 0.3:
                text_contents[i] = best.text

        # ---- Stage 5: optional retry segmentation (flagged but not yet wired)
        # Retry is triggered by the verifier's retry_queue; executed at a later
        # iteration. For v0.1 we record availability but do not invoke.

        # ---- Stage 4: text reconstruction hints (font candidates)
        font_classifier = self._build_font_classifier()
        style_overrides: dict[int, dict] = {}
        if self.config.text_reconstruction.enabled and text_contents:
            t = time.perf_counter()
            style_overrides = self._reconstruct_text_styles(
                raw_layers, text_contents, ocr_lines, rgba, w, h, font_classifier,
            )
            timings["text_reconstruction_ms"] = (time.perf_counter() - t) * 1000

        return self._finalize(
            paths, rgba, rgba_full, src_path, raw_layers, text_contents, style_overrides,
            text_granularity, export_formats or ["manifest"], open_in_browser,
            image_type=it_result.image_type,
            preprocess=preprocess,
            warnings=warnings,
            timings=timings,
        )

    # ------------------------------------------------------------------
    # Path B: external vision-model-driven decomposition
    # ------------------------------------------------------------------
    def decompose_from_vision(
        self,
        *,
        input_uri: str,
        elements: list[VisionElement],
        text_granularity: str = "line",
        max_side: int | None = None,
        export_formats: list[str] | None = None,
        open_in_browser: bool = False,
    ) -> DecomposeResult:
        timings: dict[str, float] = {}
        warnings: list[WarningItem] = []
        rgba, paths, rgba_full, src_path = self._prepare(
            input_uri, max_side or self.config.default_max_side
        )
        h, w = rgba.shape[:2]

        source_w = int(rgba_full.shape[1])
        if source_w != w:
            scale = w / source_w
            elements = _scale_elements(elements, scale)

        raw_layers = _vision_to_raw_layers(elements, rgba, h, w)
        text_contents = {i: el.text_content for i, el in enumerate(elements) if el.text_content}

        it_result = classify(rgba, [])
        self._save_debug(paths, "meta/image_type.json", it_result.to_dict())

        return self._finalize(
            paths, rgba, rgba_full, src_path, raw_layers, text_contents, {},
            text_granularity, export_formats or ["manifest"], open_in_browser,
            image_type=it_result.image_type,
            preprocess=self._decide_preprocessing(it_result.image_type),
            warnings=warnings,
            timings=timings,
            engine_override="vision-external",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare(self, input_uri: str, max_side: int):
        src_path = resolve_input_uri(input_uri)
        if not src_path.exists():
            raise FileNotFoundError(f"input not found: {src_path}")
        rgba_full = load_rgba(src_path)
        rgba, _ = resize_to_max_side(rgba_full, max_side)
        paths = self.store.new_project()
        save_png(rgba_full, paths.original_path)
        return rgba, paths, rgba_full, src_path

    def _decide_preprocessing(self, image_type: str) -> PipelinePreprocessing:
        cfg = self.config.ocr
        if cfg.orientation_correction == "on":
            return PipelinePreprocessing(orientation_correction=True, unwarping=cfg.unwarping == "on")
        if cfg.orientation_correction == "off":
            return PipelinePreprocessing(orientation_correction=False, unwarping=False)
        # auto
        if image_type == "scan_capture":
            return PipelinePreprocessing(orientation_correction=True, unwarping=True)
        if image_type == "photo_mixed":
            return PipelinePreprocessing(orientation_correction=False, unwarping=False)
        return PipelinePreprocessing(orientation_correction=False, unwarping=False)

    def _build_font_classifier(self):
        cfg = self.config.text_reconstruction
        if cfg.font_mode == "known-fonts":
            return KnownFontsBackend(fitter=self._font_fitter)
        return HeuristicBackend()

    def _reconstruct_text_styles(
        self,
        raw_layers: list[RawLayer],
        text_contents: dict[int, str],
        ocr_lines: list[OCRLine],
        rgba: np.ndarray,
        w: int,
        h: int,
        classifier,
    ) -> dict[int, dict]:
        """Run font candidate ranking on each text layer and return style overrides."""
        from .. import text_reconstruction as tr_mod
        from ..text_reconstruction.font_classifier import FontConfig

        out: dict[int, dict] = {}
        cfg = self.config.text_reconstruction
        fc_cfg = FontConfig(
            known_fonts=list(cfg.known_fonts),
            top_k=cfg.top_k_candidates,
        )
        for i, text in text_contents.items():
            raw = raw_layers[i]
            bbox = raw.bbox
            x1 = max(0, int(round(bbox.x)))
            y1 = max(0, int(round(bbox.y)))
            x2 = min(w, int(round(bbox.x + bbox.w)))
            y2 = min(h, int(round(bbox.y + bbox.h)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = rgba[y1:y2, x1:x2, :3]
            try:
                style = tr_mod.estimate_style(
                    crop_rgb=crop,
                    text=text,
                    cfg=fc_cfg,
                    classifier=classifier,
                    canvas_width=w,
                    line_bbox_x=bbox.x,
                    line_bbox_w=bbox.w,
                )
            except Exception:
                continue
            out[i] = {
                "font_family": style.font_family,
                "font_candidates": style.font_candidates,
                "font_weight": style.font_weight,
                "font_size": style.font_size,
                "color": style.color,
                "text_align": style.text_align,
                "reconstruction_confidence": style.reconstruction_confidence,
            }
        return out

    def _finalize(
        self,
        paths: ProjectPaths,
        rgba: np.ndarray,
        rgba_full: np.ndarray,
        src_path: Path,
        raw_layers: list[RawLayer],
        text_contents: dict[int, str],
        style_overrides: dict[int, dict],
        text_granularity: str,
        export_formats: list[str],
        open_in_browser: bool,
        *,
        image_type: str,
        preprocess: PipelinePreprocessing,
        warnings: list[WarningItem],
        timings: dict[str, float],
        engine_override: str | None = None,
    ) -> DecomposeResult:
        h, w = rgba.shape[:2]

        merge_result = merge(
            raw_layers, canvas_w=w, canvas_h=h,
            text_granularity=text_granularity,
            text_contents=text_contents,
        )
        promoted = merge_result.layers
        text_groups = merge_result.text_groups

        bg_hex = _guess_background(rgba)
        source = SourceInfo(
            inputUri=str(src_path),
            originalFileName=src_path.name,
            mimeType=guess_mime(src_path),
            sha256=sha256_file(src_path),
            width=int(rgba_full.shape[1]),
            height=int(rgba_full.shape[0]),
        )
        canvas = CanvasInfo(width=int(w), height=int(h), background=bg_hex)
        pipeline = PipelineInfo(
            imageType=image_type,  # type: ignore[arg-type]
            engines=PipelineEngines(
                decomposition="layerd",
                ocr=("paddleocr" if self.ocr.available else "disabled"),
                retrySegmentation=("grounded-sam" if self.retry_backend.available else None),
                fontClassifier=self.config.text_reconstruction.font_mode,  # type: ignore[arg-type]
                visionReview=self.config.engine.vision_review,  # type: ignore[arg-type]
            ),
            detailLevel="high",
            preprocessing=preprocess,
            timingsMs=timings,
        )

        manifest = build_manifest(
            project_paths=paths,
            source=source,
            canvas=canvas,
            pipeline=pipeline,
            original_rgb=rgba[..., :3],
            promoted_layers=promoted,
            text_groups=text_groups,
            warnings=warnings,
            exports=ExportIndex(manifest="manifest.json"),
            style_overrides=style_overrides,
        )
        manifest_dict = manifest.to_json_dict()

        preview_path = render_preview(manifest_dict, paths)
        annotated_path: Path | None = None
        grid_path: Path | None = None
        try:
            annotated_path = render_annotated_preview(manifest_dict, paths)
        except Exception as exc:
            warnings.append(WarningItem(code="PREVIEW_FAILED", message=f"annotated: {exc}", severity="info"))
        try:
            grid_path = render_layer_grid(manifest_dict, paths)
        except Exception as exc:
            warnings.append(WarningItem(code="PREVIEW_FAILED", message=f"grid: {exc}", severity="info"))

        exports_rel: dict[str, str] = {"manifest": "manifest.json", "preview": "preview/preview.png"}
        if "svg" in export_formats:
            try:
                svg_path = export_svg(manifest_dict, paths)
                exports_rel["svg"] = str(svg_path.relative_to(paths.dir))
            except Exception as exc:
                warnings.append(WarningItem(code="EXPORT_FAILED", message=f"svg: {exc}", severity="warn"))
        if "psd" in export_formats:
            try:
                psd_path = export_psd(manifest_dict, paths)
                if psd_path is not None:
                    exports_rel["psd"] = str(psd_path.relative_to(paths.dir))
            except Exception as exc:
                warnings.append(WarningItem(code="EXPORT_FAILED", message=f"psd: {exc}", severity="warn"))
        if "codegen-plan" in export_formats:
            try:
                plan_path = export_codegen_plan(manifest_dict, paths)
                exports_rel["codegenPlan"] = str(plan_path.relative_to(paths.dir))
            except Exception as exc:
                warnings.append(WarningItem(code="EXPORT_FAILED", message=f"codegen: {exc}", severity="warn"))

        manifest_dict["warnings"] = [w.model_dump(by_alias=True) for w in warnings]
        manifest_dict["exports"] = exports_rel

        # Stage 7: verification.
        verification = verify(
            manifest_dict, paths.dir,
            thresholds={
                "low_confidence_layer": self.config.thresholds.low_confidence_layer,
                "retry_preview_diff": self.config.thresholds.retry_preview_diff,
            },
        )
        self._save_debug(paths, "debug/verification.json", {
            "overall_score": verification.overall_score,
            "preview_diff": verification.preview_diff,
            "low_confidence_layers": verification.low_confidence_layers,
            "retry_queue": verification.retry_queue,
            "notes": verification.notes,
        })
        # Update stats with low-confidence count.
        if "stats" in manifest_dict:
            manifest_dict["stats"]["lowConfidenceLayers"] = len(verification.low_confidence_layers)

        self.store.write_manifest(paths, manifest_dict)

        viewer_path: Path | None = None
        try:
            viewer_path = render_viewer_html(manifest_dict, paths)
        except Exception as exc:
            warnings.append(WarningItem(code="VIEWER_FAILED", message=str(exc), severity="info"))

        if open_in_browser and viewer_path is not None:
            try:
                import webbrowser
                webbrowser.open(viewer_path.resolve().as_uri())
            except Exception as exc:
                warnings.append(WarningItem(code="VIEWER_OPEN_FAILED", message=str(exc), severity="info"))

        return DecomposeResult(
            project_paths=paths,
            manifest=manifest_dict,
            warnings=warnings,
            preview_path=preview_path,
            annotated_preview_path=annotated_path,
            grid_preview_path=grid_path,
            viewer_html_path=viewer_path,
            image_type=image_type,
            verification_score=verification.overall_score,
        )

    def _save_debug(self, paths: ProjectPaths, rel: str, data: dict) -> None:
        out = paths.dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _iou(a: Box, b: Box) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _scale_elements(elements: list[VisionElement], scale: float) -> list[VisionElement]:
    if scale == 1.0:
        return elements
    out: list[VisionElement] = []
    for el in elements:
        b = el.bbox
        out.append(VisionElement(
            type=el.type,
            bbox=Box(x=b.x * scale, y=b.y * scale, w=b.w * scale, h=b.h * scale),
            name=el.name, color=el.color,
            text_content=el.text_content,
            font_size=(el.font_size * scale) if el.font_size else None,
            font_weight=el.font_weight,
            children=el.children,
        ))
    return out


def _vision_to_raw_layers(
    elements: list[VisionElement], rgba: np.ndarray, h: int, w: int
) -> list[RawLayer]:
    rgb = rgba[..., :3]
    layers: list[RawLayer] = []
    border_w = max(1, min(h, w) // 20)
    border = np.concatenate([
        rgb[:border_w].reshape(-1, 3), rgb[-border_w:].reshape(-1, 3),
        rgb[:, :border_w].reshape(-1, 3), rgb[:, -border_w:].reshape(-1, 3),
    ], axis=0).astype(np.int32)
    bg_color = border.mean(axis=0).round().astype(np.uint8)

    for idx, el in enumerate(elements):
        x1 = max(0, int(round(el.bbox.x)))
        y1 = max(0, int(round(el.bbox.y)))
        x2 = min(w, int(round(el.bbox.x + el.bbox.w)))
        y2 = min(h, int(round(el.bbox.y + el.bbox.h)))
        if x2 <= x1 or y2 <= y1:
            continue

        if el.type == "background":
            canvas = np.concatenate(
                [rgb.astype(np.uint8), np.full((h, w, 1), 255, dtype=np.uint8)], axis=2
            )
            layers.append(RawLayer(
                rgba=canvas, bbox=Box(0, 0, w, h), kind="image",
                engine="vision", confidence=0.95, z_index=0,
                debug={"role_guess": "background", "color": el.color},
            ))
            continue

        region = rgb[y1:y2, x1:x2]
        diff = np.linalg.norm(region.astype(np.int32) - bg_color.astype(np.int32), axis=2)
        fg = diff > 15
        alpha = (fg.astype(np.uint8) * 255)[..., None]
        patch = np.concatenate(
            [np.where(fg[..., None], region, 0).astype(np.uint8), alpha], axis=2
        )
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = patch

        kind_map = {
            "text": "text", "icon": "unknown", "button": "unknown",
            "shape": "image", "card": "image", "badge": "image",
            "image": "image", "decoration": "vector_like",
        }
        kind = kind_map.get(el.type, "unknown")

        layers.append(RawLayer(
            rgba=canvas,
            bbox=Box(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            kind=kind, engine="vision", confidence=0.9, z_index=idx,
            debug={"vision_type": el.type, "vision_name": el.name, "color": el.color},
        ))

    return layers


def _guess_background(rgba: np.ndarray) -> str | None:
    rgb = rgba[..., :3]
    h, w = rgb.shape[:2]
    bw = max(1, min(h, w) // 20)
    strip = np.concatenate([
        rgb[:bw].reshape(-1, 3), rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3), rgb[:, -bw:].reshape(-1, 3),
    ], axis=0)
    return average_color(strip)
