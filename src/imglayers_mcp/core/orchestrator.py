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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..adapters.sam2_adapter import SAM2Adapter
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
from ..models.runtime_device import resolve_device
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
from .engine_selector import preferred_retry_engine, route_initial
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
    engine_selected: str = "layerd"
    engine_requested: str = "layerd"
    device_used: str = "cpu"
    sam2_checkpoint: str | None = None
    cross_engine_retry_used: bool = False
    retry_summary: dict = field(default_factory=dict)
    layer_engine_breakdown: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, config: Config, store: ProjectStore) -> None:
        self.config = config
        self.store = store
        self.layerd = LayerDEngine()
        self.ocr = PaddleOCREngine()
        self.retry_backend = GroundedSAMEngine()
        self._font_fitter = RerenderFitter() if config.text_reconstruction.rerender_fit else None
        self._sam2_cache: dict[tuple[str, str], SAM2Adapter] = {}

    def _get_sam2(self, device_pref: str, checkpoint_req: str) -> tuple[SAM2Adapter, str, str]:
        """Return (adapter, device, checkpoint) honoring the config."""
        device_info = resolve_device(device_pref, checkpoint_req)
        key = (device_info.device, device_info.checkpoint)
        adapter = self._sam2_cache.get(key)
        if adapter is None:
            adapter = SAM2Adapter(
                checkpoint=device_info.checkpoint,
                device=device_info.device,
            )
            self._sam2_cache[key] = adapter
        return adapter, device_info.device, device_info.checkpoint

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
        engine: str = "layerd",
        device_preference: str = "auto",
        sam2_checkpoint: str = "auto",
        allow_cross_engine_retry: bool = True,
    ) -> DecomposeResult:
        timings: dict[str, float] = {}
        warnings: list[WarningItem] = []
        rgba, paths, rgba_full, src_path = self._prepare(
            input_uri, max_side or self.config.default_max_side
        )
        h, w = rgba.shape[:2]

        t = time.perf_counter()
        ocr_boxes_for_classifier: list = []
        it_result: ImageTypeResult = classify(rgba, ocr_boxes_for_classifier)
        timings["image_type_ms"] = (time.perf_counter() - t) * 1000
        self._save_debug(paths, "meta/image_type.json", it_result.to_dict())

        preprocess = self._decide_preprocessing(it_result.image_type)

        sam2, sam2_device, sam2_ckpt = self._get_sam2(device_preference, sam2_checkpoint)
        sam2_available = sam2.available
        decision = route_initial(engine, it_result.image_type, sam2_available=sam2_available)
        initial_engine = decision.initial
        self._save_debug(paths, "meta/engine_decision.json", decision.to_dict())

        # ---- Stage 3: OCR extraction (image-type-aware preprocessing)
        ocr_lines: list[OCRLine] = []
        if enable_ocr and self.ocr.available:
            t = time.perf_counter()
            try:
                ocr_lines = self.ocr.extract(
                    rgba[..., :3],
                    orientation=preprocess.orientation_correction,
                    unwarping=preprocess.unwarping,
                )
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

        ocr_bboxes = [ln.bbox for ln in ocr_lines] if ocr_lines else None
        t = time.perf_counter()
        if initial_engine == "sam2":
            from .sam2_pipeline import run as run_sam2
            sam2_result = run_sam2(rgba, it_result.image_type, sam2=sam2, text_boxes=ocr_bboxes)
            raw_layers = sam2_result.raw_layers
        else:
            raw_layers = self.layerd.decompose(rgba, detail=detail_level, text_boxes=ocr_bboxes)
        timings["decompose_ms"] = (time.perf_counter() - t) * 1000

        text_contents = _match_v2(raw_layers, ocr_lines)

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
            engine_requested=engine,
            engine_initial=initial_engine,
            device_used=sam2_device if initial_engine == "sam2" else "cpu",
            sam2_checkpoint=sam2_ckpt if initial_engine == "sam2" else None,
            sam2_available=sam2_available,
            sam2_adapter=sam2,
            allow_cross_engine_retry=allow_cross_engine_retry,
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

        # Provisional role estimate: rank text layers by height (headline =
        # tallest, etc.) to pass a hint to the style estimator. Final role
        # is still decided by naming.infer_semantic_role later.
        text_items = [(i, raw_layers[i]) for i in text_contents if raw_layers[i].kind == "text"]
        text_items.sort(key=lambda p: -p[1].bbox.h)
        provisional_role: dict[int, str] = {}
        for rank, (i, raw) in enumerate(text_items):
            tc = text_contents.get(i, "")
            if rank == 0:
                provisional_role[i] = "headline"
            elif rank <= 2:
                provisional_role[i] = "subheadline"
            elif tc and len(tc) <= 14 and raw.bbox.w <= w * 0.35 and raw.bbox.h <= h * 0.12:
                provisional_role[i] = "button"
            else:
                provisional_role[i] = "body_text"

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
                    semantic_role=provisional_role.get(i),
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
        engine_requested: str = "layerd",
        engine_initial: str = "layerd",
        device_used: str = "cpu",
        sam2_checkpoint: str | None = None,
        sam2_available: bool = False,
        sam2_adapter: SAM2Adapter | None = None,
        allow_cross_engine_retry: bool = False,
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
        retry_segmentation_label = None
        if sam2_available:
            retry_segmentation_label = "sam2"
        elif self.retry_backend.available:
            retry_segmentation_label = "grounded-sam"
        pipeline = PipelineInfo(
            imageType=image_type,  # type: ignore[arg-type]
            engines=PipelineEngines(
                decomposition=engine_initial if engine_initial in ("layerd", "sam2") else "layerd",  # type: ignore[arg-type]
                ocr=("paddleocr" if self.ocr.available else "disabled"),
                retrySegmentation=retry_segmentation_label,  # type: ignore[arg-type]
                fontClassifier=self.config.text_reconstruction.font_mode,  # type: ignore[arg-type]
                visionReview=self.config.engine.vision_review,  # type: ignore[arg-type]
            ),
            engineRequested=engine_requested,
            engineInitial=engine_initial,
            deviceUsed=device_used,
            sam2Checkpoint=sam2_checkpoint,
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

        # WS2 pass-2: re-fit style with the final semantic role per layer.
        if self.config.text_reconstruction.enabled and text_contents:
            t = time.perf_counter()
            self._refit_styles_pass2(manifest_dict, raw_layers, text_contents, rgba, w, h)
            timings["text_pass2_ms"] = (time.perf_counter() - t) * 1000

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

        # Stage 7: verification (image-type-specific thresholds).
        thresholds = _thresholds_for_image_type(
            image_type, self.config.thresholds
        )
        verification = verify(
            manifest_dict, paths.dir, thresholds=thresholds,
            sam2_available=sam2_available,
        )

        # Stage 5: CV-based retry for low-confidence layers (Grounded-SAM stub
        self._last_cross_engine_used = False
        if verification.retry_queue:
            retried = self._run_cv_retry(
                manifest_dict, verification.retry_queue, rgba, paths,
                sam2_adapter=sam2_adapter if allow_cross_engine_retry else None,
            )
            if retried:
                pass
        self._save_debug(paths, "debug/verification.json", verification.to_dict())
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

        layer_engine_breakdown: dict[str, int] = {}
        for layer in manifest_dict.get("layers", []):
            eu = layer.get("engineUsed") or "unknown"
            layer_engine_breakdown[eu] = layer_engine_breakdown.get(eu, 0) + 1

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
            engine_selected=engine_initial,
            engine_requested=engine_requested,
            device_used=device_used,
            sam2_checkpoint=sam2_checkpoint,
            cross_engine_retry_used=getattr(self, "_last_cross_engine_used", False),
            retry_summary={"queued": len(verification.retry_queue)},
            layer_engine_breakdown=layer_engine_breakdown,
        )

    def _run_cv_retry(
        self,
        manifest: dict,
        retry_queue: list[dict],
        rgba: np.ndarray,
        paths: ProjectPaths,
        *,
        sam2_adapter: SAM2Adapter | None = None,
    ) -> list[str]:
        """Tier-ladder retry (spec WS5).

        Tiers: cc_refine → edge_refine → mask_expand_shrink → grounded_sam.
        Only the top-K hardest items escalate to heavy tiers.
        """
        from . import retry_segmentation
        from ..adapters.grounded_sam_adapter import GroundedSAMAdapter
        from ..utils.image_io import save_png

        gsam = GroundedSAMAdapter()
        improved: list[str] = []
        layer_map = {l["id"]: l for l in manifest.get("layers", [])}
        retry_log: list[dict] = []
        # Sort by priority, cap heavy retries to top-3.
        sorted_queue = sorted(retry_queue, key=lambda x: -float(x.get("priority", 0.0)))
        HARD_TOP_K = 3

        for rank, item in enumerate(sorted_queue):
            lid = item.get("layer_id")
            role = item.get("role", "")
            if role in ("background", "unknown"):
                continue
            layer = layer_map.get(lid)
            if layer is None:
                continue
            # Retry is only useful for non-text layers. Text layers already
            # sit on OCR-provided bboxes; re-segmenting them shrinks the
            # frame around the glyphs and hurts matching.
            if layer.get("type") == "text":
                continue
            bb_dict = layer["bbox"]
            bb = Box(
                x=float(bb_dict["x"]), y=float(bb_dict["y"]),
                w=float(bb_dict["width"]), h=float(bb_dict["height"]),
            )
            # CC-refine is the proven-safe tier and handles >90% of
            # recoverable cases without over-shrinking the bbox. Heavier
            # tiers (edge/morph/grounded-sam) escalate only for the top-K
            # hardest items, and only when the result stays near the
            # original bbox size.
            preferred = item.get("preferredRetryEngine", "same_engine")
            result = retry_segmentation.refine_cc(rgba, bb)
            if (result is None or result.score < 0.7) and rank < HARD_TOP_K:
                result = retry_segmentation.refine(
                    rgba, bb,
                    role=role,
                    gsam_adapter=gsam,
                    sam2_adapter=sam2_adapter,
                    allow_grounded_sam=True,
                    allow_sam2=(preferred == "sam2" and sam2_adapter is not None),
                    tier_cap=3,
                )
                if result is not None and result.backend.startswith("sam2"):
                    self._last_cross_engine_used = True
            if result is not None:
                area_ratio = result.bbox.area / max(1.0, bb.area)
                if area_ratio < 0.6 or area_ratio > 1.5:
                    result = None
            entry = {
                "layer_id": lid,
                "role": role,
                "attempted": True,
                "improved": False,
                "backend": result.backend if result else None,
                "score": result.score if result else None,
                "priority": float(item.get("priority", 0.0)),
            }
            if result is not None and result.score > 0.6:
                h, w = rgba.shape[:2]
                canvas = np.zeros((h, w, 4), dtype=np.uint8)
                canvas[result.mask, :3] = rgba[result.mask, :3]
                canvas[result.mask, 3] = 255
                save_png(canvas, paths.layers_dir / f"{lid}.png")
                layer["bbox"] = {
                    "x": result.bbox.x, "y": result.bbox.y,
                    "width": result.bbox.w, "height": result.bbox.h,
                }
                layer["confidence"] = min(1.0, float(layer.get("confidence", 0.5)) + 0.2)
                layer["retryState"] = {
                    "attempted": True,
                    "backend": result.backend,
                    "reason": item.get("reason"),
                    "improved": True,
                }
                entry["improved"] = True
                improved.append(lid)
            else:
                layer["retryState"] = {
                    "attempted": True,
                    "backend": result.backend if result else "none",
                    "reason": item.get("reason"),
                    "improved": False,
                }
            retry_log.append(entry)

        self._save_debug(paths, "debug/retry_log.json", {"entries": retry_log})
        return improved

    def _refit_styles_pass2(
        self,
        manifest: dict,
        raw_layers: list[RawLayer],
        text_contents: dict[int, str],
        rgba: np.ndarray,
        w: int,
        h: int,
    ) -> None:
        """Re-score font candidates using the final semantic role (WS2 pass-2)."""
        from ..text_reconstruction.font_classifier import FontConfig
        from ..text_reconstruction.style_estimator import (
            ReconstructedTextStyle, refine_with_final_role,
        )

        classifier = self._build_font_classifier()
        cfg = self.config.text_reconstruction
        fc_cfg = FontConfig(known_fonts=list(cfg.known_fonts), top_k=cfg.top_k_candidates)

        for layer in manifest.get("layers", []):
            if layer.get("type") != "text":
                continue
            style = layer.get("style") or {}
            if not style:
                continue
            role = layer.get("semanticRole")
            text = (layer.get("text") or {}).get("content") or ""
            if not text:
                continue
            bb = layer["bbox"]
            x1 = max(0, int(round(bb["x"])))
            y1 = max(0, int(round(bb["y"])))
            x2 = min(w, int(round(bb["x"] + bb["width"])))
            y2 = min(h, int(round(bb["y"] + bb["height"])))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = rgba[y1:y2, x1:x2, :3]
            pass1 = ReconstructedTextStyle(
                font_family=style.get("fontFamily"),
                font_candidates=style.get("fontCandidates") or [],
                font_weight=int(style.get("fontWeight", 400)),
                font_size=float(style.get("fontSize", 12)),
                color=style.get("color"),
                text_align=style.get("textAlign"),
                reconstruction_confidence=float(style.get("reconstructionConfidence", 0.1)),
                fit_score_pass1=float(style.get("reconstructionConfidence", 0.1)),
            )
            refined = refine_with_final_role(
                pass1_style=pass1, crop_rgb=crop, text=text, cfg=fc_cfg,
                classifier=classifier, final_role=role,
            )
            if refined.pass2_updated:
                style["fontFamily"] = refined.font_family
                style["fontWeight"] = refined.font_weight
                style["fontCandidates"] = refined.font_candidates
                style["reconstructionConfidence"] = refined.reconstruction_confidence
            style["fitScorePass1"] = refined.fit_score_pass1
            style["fitScorePass2"] = refined.fit_score_pass2
            style["styleConfidence"] = refined.style_confidence
            style["pass2Updated"] = refined.pass2_updated
            layer["style"] = style

    def _save_debug(self, paths: ProjectPaths, rel: str, data: dict) -> None:
        out = paths.dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _thresholds_for_image_type(image_type: str, base) -> dict:
    """Pick per-image-type verifier thresholds.

    - ui_mock: strict preview diff (regular layouts should reconstruct well)
    - photo_mixed: looser everything (complex backgrounds produce more noise)
    - scan_capture: medium preview tolerance (unwarping may introduce small
      pixel shifts)
    - others: spec defaults.
    """
    low_conf = base.low_confidence_layer
    preview_diff = base.retry_preview_diff
    edge_leak = base.retry_edge_leakage
    if image_type == "ui_mock":
        preview_diff = min(0.08, preview_diff)
    elif image_type == "photo_mixed":
        low_conf = max(0.45, low_conf - 0.1)
        preview_diff = max(0.2, preview_diff)
        edge_leak = max(0.25, edge_leak)
    elif image_type == "scan_capture":
        preview_diff = max(0.15, preview_diff)
    return {
        "low_confidence_layer": low_conf,
        "retry_preview_diff": preview_diff,
        "retry_edge_leakage": edge_leak,
    }


def _match_v2(
    raw_layers: list[RawLayer], ocr_lines: list[OCRLine]
) -> dict[int, str]:
    """Wraps ocr_matching.match() so the orchestrator stays lightweight."""
    from . import ocr_matching

    raw_items = [
        (i, r.bbox, (r.debug or {}).get("role_guess"))
        for i, r in enumerate(raw_layers) if r.kind == "text"
    ]
    ocr_items = [(oi, ln.bbox, ln.text) for oi, ln in enumerate(ocr_lines)]
    containers = [r.bbox for r in raw_layers if r.kind != "text" and r.bbox.area > 0]
    results = ocr_matching.match(raw_items, ocr_items, container_bboxes=containers)
    out: dict[int, str] = {}
    for m in results:
        out[m.raw_index] = m.text
    return out


def _match_ocr_to_raw_text(
    raw_layers: list[RawLayer], ocr_lines: list[OCRLine]
) -> dict[int, str]:
    """Associate OCR lines with raw text layers using a composite score,
    then pick a one-to-one assignment (greedy-best-first, Hungarian-lite).

    The score favors pairs that overlap spatially (IoU), share a vertical
    center (baseline alignment), and have similar widths.
    """
    text_raw_idx: list[int] = [i for i, r in enumerate(raw_layers) if r.kind == "text"]
    if not text_raw_idx or not ocr_lines:
        return {}

    # Score matrix: higher is better.
    scores: list[tuple[float, int, int]] = []  # (score, text_idx, ocr_idx)
    for ri in text_raw_idx:
        rb = raw_layers[ri].bbox
        for oi, ln in enumerate(ocr_lines):
            s = _pair_score(rb, ln.bbox)
            if s > 0:
                scores.append((s, ri, oi))

    # Pick in descending order of score, skipping already-used indices.
    scores.sort(reverse=True)
    used_raw: set[int] = set()
    used_ocr: set[int] = set()
    out: dict[int, str] = {}
    for s, ri, oi in scores:
        if ri in used_raw or oi in used_ocr:
            continue
        if s < 0.15:  # quality floor
            break
        used_raw.add(ri)
        used_ocr.add(oi)
        out[ri] = ocr_lines[oi].text
    return out


def _pair_score(a: Box, b: Box) -> float:
    """Composite score in [0, 1]: IoU * baseline * width-ratio."""
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    iou = inter / union if union > 0 else 0.0
    if iou == 0:
        return 0.0

    # Baseline: how close the vertical centers are (relative to mean height).
    a_cy = a.y + a.h / 2
    b_cy = b.y + b.h / 2
    mean_h = max(1.0, (a.h + b.h) / 2)
    baseline = max(0.0, 1.0 - abs(a_cy - b_cy) / mean_h)

    # Width ratio: penalize very different widths (text block size mismatch).
    if max(a.w, b.w) == 0:
        width_ratio = 0.0
    else:
        width_ratio = min(a.w, b.w) / max(a.w, b.w)

    return iou * 0.5 + baseline * 0.3 + width_ratio * 0.2


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
