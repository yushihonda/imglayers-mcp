"""SAM2 adapter.

Loads a SAM2 checkpoint on the selected device and produces a list of
``MaskCandidate`` objects via automatic mask generation. If the `sam2`
package or its weights are missing, the adapter reports unavailability
and the orchestrator falls back to LayerD.

Checkpoint selection:
  tiny / small / base_plus / large
  When "auto" is given, the caller should already have resolved it via
  ``models/runtime_device.resolve_device``.

Device selection:
  cuda / mps / cpu
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..models.mask_candidate import MaskCandidate
from ..utils.bbox import Box
from ..utils.logging import get_logger

log = get_logger(__name__)


_CHECKPOINT_FILES = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}
_CONFIG_FILES = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


@dataclass
class SAM2GeneratorConfig:
    points_per_side: int = 32
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.85
    box_nms_thresh: float = 0.7
    min_mask_region_area: int = 64


class SAM2Adapter:
    def __init__(
        self,
        checkpoint: str = "small",
        device: str = "cpu",
        weights_dir: str | Path | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._device = device
        self._weights_dir = Path(
            weights_dir or os.environ.get("SAM2_WEIGHTS_DIR", "./weights/sam2")
        )
        self._predictor: Any = None
        self._generator: Any = None
        self._available = False
        self._init_error: str | None = None

    def _ensure(self) -> None:
        if self._available or self._init_error is not None:
            return
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except Exception as exc:
            self._init_error = f"sam2_not_installed: {exc}"
            log.info("SAM2 adapter unavailable: %s", exc)
            return

        ckpt_name = _CHECKPOINT_FILES.get(self._checkpoint)
        cfg_name = _CONFIG_FILES.get(self._checkpoint)
        if ckpt_name is None or cfg_name is None:
            self._init_error = f"invalid_checkpoint: {self._checkpoint}"
            return
        ckpt_path = self._weights_dir / ckpt_name
        if not ckpt_path.exists():
            self._init_error = f"checkpoint_missing: {ckpt_path}"
            log.info("SAM2 checkpoint not found: %s", ckpt_path)
            return

        try:
            torch_device = torch.device(self._device)
            sam2_model = build_sam2(cfg_name, str(ckpt_path), device=torch_device)
            gen_cfg = SAM2GeneratorConfig()
            self._generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=gen_cfg.points_per_side,
                pred_iou_thresh=gen_cfg.pred_iou_thresh,
                stability_score_thresh=gen_cfg.stability_score_thresh,
                box_nms_thresh=gen_cfg.box_nms_thresh,
                min_mask_region_area=gen_cfg.min_mask_region_area,
            )
            self._available = True
            log.info(
                "SAM2 ready (checkpoint=%s, device=%s)",
                self._checkpoint, self._device,
            )
        except Exception as exc:
            self._init_error = f"sam2_init_failed: {exc}"
            log.warning("SAM2 initialization failed: %s", exc)

    @property
    def available(self) -> bool:
        self._ensure()
        return self._available

    @property
    def init_error(self) -> str | None:
        self._ensure()
        return self._init_error

    def generate_masks(self, rgb: np.ndarray) -> list[MaskCandidate]:
        self._ensure()
        if not self._available:
            return []
        try:
            raw = self._generator.generate(rgb)
        except Exception as exc:
            log.warning("SAM2 generate failed: %s", exc)
            return []
        out: list[MaskCandidate] = []
        for item in raw:
            mask = item.get("segmentation")
            if mask is None:
                continue
            mask_arr = np.asarray(mask, dtype=bool)
            if not mask_arr.any():
                continue
            ys, xs = np.nonzero(mask_arr)
            bbox = Box(
                x=float(xs.min()),
                y=float(ys.min()),
                w=float(xs.max() - xs.min() + 1),
                h=float(ys.max() - ys.min() + 1),
            )
            out.append(MaskCandidate(
                mask=mask_arr,
                bbox=bbox,
                area=int(mask_arr.sum()),
                score=float(item.get("predicted_iou", item.get("stability_score", 0.0))),
                stability=float(item.get("stability_score", 0.0)),
                predicted_iou=float(item.get("predicted_iou", 0.0)),
                source="sam2",
                checkpoint=self._checkpoint,
                device=self._device,
                crop_region=tuple(item["crop_box"]) if "crop_box" in item else None,
            ))
        return out

    def refine_bbox(
        self, rgb: np.ndarray, bbox: Box
    ) -> MaskCandidate | None:
        """Prompt-mode refinement for one bbox (used by cross-engine retry)."""
        self._ensure()
        if not self._available:
            return None
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception:
            return None
        if self._predictor is None:
            try:
                import torch
                from sam2.build_sam import build_sam2
                torch_device = torch.device(self._device)
                sam2_model = build_sam2(
                    _CONFIG_FILES[self._checkpoint],
                    str(self._weights_dir / _CHECKPOINT_FILES[self._checkpoint]),
                    device=torch_device,
                )
                self._predictor = SAM2ImagePredictor(sam2_model)
            except Exception as exc:
                log.warning("SAM2 predictor init failed: %s", exc)
                return None
        try:
            self._predictor.set_image(rgb)
            box = np.array([bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h], dtype=np.float32)
            masks, scores, _ = self._predictor.predict(box=box[None, :], multimask_output=False)
            if masks is None or len(masks) == 0:
                return None
            mask_arr = np.asarray(masks[0], dtype=bool)
            if not mask_arr.any():
                return None
            ys, xs = np.nonzero(mask_arr)
            new_bbox = Box(
                x=float(xs.min()),
                y=float(ys.min()),
                w=float(xs.max() - xs.min() + 1),
                h=float(ys.max() - ys.min() + 1),
            )
            return MaskCandidate(
                mask=mask_arr,
                bbox=new_bbox,
                area=int(mask_arr.sum()),
                score=float(scores[0]) if scores is not None and len(scores) > 0 else 0.8,
                source="sam2",
                checkpoint=self._checkpoint,
                device=self._device,
            )
        except Exception as exc:
            log.warning("SAM2 prompt refine failed: %s", exc)
            return None
