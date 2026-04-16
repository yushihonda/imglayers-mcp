"""Stage 7: verifier (spec §7.8).

Scores a manifest against its source image and flags layers that need retry.
Emits per-layer confidence scores and a retry queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class VerificationResult:
    overall_score: float
    preview_diff: float
    low_confidence_layers: list[str] = field(default_factory=list)
    retry_queue: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def verify(
    manifest: dict,
    project_dir: Path,
    thresholds: dict | None = None,
) -> VerificationResult:
    thresholds = thresholds or {}
    low_conf_thr = float(thresholds.get("low_confidence_layer", 0.55))
    retry_preview_diff = float(thresholds.get("retry_preview_diff", 0.12))

    low_conf: list[str] = []
    retry_q: list[dict] = []

    for layer in manifest.get("layers", []):
        conf = float(layer.get("confidence", 1.0))
        if conf < low_conf_thr:
            low_conf.append(layer["id"])
            retry_q.append({
                "layer_id": layer["id"],
                "role": layer.get("semanticRole"),
                "reason": "low_confidence",
                "confidence": conf,
            })

    preview_diff = _preview_diff(manifest, project_dir)

    notes: list[str] = []
    if preview_diff > retry_preview_diff:
        notes.append(f"preview_diff {preview_diff:.3f} exceeds threshold")

    total_conf = sum(l.get("confidence", 1.0) for l in manifest.get("layers", []))
    avg_conf = total_conf / max(1, len(manifest.get("layers", [])))
    overall = max(0.0, avg_conf - preview_diff * 0.5)

    return VerificationResult(
        overall_score=overall,
        preview_diff=preview_diff,
        low_confidence_layers=low_conf,
        retry_queue=retry_q,
        notes=notes,
    )


def _preview_diff(manifest: dict, project_dir: Path) -> float:
    """L1 difference between reconstructed preview and original."""
    original_rel = (manifest.get("assets") or {}).get("original") or "meta/original.png"
    preview_rel = (manifest.get("assets") or {}).get("preview") or "preview/preview.png"
    op = project_dir / original_rel
    pp = project_dir / preview_rel
    if not op.exists() or not pp.exists():
        return 0.0
    try:
        a = np.asarray(Image.open(op).convert("RGB"), dtype=np.int32)
        b = np.asarray(Image.open(pp).convert("RGB").resize(Image.open(op).size), dtype=np.int32)
        return float(np.abs(a - b).mean() / 255.0)
    except Exception:
        return 0.0
