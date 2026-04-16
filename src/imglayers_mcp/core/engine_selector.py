"""Engine routing: pick initial decomposition backend and retry engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EngineKind = Literal["layerd", "sam2", "hybrid"]
RoutedEngine = Literal["layerd", "sam2"]


@dataclass
class EngineDecision:
    requested: EngineKind
    initial: RoutedEngine
    sam2_available: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "initial": self.initial,
            "sam2Available": self.sam2_available,
            "reason": self.reason,
        }


def route_initial(
    requested: str,
    image_type: str,
    *,
    sam2_available: bool,
) -> EngineDecision:
    requested_norm: EngineKind = requested if requested in ("layerd", "sam2", "hybrid") else "layerd"  # type: ignore[assignment]

    if requested_norm == "layerd":
        return EngineDecision(
            requested=requested_norm, initial="layerd",
            sam2_available=sam2_available, reason="explicit_layerd",
        )
    if requested_norm == "sam2":
        if sam2_available:
            return EngineDecision(
                requested=requested_norm, initial="sam2",
                sam2_available=True, reason="explicit_sam2",
            )
        return EngineDecision(
            requested=requested_norm, initial="layerd",
            sam2_available=False, reason="sam2_requested_but_unavailable",
        )
    # hybrid
    prefers_sam2 = image_type in ("illustration", "photo_mixed")
    if prefers_sam2 and sam2_available:
        return EngineDecision(
            requested="hybrid", initial="sam2",
            sam2_available=True, reason=f"hybrid_sam2_for_{image_type}",
        )
    return EngineDecision(
        requested="hybrid", initial="layerd",
        sam2_available=sam2_available, reason=f"hybrid_layerd_for_{image_type}",
    )


def preferred_retry_engine(
    layer_engine_used: str,
    role: str | None,
    failure_signals: list[str],
    *,
    image_type: str,
    sam2_available: bool,
) -> str:
    role = role or "unknown"

    if role in ("background", "unknown"):
        return "same_engine"

    is_text = role in ("headline", "subheadline", "body_text", "button", "badge", "logo") and "ocr_inconsistency" in failure_signals
    if is_text:
        return "ocr_only"

    soft_signals = {"alpha_edge_mismatch", "high_residual_p95", "mask_fragmentation"}
    has_soft = bool(soft_signals.intersection(failure_signals))

    if layer_engine_used == "layerd" and sam2_available and has_soft:
        return "sam2"
    if layer_engine_used == "sam2" and (
        "fragmentation" in failure_signals or image_type in ("ui_mock", "banner", "poster")
    ):
        return "layerd"
    return "same_engine"
