"""Device selection policy for heavy segmentation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DeviceKind = Literal["cuda", "mps", "cpu"]
CheckpointKind = Literal["tiny", "small", "base_plus", "large", "auto"]


@dataclass
class RuntimeDevice:
    device: DeviceKind
    checkpoint: CheckpointKind
    max_input_side: int
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "checkpoint": self.checkpoint,
            "maxInputSide": self.max_input_side,
            "notes": self.notes,
        }


def resolve_device(
    preference: str = "auto",
    requested_checkpoint: str = "auto",
    *,
    allow_cuda_extension: bool = True,
    allow_mps: bool = True,
    allow_cpu_fallback: bool = True,
) -> RuntimeDevice:
    notes: list[str] = []
    device: DeviceKind = "cpu"
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        has_cuda = False
        has_mps = False
        notes.append("torch_unavailable")

    if preference == "cuda" or (preference == "auto" and has_cuda):
        if has_cuda:
            device = "cuda"
        else:
            notes.append("cuda_requested_but_unavailable")
    if device == "cpu" and (preference == "mps" or preference == "auto") and has_mps and allow_mps:
        device = "mps"
    if device == "cpu" and preference not in ("cpu", "auto"):
        notes.append(f"{preference}_unavailable_fallback_cpu")
    if device == "cpu" and not allow_cpu_fallback:
        notes.append("cpu_fallback_disabled")

    checkpoint: CheckpointKind
    if requested_checkpoint == "auto":
        if device == "cuda":
            checkpoint = "base_plus"
        elif device == "mps":
            checkpoint = "small"
        else:
            checkpoint = "tiny"
    else:
        checkpoint = requested_checkpoint  # type: ignore[assignment]

    max_input_side = {"cuda": 1536, "mps": 1024, "cpu": 768}[device]

    if device == "cuda" and not allow_cuda_extension:
        notes.append("cuda_extension_disabled")

    return RuntimeDevice(
        device=device,
        checkpoint=checkpoint,
        max_input_side=max_input_side,
        notes=notes,
    )
