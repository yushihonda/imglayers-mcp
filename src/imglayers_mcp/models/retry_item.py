"""Retry-queue item carrying engine-aware routing hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

RetryState = Literal["none", "queued", "retried", "accepted", "rejected"]
PreferredRetryEngine = Literal["same_engine", "layerd", "sam2", "ocr_only", "grounded_sam2"]


@dataclass
class RetryItem:
    layer_id: str
    role: str | None
    engine_used: str
    preferred_retry_engine: PreferredRetryEngine
    priority: float
    failure_signals: list[str] = field(default_factory=list)
    retry_depth: int = 0
    state: RetryState = "queued"
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "role": self.role,
            "engineUsed": self.engine_used,
            "preferredRetryEngine": self.preferred_retry_engine,
            "priority": round(self.priority, 4),
            "failureSignals": self.failure_signals,
            "retryDepth": self.retry_depth,
            "state": self.state,
            "notes": self.notes,
        }
