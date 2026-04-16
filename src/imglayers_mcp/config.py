"""Runtime configuration for the server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    project_root: Path
    default_max_side: int

    @classmethod
    def from_env(cls) -> "Config":
        root = Path(os.environ.get("IMGLAYERS_PROJECT_ROOT", "./projects")).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=root,
            default_max_side=int(os.environ.get("IMGLAYERS_MAX_SIDE", "2048")),
        )
