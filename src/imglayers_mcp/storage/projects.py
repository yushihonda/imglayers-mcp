"""Project lifecycle: allocate new project ids, load existing projects."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .paths import ProjectPaths

_PROJECT_ID_RE = re.compile(r"^proj_[A-Za-z0-9_-]+$")


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def new_project(self) -> ProjectPaths:
        now = datetime.now(timezone.utc)
        date_part = now.strftime("%Y%m%d")
        prefix = f"proj_{date_part}_"
        existing = sorted(
            int(p.name.rsplit("_", 1)[-1])
            for p in self.root.glob(f"{prefix}*")
            if p.is_dir() and p.name.rsplit("_", 1)[-1].isdigit()
        )
        next_seq = (existing[-1] + 1) if existing else 1
        pid = f"{prefix}{next_seq:03d}"
        paths = ProjectPaths(root=self.root, project_id=pid)
        paths.ensure()
        return paths

    def get(self, project_id: str) -> ProjectPaths:
        if not _PROJECT_ID_RE.match(project_id):
            raise ValueError(f"invalid project_id: {project_id!r}")
        paths = ProjectPaths(root=self.root, project_id=project_id)
        if not paths.dir.is_dir():
            raise FileNotFoundError(f"project not found: {project_id}")
        return paths

    def list_projects(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and _PROJECT_ID_RE.match(p.name))

    def load_manifest(self, project_id: str) -> dict:
        paths = self.get(project_id)
        if not paths.manifest_path.exists():
            raise FileNotFoundError(f"manifest missing for {project_id}")
        return json.loads(paths.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, paths: ProjectPaths, manifest: dict) -> None:
        paths.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
