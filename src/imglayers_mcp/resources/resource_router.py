"""Resolve project:// URIs into (mime, bytes, relative_path) tuples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..storage.projects import ProjectStore


@dataclass(frozen=True)
class ResourcePayload:
    uri: str
    mime_type: str
    data: bytes
    path: Path


class ResourceRouter:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store

    def list_resources(self) -> list[dict]:
        items: list[dict] = []
        for pid in self.store.list_projects():
            items.extend(self._project_resources(pid))
        return items

    def _project_resources(self, project_id: str) -> list[dict]:
        try:
            paths = self.store.get(project_id)
        except FileNotFoundError:
            return []
        items: list[dict] = []
        if paths.manifest_path.exists():
            items.append(
                {
                    "uri": f"project://{project_id}/manifest",
                    "name": f"{project_id} manifest",
                    "mimeType": "application/json",
                }
            )
        if paths.preview_path.exists():
            items.append(
                {
                    "uri": f"project://{project_id}/preview",
                    "name": f"{project_id} preview",
                    "mimeType": "image/png",
                }
            )
        for extra in ("preview_annotated.png", "preview_grid.png"):
            ep = paths.preview_dir / extra
            if ep.exists():
                items.append(
                    {
                        "uri": f"project://{project_id}/preview/{extra}",
                        "name": f"{project_id} {extra}",
                        "mimeType": "image/png",
                    }
                )
        if paths.ocr_raw_path.exists():
            items.append(
                {
                    "uri": f"project://{project_id}/ocr",
                    "name": f"{project_id} ocr",
                    "mimeType": "application/json",
                }
            )
        for p in sorted(paths.layers_dir.glob("*.png")):
            items.append(
                {
                    "uri": f"project://{project_id}/layers/{p.stem}",
                    "name": f"{project_id} layer {p.stem}",
                    "mimeType": "image/png",
                }
            )
        for p in sorted(paths.exports_dir.glob("*")):
            mime = _mime_for(p)
            items.append(
                {
                    "uri": f"project://{project_id}/exports/{p.name}",
                    "name": f"{project_id} export {p.name}",
                    "mimeType": mime,
                }
            )
        return items

    def read(self, uri: str) -> ResourcePayload:
        parsed = urlparse(uri)
        if parsed.scheme != "project":
            raise ValueError(f"unsupported scheme: {parsed.scheme}")
        project_id = parsed.netloc
        path_parts = [p for p in parsed.path.split("/") if p]
        if not path_parts:
            raise ValueError("missing resource path")
        paths = self.store.get(project_id)
        head, *rest = path_parts

        if head == "manifest":
            data = paths.manifest_path.read_bytes()
            return ResourcePayload(uri, "application/json", data, paths.manifest_path)
        if head == "preview":
            target = paths.preview_path if not rest else paths.preview_dir / rest[0]
            return ResourcePayload(uri, "image/png", target.read_bytes(), target)
        if head == "ocr":
            return ResourcePayload(
                uri, "application/json", paths.ocr_raw_path.read_bytes(), paths.ocr_raw_path
            )
        if head == "layers":
            if not rest:
                raise ValueError("layer id missing")
            layer_id = rest[0]
            target = paths.layer_path(layer_id)
            return ResourcePayload(uri, "image/png", target.read_bytes(), target)
        if head == "exports":
            if not rest:
                raise ValueError("export name missing")
            target = paths.export_path(rest[0])
            return ResourcePayload(uri, _mime_for(target), target.read_bytes(), target)
        raise ValueError(f"unknown resource category: {head}")


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".psd": "image/vnd.adobe.photoshop",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix, "application/octet-stream")
