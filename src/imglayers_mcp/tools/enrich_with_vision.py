"""enrich_with_vision — apply Claude Vision-generated semantic labels to an existing manifest.

Flow:
  1. decompose_image produces a manifest with raw layers (bbox + alpha + basic kind)
  2. Caller (Claude) inspects the image and the manifest, then produces
     a list of {layer_id, semantic_role, name, codegen_hints, group_id}
  3. This tool merges those labels back into the manifest.

The pixel data is NOT touched — only semantic metadata is updated.
"""

from __future__ import annotations

from typing import Any

from ..storage.projects import ProjectStore


def enrich_with_vision(store: ProjectStore, raw: dict[str, Any]) -> dict[str, Any]:
    project_id = raw["project_id"]
    updates = raw["layer_updates"]  # list of {layer_id, semantic_role?, name?, codegen_hints?, group_id?}
    group_defs = raw.get("groups", [])  # list of {id, name, layer_ids}

    paths = store.get(project_id)
    manifest = store.load_manifest(project_id)

    layer_map = {l["id"]: l for l in manifest.get("layers", [])}
    updated_count = 0
    for upd in updates:
        lid = upd.get("layer_id")
        if lid not in layer_map:
            continue
        layer = layer_map[lid]
        if "semantic_role" in upd and upd["semantic_role"]:
            layer["semanticRole"] = upd["semantic_role"]
        if "name" in upd and upd["name"]:
            layer["name"] = upd["name"]
        if "codegen_hints" in upd and upd["codegen_hints"]:
            layer.setdefault("codegenHints", {})
            layer["codegenHints"].update(upd["codegen_hints"])
        updated_count += 1

    # Replace groups if provided.
    if group_defs:
        manifest["groups"] = [
            {"id": g["id"], "name": g.get("name", g["id"]), "layerIds": g.get("layer_ids", [])}
            for g in group_defs
        ]
        # Set children cross-reference.
        group_lookup = {lid: g["id"] for g in group_defs for lid in g.get("layer_ids", [])}
        for layer in manifest["layers"]:
            if layer["id"] in group_lookup:
                layer["children"] = [group_lookup[layer["id"]]]

    # Refresh container_group names to reflect the (possibly updated) role
    # of the parent layer. Groups are stored with their parent as the first
    # layer_id; re-fetch its role and rebuild the group's display name.
    for grp in manifest.get("groups", []):
        if not grp["id"].startswith("container_"):
            continue
        layer_ids = grp.get("layerIds", [])
        if not layer_ids:
            continue
        parent = layer_map.get(layer_ids[0])
        if parent:
            role = parent.get("semanticRole", "container")
            grp["name"] = f"{role}"

    store.write_manifest(paths, manifest)

    return {
        "project_id": project_id,
        "updated_layers": updated_count,
        "groups": len(manifest.get("groups", [])),
    }
