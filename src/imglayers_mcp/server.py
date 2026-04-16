"""MCP server entrypoint.

Exposes `apply_vision_analysis` as the primary decomposition tool — the caller
(a vision-capable LLM) provides the element list, this server cuts pixels and
builds the manifest. Additional tools: get_manifest, list_layers, get_layer,
render_preview, render_annotated_preview, export_project, derive_codegen_plan.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .config import Config
from .core.orchestrator import Orchestrator
from .resources.resource_router import ResourceRouter
from .storage.projects import ProjectStore
from .tools import (
    apply_vision_analysis,
    decompose_image,
    derive_codegen_plan,
    enrich_with_vision,
    export_project,
    get_layer,
    get_manifest,
    list_layers,
    render_annotated_preview_tool,
    render_preview,
)
from .utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _build_tool_specs(orchestrator: Orchestrator, store: ProjectStore) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="decompose_image",
            description=(
                "Automatic decomposition. engine=layerd runs the fast LayerD (CV) + "
                "PaddleOCR path; engine=sam2 runs SAM2 mask generation with alpha "
                "refinement; engine=hybrid picks per image-type and allows cross-engine "
                "retry. Produces a manifest with RGBA layers, OCR text, and style hints."
            ),
            input_schema={
                "type": "object",
                "required": ["input_uri"],
                "properties": {
                    "input_uri": {"type": "string"},
                    "engine": {"type": "string", "enum": ["layerd", "sam2", "hybrid"], "default": "layerd"},
                    "device_preference": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
                    "sam2_checkpoint": {"type": "string", "enum": ["auto", "tiny", "small", "base_plus", "large"], "default": "auto"},
                    "allow_cross_engine_retry": {"type": "boolean", "default": True},
                    "detail_level": {"type": "string", "enum": ["fast", "balanced", "high"], "default": "balanced"},
                    "max_side": {"type": "integer", "default": 2048},
                    "text_granularity": {"type": "string", "enum": ["char", "line", "block"], "default": "line"},
                    "enable_ocr": {"type": "boolean", "default": True},
                    "export_formats": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["manifest", "svg", "psd", "codegen-plan", "preview"]},
                        "default": ["manifest"],
                    },
                    "open_in_browser": {"type": "boolean", "default": True},
                },
            },
            handler=lambda raw: decompose_image(orchestrator, raw),
        ),
        ToolSpec(
            name="apply_vision_analysis",
            description=(
                "Vision-LLM-driven decomposition: caller supplies a complete element list "
                "with bboxes + types + text_content. Cuts pixels per element, bypasses CV/OCR. "
                "Use this when the automatic decompose_image misses elements or mis-groups "
                "them (dark gradients, complex layouts). A vision-capable LLM analyzes the "
                "image first, produces the element list, then invokes this tool."
            ),
            input_schema={
                "type": "object",
                "required": ["input_uri", "elements"],
                "properties": {
                    "input_uri": {"type": "string"},
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "bbox"],
                            "properties": {
                                "type": {"type": "string"},
                                "bbox": {
                                    "type": "object",
                                    "required": ["x", "y", "width", "height"],
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                        "width": {"type": "number"},
                                        "height": {"type": "number"},
                                    },
                                },
                                "name": {"type": "string"},
                                "text_content": {"type": ["string", "null"]},
                                "color": {"type": ["string", "null"]},
                                "font_size": {"type": ["number", "null"]},
                                "font_weight": {"type": ["integer", "null"]},
                            },
                        },
                    },
                    "text_granularity": {"type": "string", "enum": ["char", "line", "block"], "default": "line"},
                    "export_formats": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["manifest", "svg", "psd", "codegen-plan", "preview"]},
                        "default": ["manifest"],
                    },
                    "open_in_browser": {"type": "boolean", "default": True},
                },
            },
            handler=lambda raw: apply_vision_analysis(orchestrator, raw),
        ),
        ToolSpec(
            name="enrich_with_vision",
            description=(
                "Update an existing manifest with vision-LLM-provided semantic metadata. "
                "Pass layer_updates (semantic_role, name, codegen_hints) and groups "
                "(id, name, layer_ids). The pixel data is never touched — only metadata. "
                "Typical flow: decompose_image → look at annotated preview → enrich_with_vision "
                "to add semantic roles and Figma-style groups."
            ),
            input_schema={
                "type": "object",
                "required": ["project_id", "layer_updates"],
                "properties": {
                    "project_id": {"type": "string"},
                    "layer_updates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["layer_id"],
                            "properties": {
                                "layer_id": {"type": "string"},
                                "semantic_role": {"type": ["string", "null"]},
                                "name": {"type": ["string", "null"]},
                                "codegen_hints": {"type": ["object", "null"]},
                            },
                        },
                    },
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "layer_ids"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "layer_ids": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
            },
            handler=lambda raw: enrich_with_vision(store, raw),
        ),
        ToolSpec(
            name="get_manifest",
            description="Return manifest.json for a project.",
            input_schema={
                "type": "object",
                "required": ["project_id"],
                "properties": {"project_id": {"type": "string"}},
            },
            handler=lambda raw: get_manifest(store, raw),
        ),
        ToolSpec(
            name="list_layers",
            description="List layers with optional filters.",
            input_schema={
                "type": "object",
                "required": ["project_id"],
                "properties": {
                    "project_id": {"type": "string"},
                    "filter": {
                        "type": "object",
                        "properties": {
                            "type": {"type": ["string", "null"]},
                            "semantic_role": {"type": ["string", "null"]},
                            "editable_only": {"type": "boolean"},
                        },
                    },
                },
            },
            handler=lambda raw: list_layers(store, raw),
        ),
        ToolSpec(
            name="get_layer",
            description="Return detailed info for one layer.",
            input_schema={
                "type": "object",
                "required": ["project_id", "layer_id"],
                "properties": {
                    "project_id": {"type": "string"},
                    "layer_id": {"type": "string"},
                    "include_asset_path": {"type": "boolean", "default": True},
                },
            },
            handler=lambda raw: get_layer(store, raw),
        ),
        ToolSpec(
            name="render_preview",
            description="Re-render the plain composited preview from the current manifest.",
            input_schema={
                "type": "object",
                "required": ["project_id"],
                "properties": {
                    "project_id": {"type": "string"},
                    "output_name": {"type": "string", "default": "preview.png"},
                },
            },
            handler=lambda raw: render_preview(store, raw),
        ),
        ToolSpec(
            name="render_annotated_preview",
            description=(
                "Generate a diagnostic preview showing how the image was decomposed. "
                "style=overlay draws labelled bboxes on the composited preview; "
                "style=grid arranges one thumbnail per layer."
            ),
            input_schema={
                "type": "object",
                "required": ["project_id"],
                "properties": {
                    "project_id": {"type": "string"},
                    "style": {"type": "string", "enum": ["overlay", "grid"], "default": "overlay"},
                    "output_name": {"type": ["string", "null"]},
                    "show_labels": {"type": "boolean", "default": True},
                    "include_background": {"type": "boolean", "default": False},
                    "only_ids": {"type": ["array", "null"], "items": {"type": "string"}},
                    "stroke_width": {"type": "integer", "default": 3},
                    "columns": {"type": "integer", "default": 3},
                },
            },
            handler=lambda raw: render_annotated_preview_tool(store, raw),
        ),
        ToolSpec(
            name="export_project",
            description="Generate derived outputs (svg/psd/codegen-plan/preview) from the manifest.",
            input_schema={
                "type": "object",
                "required": ["project_id"],
                "properties": {
                    "project_id": {"type": "string"},
                    "formats": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["manifest", "svg", "psd", "codegen-plan", "preview"]},
                    },
                },
            },
            handler=lambda raw: export_project(store, raw),
        ),
        ToolSpec(
            name="derive_codegen_plan",
            description="Produce a structured component plan from the manifest for downstream codegen.",
            input_schema={
                "type": "object",
                "required": ["project_id"],
                "properties": {
                    "project_id": {"type": "string"},
                    "target": {"type": "string", "enum": ["react", "html", "generic"], "default": "generic"},
                },
            },
            handler=lambda raw: derive_codegen_plan(store, raw),
        ),
    ]


# ---------------------------------------------------------------------------
# MCP SDK path — preferred when installed.
# ---------------------------------------------------------------------------
async def _run_mcp_sdk(tools: list[ToolSpec], router: ResourceRouter) -> None:
    from mcp.server import Server  # type: ignore
    from mcp.server.stdio import stdio_server  # type: ignore
    import mcp.types as types  # type: ignore

    server = Server("imglayers-mcp")
    tool_map = {t.name: t for t in tools}

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=t.name, description=t.description, inputSchema=t.input_schema)
            for t in tools
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name not in tool_map:
            raise ValueError(f"unknown tool: {name}")
        try:
            result = tool_map[name].handler(arguments or {})
        except Exception as exc:
            log.exception("tool %s failed", name)
            return [types.TextContent(type="text", text=json.dumps({"error": {"message": str(exc)}}))]
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        return [
            types.Resource(uri=r["uri"], name=r["name"], mimeType=r["mimeType"])
            for r in router.list_resources()
        ]

    @server.read_resource()
    async def _read_resource(uri: str) -> str | bytes:
        payload = router.read(uri)
        if payload.mime_type == "application/json":
            return payload.data.decode("utf-8")
        return payload.data

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


# ---------------------------------------------------------------------------
# JSON-RPC fallback (minimal, stdio line-delimited).
# ---------------------------------------------------------------------------
def _run_jsonrpc_fallback(tools: list[ToolSpec], router: ResourceRouter) -> None:
    tool_map = {t.name: t for t in tools}

    def write(msg: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def tool_list_response() -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            for t in tools
        ]

    def handle(req: dict[str, Any]) -> dict[str, Any] | None:
        method = req.get("method")
        req_id = req.get("id")
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "imglayers-mcp", "version": "0.1.0"},
                    "capabilities": {"tools": {}, "resources": {}},
                }
            elif method in ("notifications/initialized", "initialized"):
                return None
            elif method == "tools/list":
                result = {"tools": tool_list_response()}
            elif method == "tools/call":
                params = req.get("params") or {}
                name = params.get("name")
                args = params.get("arguments") or {}
                if name not in tool_map:
                    raise ValueError(f"unknown tool: {name}")
                payload = tool_map[name].handler(args)
                result = {
                    "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                    "isError": False,
                }
            elif method == "resources/list":
                result = {"resources": router.list_resources()}
            elif method == "resources/read":
                params = req.get("params") or {}
                uri = params.get("uri")
                payload = router.read(uri)
                if payload.mime_type.startswith("image/") or payload.mime_type == "image/vnd.adobe.photoshop":
                    result = {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": payload.mime_type,
                                "blob": base64.b64encode(payload.data).decode("ascii"),
                            }
                        ]
                    }
                else:
                    result = {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": payload.mime_type,
                                "text": payload.data.decode("utf-8"),
                            }
                        ]
                    }
            elif method == "ping":
                result = {}
            elif method == "shutdown":
                return {"jsonrpc": "2.0", "id": req_id, "result": None, "_stop": True}
            else:
                raise ValueError(f"unknown method: {method}")
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:
            log.exception("jsonrpc handler error")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc), "data": {"trace": traceback.format_exc()}},
            }

    log.warning("mcp SDK not installed; running line-delimited JSON-RPC fallback on stdio")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}})
            continue
        response = handle(req)
        if response is None:
            continue
        stop = response.pop("_stop", False)
        write(response)
        if stop:
            break


def build_runtime(config: Config | None = None) -> tuple[list[ToolSpec], ResourceRouter, Orchestrator, ProjectStore]:
    config = config or Config.from_env()
    store = ProjectStore(root=config.project_root)
    orchestrator = Orchestrator(config, store)
    router = ResourceRouter(store)
    tools = _build_tool_specs(orchestrator, store)
    return tools, router, orchestrator, store


def main() -> None:
    config = Config.from_env()
    tools, router, _, _ = build_runtime(config)
    try:
        import mcp  # type: ignore  # noqa: F401

        asyncio.run(_run_mcp_sdk(tools, router))
    except ImportError:
        _run_jsonrpc_fallback(tools, router)


if __name__ == "__main__":
    main()
