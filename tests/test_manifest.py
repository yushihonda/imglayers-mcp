import json
from datetime import datetime, timezone

import pytest

from imglayers_mcp.models.manifest import (
    AssetIndex,
    BBox,
    CanvasInfo,
    ExportIndex,
    LayerAsset,
    LayerNode,
    Manifest,
    PipelineInfo,
    Provenance,
    SourceInfo,
    StatsInfo,
)


def test_manifest_roundtrip():
    m = Manifest(
        projectId="proj_20260416_001",
        createdAt=datetime.now(timezone.utc).isoformat(),
        source=SourceInfo(
            inputUri="/tmp/x.png", originalFileName="x.png", mimeType="image/png", width=100, height=80
        ),
        canvas=CanvasInfo(width=100, height=80, background="#FFFFFF"),
        pipeline=PipelineInfo(
            engineRequested="auto",
            engineSelected="layerd",
            enableOCR=True,
            detailLevel="balanced",
        ),
        stats=StatsInfo(totalLayers=1, textLayers=0, imageLayers=1, vectorLikeLayers=0, unknownLayers=0),
        assets=AssetIndex(original="meta/original.png", preview="preview/preview.png", layersDir="layers"),
        layers=[
            LayerNode(
                id="bg_001",
                name="background",
                type="image",
                semanticRole="background",
                bbox=BBox(x=0, y=0, width=100, height=80),
                zIndex=0,
                asset=LayerAsset(path="layers/bg_001.png", format="png", hasAlpha=False),
                provenance=Provenance(engines=["layerd"], confidence=0.9),
            )
        ],
        exports=ExportIndex(manifest="manifest.json"),
    )
    as_json = m.to_json_dict()
    # Must round-trip through JSON.
    dumped = json.dumps(as_json, ensure_ascii=False)
    loaded = json.loads(dumped)
    assert loaded["projectId"] == "proj_20260416_001"
    assert loaded["layers"][0]["semanticRole"] == "background"
    assert loaded["layers"][0]["bbox"]["width"] == 100
    assert loaded["pipeline"]["engineSelected"] == "layerd"


def test_invalid_semantic_role_rejected():
    with pytest.raises(Exception):
        LayerNode(
            id="x",
            name="x",
            type="image",
            semanticRole="not-a-role",  # type: ignore[arg-type]
            bbox=BBox(x=0, y=0, width=1, height=1),
            zIndex=0,
            provenance=Provenance(engines=["layerd"], confidence=0.5),
        )
