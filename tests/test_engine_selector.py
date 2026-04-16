from imglayers_mcp.core.engine_selector import select_engine
from imglayers_mcp.core.preflight import PreflightFeatures


def _feat(**kw):
    base = dict(
        width=800,
        height=600,
        has_alpha=False,
        text_density=0.0,
        layout_regularity=0.0,
        palette_compactness=0.0,
        photo_score=0.0,
        aspect_ratio=800 / 600,
        gpu_available=False,
        estimated_complexity=0.0,
    )
    base.update(kw)
    return PreflightFeatures(**base)


def test_explicit_request_wins():
    d = select_engine(_feat(gpu_available=True, photo_score=0.9), requested="layerd", qwen_enabled=True, qwen_available=True)
    assert d.selected == "layerd"

    d = select_engine(_feat(gpu_available=True, photo_score=0.9), requested="qwen-layered", qwen_enabled=True, qwen_available=True)
    assert d.selected == "qwen-layered"


def test_no_gpu_falls_to_layerd():
    d = select_engine(_feat(gpu_available=False, photo_score=0.9), qwen_enabled=True, qwen_available=True)
    assert d.selected == "layerd"
    assert "GPU" in d.reason


def test_text_dense_prefers_layerd():
    d = select_engine(
        _feat(gpu_available=True, text_density=0.1, photo_score=0.9),
        qwen_enabled=True,
        qwen_available=True,
    )
    assert d.selected == "layerd"


def test_layout_regular_prefers_layerd():
    d = select_engine(
        _feat(gpu_available=True, layout_regularity=0.8, photo_score=0.9),
        qwen_enabled=True,
        qwen_available=True,
    )
    assert d.selected == "layerd"


def test_photo_like_prefers_qwen_when_available():
    d = select_engine(
        _feat(gpu_available=True, photo_score=0.85),
        qwen_enabled=True,
        qwen_available=True,
    )
    assert d.selected == "qwen-layered"


def test_photo_like_without_qwen_falls_back():
    d = select_engine(
        _feat(gpu_available=True, photo_score=0.85),
        qwen_enabled=False,
        qwen_available=False,
    )
    assert d.selected == "layerd"
