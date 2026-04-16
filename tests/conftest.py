from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def synth_banner(tmp_path: Path) -> Path:
    """Synthetic banner: white bg, blue headline bar, orange button rectangle."""
    img = Image.new("RGB", (640, 320), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(40, 60), (600, 120)], fill=(40, 80, 200))
    draw.rectangle([(40, 200), (260, 260)], fill=(230, 110, 40))
    path = tmp_path / "banner.png"
    img.save(path)
    return path


@pytest.fixture
def tiny_rgba() -> np.ndarray:
    arr = np.zeros((20, 20, 4), dtype=np.uint8)
    arr[2:8, 2:8] = [255, 0, 0, 255]
    arr[10:18, 10:18] = [0, 255, 0, 255]
    arr[..., 3] = np.where(arr[..., :3].sum(axis=2) > 0, 255, 0)
    return arr
