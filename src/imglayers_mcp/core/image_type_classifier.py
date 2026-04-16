"""Stage 1: image type classifier (spec §7.2).

Classifies an input image into one of the design archetypes. The label
drives downstream preprocessing (OCR orientation/unwarping), LayerD
refinement strength, and retry-segmentation trigger thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ImageType = Literal["ui_mock", "banner", "poster", "illustration", "photo_mixed", "scan_capture"]


@dataclass
class ImageTypeFeatures:
    text_density: float
    palette_compactness: float
    layout_regularity: float
    edge_density: float
    photo_score: float
    aspect_ratio: float
    has_alpha: bool

    def to_dict(self) -> dict:
        return {
            "text_density": round(self.text_density, 3),
            "palette_compactness": round(self.palette_compactness, 3),
            "layout_regularity": round(self.layout_regularity, 3),
            "edge_density": round(self.edge_density, 3),
            "photo_score": round(self.photo_score, 3),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "has_alpha": self.has_alpha,
        }


@dataclass
class ImageTypeResult:
    image_type: ImageType
    features: ImageTypeFeatures
    confidence: float

    def to_dict(self) -> dict:
        return {
            "image_type": self.image_type,
            "confidence": round(self.confidence, 3),
            "features": self.features.to_dict(),
        }


def classify(rgba: np.ndarray, ocr_boxes: list | None = None) -> ImageTypeResult:
    """Classify image type from pixel statistics and optional OCR hints."""
    features = _extract_features(rgba, ocr_boxes or [])
    image_type, confidence = _route(features)
    return ImageTypeResult(image_type=image_type, features=features, confidence=confidence)


def _extract_features(rgba: np.ndarray, ocr_boxes: list) -> ImageTypeFeatures:
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3]

    # Palette compactness: ratio of canvas covered by top-3 color buckets.
    q = (rgb.astype(np.int32) // 24) * 24 + 12
    packed = (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]
    unique, counts = np.unique(packed.reshape(-1), return_counts=True)
    order = np.argsort(-counts)
    top3 = counts[order[:3]].sum() if len(counts) >= 3 else counts.sum()
    palette_compactness = float(top3) / float(packed.size)

    # Edge density: proportion of high-gradient pixels.
    gray = rgb.astype(np.int32).sum(axis=2) / 3
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edges = (gx[:-1, :] > 20) | (gy[:, :-1] > 20)
    edge_density = float(edges.mean())

    # Photo score: color variety × edge density (photos are colorful + textured).
    photo_score = min(1.0, (1.0 - palette_compactness) * 1.2 + edge_density * 2.0)
    photo_score = max(0.0, photo_score - 0.2)

    # Layout regularity: proportion of distinct y-positions of OCR boxes
    # that align to grid (simple proxy).
    layout_regularity = 0.5
    if ocr_boxes:
        ys = sorted(
            b[1] if isinstance(b, (tuple, list)) else b.y
            for b in ocr_boxes
        )
        if len(ys) >= 2:
            diffs = np.diff(ys)
            if len(diffs) > 0:
                reg = float((np.abs(diffs - np.median(diffs)) < max(2.0, np.median(diffs) * 0.1)).mean())
                layout_regularity = reg

    # Text density: fraction of canvas covered by OCR bbox area.
    text_density = 0.0
    if ocr_boxes:
        total = 0.0
        for b in ocr_boxes:
            if isinstance(b, (tuple, list)):
                total += b[2] * b[3]
            else:
                total += b.w * b.h
        text_density = min(1.0, total / float(h * w))

    aspect_ratio = w / max(1.0, h)
    has_alpha = rgba.shape[2] == 4 and bool((rgba[..., 3] < 255).any())

    return ImageTypeFeatures(
        text_density=text_density,
        palette_compactness=palette_compactness,
        layout_regularity=layout_regularity,
        edge_density=edge_density,
        photo_score=photo_score,
        aspect_ratio=aspect_ratio,
        has_alpha=has_alpha,
    )


def _route(f: ImageTypeFeatures) -> tuple[ImageType, float]:
    """Spec §7.2 routing rules, lightly enhanced."""
    # Photo-mixed: dominated by texture/color variety.
    if f.photo_score > 0.72:
        return "photo_mixed", 0.75
    # UI mock: dense text + regular layout.
    if f.text_density > 0.06 and f.layout_regularity > 0.55:
        return "ui_mock", 0.8
    # Scan/capture: high aspect-1, low palette compactness, moderate photo.
    if f.palette_compactness < 0.35 and f.edge_density > 0.08:
        return "scan_capture", 0.6
    # Banner: strong palette (flat graphics), some text.
    if f.palette_compactness > 0.65 and f.text_density > 0.01:
        return "banner", 0.85
    # Poster: strong palette but tall aspect or heavy headline.
    if f.palette_compactness > 0.55 and f.aspect_ratio < 1.1:
        return "poster", 0.7
    # Illustration: flat palette, low text.
    return "illustration", 0.6
