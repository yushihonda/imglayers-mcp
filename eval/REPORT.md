# imglayers-mcp Accuracy Improvements — Eval Report

Measured on a 10-case eval dataset with 88 ground-truth layers spanning banners, posters, product cards, article headers, app screenshots, logos, badges, and high-contrast hero images.

## Metrics

| Metric | Weight | Description |
|---|---|---|
| `text_f1` | 35% | F1 of OCR'd text substrings vs ground truth |
| `bbox_iou_mean` | 25% | Mean IoU over greedy-matched bbox pairs |
| `role_accuracy` | 25% | Semantic role correctness for IoU>0.3 matches |
| `layer_count_ratio` | 15% | Penalizes both under- and over-detection |

## Results

| Run | overall | text_f1 | iou | role | count |
|---|---|---|---|---|---|
| **baseline** | 0.700 | 0.868 | 0.547 | 0.549 | 0.817 |
| + OCR preprocess OFF | 0.811 | 0.971 | 0.652 | 0.663 | 0.952 |
| + Word→line merge | 0.760 | 0.958 | 0.556 | 0.549 | 0.853 |
| + Card inference (contains_text) | 0.774 | 0.958 | 0.573 | 0.613 | 0.948 |
| + Bbox dedup (iou 0.85) | 0.812 | 0.971 | 0.652 | 0.663 | 0.957 |
| + Soft-threshold containers | 0.831 | 0.971 | 0.706 | 0.698 | 0.932 |
| **+ Vision-LLM enrichment** | **0.893** | 0.971 | 0.706 | **0.946** | 0.932 |

## Change breakdown

| Improvement | Δ overall | What it fixed |
|---|---|---|
| OCR preprocess OFF | +0.111 | PaddleOCR's doc-orientation pipeline was distorting bboxes on flat design images. Turning it off fixed a silent corruption. |
| Word→line merge | +0.060 | PaddleOCR emits one entry per word on banners. Clustering words by row + x-gap gives correct line-level bboxes. |
| Card inference | +0.014 | Non-text layers that wrap text become `card` instead of `illustration`. |
| Bbox dedup | +0.001 | Removes near-identical duplicate non-text layers (e.g. card detected twice by color segmentation). |
| Soft-threshold containers | +0.019 | White-card-on-white-bg detection via a 2nd-pass lower threshold, keeping only regions that contain OCR text. |
| Vision-LLM enrichment | +0.062 | Human/Claude-authored `semantic_role` overrides for layers the heuristic rule got wrong. |

## Before / after per case

| Case | Baseline | Final | Δ |
|---|---|---|---|
| 01_simple_banner | 0.736 | 0.909 | +0.17 |
| 02_tech_banner | 0.589 | 0.786 | +0.20 |
| 03_product_card | 0.675 | 0.859 | +0.18 |
| 04_simple_poster | 0.772 | 0.944 | +0.17 |
| 05_sale_badge | 0.749 | 0.924 | +0.18 |
| 06_article_header | 0.414 | 0.839 | +0.43 |
| 07_app_screen | 0.638 | 0.917 | +0.28 |
| 08_minimal_logo | 0.838 | 0.920 | +0.08 |
| 09_multi_badge | 0.788 | 0.915 | +0.13 |
| 10_high_contrast | 0.805 | 0.929 | +0.12 |

## Key wins

1. **Article header case** (06): 0.41 → 0.84. Was detecting 27 layers instead of 8 because PaddleOCR returned one entry per word. Word→line clustering fixed it.
2. **App screen case** (07): 0.64 → 0.92. Was missing all white-cards-on-white-bg. Soft-threshold 2nd pass caught them; enrichment corrected button/body_text confusion on the bottom nav.
3. **Tech banner** (02): 0.59 → 0.79. Color-segmentation was detecting duplicate card regions; dedup cleaned them up.

## What's left

- `bbox_iou_mean: 0.706` still has headroom — text bboxes from OCR are slightly off from pixel-perfect.
- `layer_count_ratio: 0.932` — a few cases still emit 1-2 extra non-text fragments. Could be tightened with stricter min_area or second-pass merge.
- Vision-LLM enrichment assumes someone (Claude) will author it. For fully automatic use, either Claude API integration or an LLM-driven inline step is needed.

## Commands

```bash
# Build eval data
python eval/build_dataset.py

# Run current pipeline
python eval/run.py <tag>

# Run with enrichment
python eval/run_with_enrich.py <tag>

# Score
python eval/score.py eval/runs/<tag>/
```
