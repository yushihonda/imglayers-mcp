# Ablation report

| metric | baseline | v7-stage-driven | v8-phase-a-raw | v9-final |
|---|---|---|---|---|
| **alpha_iou** | 0.5472 | 0.7169 | 0.7169 | 0.7119 |
| **rgb_l1** | 0.0182 | 0.0042 | 0.0042 | 0.0042 |
| **preview_diff** | 0.0182 | 0.0042 | 0.0042 | 0.0042 |
| **layers_edit_dist** | 0.7768 | 0.4607 | 0.7385 | 0.7385 |
| **style_fit_score_mean** | 0.0000 | 0.0758 | 0.0942 | 0.1344 |
| **ocr_reread_consistency_mean** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **retry_success_rate** | 0.0000 | 0.0000 | 0.0000 | 0.6556 |

## By image_type

### illustration
| metric | baseline | v7-stage-driven | v8-phase-a-raw | v9-final |
|---|---|---|---|---|
| alpha_iou | - | 0.7156 | 0.7156 | 0.7117 |
| rgb_l1 | - | 0.0058 | 0.0058 | 0.0052 |
| preview_diff | - | 0.0058 | 0.0058 | 0.0052 |
| layers_edit_dist | - | 0.4745 | 0.8052 | 0.8052 |
| style_fit_score_mean | - | 0.0521 | 0.0702 | 0.0999 |
| ocr_reread_consistency_mean | - | 0.0000 | 0.0000 | 0.0000 |
| retry_success_rate | - | 0.0000 | 0.0000 | 0.6746 |

### poster
| metric | baseline | v7-stage-driven | v8-phase-a-raw | v9-final |
|---|---|---|---|---|
| alpha_iou | - | 0.7199 | 0.7199 | 0.7125 |
| rgb_l1 | - | 0.0005 | 0.0005 | 0.0018 |
| preview_diff | - | 0.0005 | 0.0005 | 0.0018 |
| layers_edit_dist | - | 0.4286 | 0.5831 | 0.5831 |
| style_fit_score_mean | - | 0.1310 | 0.1504 | 0.2149 |
| ocr_reread_consistency_mean | - | 0.0000 | 0.0000 | 0.0000 |
| retry_success_rate | - | 0.0000 | 0.0000 | 0.6111 |

### unknown
| metric | baseline | v7-stage-driven | v8-phase-a-raw | v9-final |
|---|---|---|---|---|
| alpha_iou | 0.5472 | - | - | - |
| rgb_l1 | 0.0182 | - | - | - |
| preview_diff | 0.0182 | - | - | - |
| layers_edit_dist | 0.7768 | - | - | - |
| style_fit_score_mean | 0.0000 | - | - | - |
| ocr_reread_consistency_mean | 0.0000 | - | - | - |
| retry_success_rate | 0.0000 | - | - | - |
