# Grocery-Store-Product-Classification-Shelf-Monitoring


```markdown
# Grocery Store Product Classification & Shelf Monitoring
**Leeds School of Business · MSBC5190 Modern AI · Spring 2026 · Group 2**

![Python](https://img.shields.io/badge/Python-3.10-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green) ![Roboflow](https://img.shields.io/badge/Annotated-Roboflow-orange)

## Overview
An end-to-end computer vision pipeline that detects and classifies 19 fresh produce categories from grocery shelf images, estimates shelf space allocation per product, and generates planogram compliance reports (PASS/FAIL within ±15% threshold).

## Results
| Phase | Dataset | mAP50 |
|-------|---------|-------|
| Phase 1 — system logic | 124 synthetic images | 0.784 |
| Phase 2 — benchmark | 20 synthetic shelves | 0.784 |
| Phase 3 — real world | 11 annotated images | 0.014 (−0.770) |

- Mean shelf space error: **2.93%** across 20 synthetic shelves
- Detection rate on synthetic data: **100%**

## Data
- **Grocery Store Dataset** (Klasson et al., 2019) — 2,640 labeled produce images, 19 coarse classes
- **Synthetic shelf dataset** — 400+ procedurally generated shelf images, auto-annotated in YOLO format
- **Real-world dataset** — 11 manually annotated grocery shelf photos (Roboflow), held out for evaluation only

## Methodology
- Fine-tuned YOLOv8n on COCO over 100 epochs using AdamW
- Augmentations (color jitter, random scaling) to reduce synthetic-to-real gap
- Pixel masking for shelf space estimation — prevents overlap inflation
- Citrus classes merged due to near-identical visual features

## Failure Modes (Real-World Gap)
The 77-point mAP50 drop on real images stems from three root causes:
1. **Class hallucination** — Pineapple predicted where none exists
2. **Fallback misclassification** — Melon/Kiwi as default uncertain predictions
3. **Non-detection** — 10 of 12 real classes produced zero true positives

## Tech Stack
`Python` `YOLOv8` `Roboflow` `AdamW` `COCO pretraining` `Synthetic data generation`

## References
Klasson, M., Zhang, C., & Kjellström, H. (2019). A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels. *IEEE WACV*.
```
