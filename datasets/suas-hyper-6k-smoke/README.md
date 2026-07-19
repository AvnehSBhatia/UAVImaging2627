# suas-hyper-6k

Hyper-accurate SUAS set from `datasetgen-2026/gen_hyper`.

- background-only: 4
- single-object: 4 (mannequin_fraction=0.5)
- scenarios: open_field, runway_drygrass, tree_clearing, brush_occlusion
- classes: 0=mannequin, 1=tent
- canvas: 1920x1080 YOLO labels

Train:
```bash
cd ANetV1
DATA_ROOT=../datasets/suas-hyper-6k python scripts/train_anet.py
```
