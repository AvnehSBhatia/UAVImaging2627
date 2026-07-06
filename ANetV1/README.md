# ANetV1

17k-parameter per-cell detector for SUAS mannequin/tent search. Full spec and
design record: [ARCHITECTURE.md](ARCHITECTURE.md).

Dataset: `datasets/suas-synth-50k` (YOLO format; synthetic 1920x1080 + VisDrone
mixed sizes — everything is letterboxed to 960x540, see `anet/data/`).

## Run order (all on MPS)

```bash
.venv/bin/pip install -r ANetV1/requirements.txt
cd ANetV1

# 0. sanity: ~17k params, fwd/bwd, dataset loading
python scripts/smoke_test.py

# 1. baseline + future teacher
python scripts/train_yolo.py

# 2. ANetV1 from scratch
python scripts/train_anet.py

# 3. distilled ANetV1
python scripts/cache_teacher.py --weights runs/yolo/baseline/weights/best.pt
python scripts/train_anet_distill.py

# 4. three-way comparison (test split, identical cell metrics)
python scripts/evaluate_all.py \
  --yolo runs/yolo/baseline/weights/best.pt \
  --anet runs/anet/best.pt \
  --anet-distill runs/anet_distill/best.pt
```

Decision metric (ARCHITECTURE.md §10): worst-decile mannequin object recall.
If MPS hits an unsupported op, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
