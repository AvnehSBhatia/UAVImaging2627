# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

SUAS 2026 UAV target detection: find mannequins and tents in nadir aerial frames (150 ft AGL) and run at ≥30 FPS on a Raspberry Pi 5 + AI HAT+ (Hailo-8). Not a git repository. Python via the repo-root venv: use `.venv/bin/python` / `.venv/bin/pip`.

Three sibling projects:

- **`ANetV1/`** — the custom 17k-parameter per-cell detector plus a YOLO baseline and a distillation pipeline. `ARCHITECTURE.md` is the locked v6 spec and the authoritative design record (every mechanism, parameter count, and design decision D1–D30 with rationale). Read it before touching model code; changes to model semantics must stay consistent with it or update it.
- **`datasetgen-2026/gen2/`** — the current synthetic dataset generator (photorealistic compositing: CLIP-gated OpenAerialMap backgrounds, Blender-rendered assets, sun-consistent shadows, Reinhard harmonization, full sensor sim). Deterministic per-index seeds, resume-safe. `datasetgen-2026/uav-yolo-dataset-generator/` is the legacy gen1 tool — don't extend it.
- **`datasets/`** — generated data. `suas-synth-50k/` is the active YOLO-format dataset (synthetic + VisDrone person subset remapped to `mannequin`); `gen-assets/` holds generator inputs; its README documents composition and caveats.

## Commands

```bash
.venv/bin/pip install -r ANetV1/requirements.txt
cd ANetV1

# sanity: ~17k params, fwd/bwd, dataset loading — run after any model change
python scripts/smoke_test.py

# training pipeline, in order
python scripts/train_yolo.py                                          # 1. YOLO baseline + future teacher
python scripts/train_anet.py                                          # 2. ANetV1 from scratch
python scripts/cache_teacher.py --weights runs/yolo/baseline/weights/best.pt
python scripts/train_anet_distill.py                                  # 3. distilled ANetV1

# three-way comparison on the test split (identical cell metrics)
python scripts/evaluate_all.py \
  --yolo runs/yolo/baseline/weights/best.pt \
  --anet runs/anet/best.pt \
  --anet-distill runs/anet_distill/best.pt

# fast local inference (see ARCHITECTURE.md D34/D35):
#   fastest: anet.metal.MetalANet.from_checkpoint(ckpt) — fused Metal kernel, ~7.4 ms/img
#   portable: self-contained ONNX + CoreML EP (~21 ms/img)
python scripts/export_onnx.py --ckpt runs/anet/best.pt --bench   # exports + benches all paths
```

Training runs on Apple Silicon MPS; if MPS hits an unsupported op, set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Hyperparameters live in `ANetV1/configs/anet.yaml`.

Dataset generation (from `datasetgen-2026/`):

```bash
.venv/bin/python -m gen2.run --workers 10      # generate/extend (resume-safe)
.venv/bin/python -m gen2.run --preview-only    # labeled preview frames → previews/
```

Generator config: `datasetgen-2026/gen2/config.yaml` (also the ground truth for GSD/object-size math used in the architecture spec).

## Architecture big picture

**ANetV1 model** (`ANetV1/anet/`): a 5-stage pipeline classifying every 10×10 cell of a 960×540 frame as {nothing, mannequin, tent} — region marking, not bounding boxes.

1. **Stage 0** (`data/`): frame → 5,035 overlapping 20×20 windows (stride 10) via `F.unfold`; pixel tokens carry window-relative (u,v) coords, embeddings carry global (x,y).
2. **Stage 1** (`model/encoder.py`): one shared window encoder — dual-quaternion RGB transform, 3 cosine-gated mixing rounds, per-token MLP, cosine-gated pool → 18-d embedding per window. ~96% of FLOPs.
3. **Stage 2** (`model/pyramid.py`): Path A (3/7/11 scalar-kernel local maps, kept per-window and fed to the head) + Path B (gated global pooling → three 256-d states).
4. **Stage 3** (`model/globalmix.py`): multi-cosine mixing over 771 floats — deliberately runs on CPU in fp32 at deploy time.
5. **Stage 4** (`model/head.py`): per-window cosine-gated pooling over 20 tokens → 3-class logits, overlap-averaged into 54×96 cells. Loss applies at cell level.

Training support: `train/` (focal loss + targeted L2/L1 regularization, balanced sampler, object-level metrics), `data/rasterize.py` (YOLO boxes → cell grids; the same rasterizer processes YOLO predictions so all three experiments are comparable), `distill/` (cached YOLO teacher → KL soft labels).

**Hardware constraints shape everything.** The Hailo-8 compiler is conv-centric: no QK attention, no data-dependent normalization, all cosine arguments bounded to one period for int8 LUTs; hence sigmoid-gated pooling instead of softmax/attention, BatchNorm instead of RMSNorm. The deployment-safe forms are used during training from step 0 ("train what you deploy") — do not "simplify" gates back to softmax or add attention. Export uses a 4-phase dense formulation instead of unfold (spec §13, D25).

**Evaluation**: accuracy is never reported (99.9% of cells are background). The decision metric is worst-decile mannequin object-level recall (max-GSD × occluded × grainy slices via gen2 metadata). Per-class cell P/R/F1, object recall, and object FP/image are the reported numbers.

**Dataset caveats** (from `datasets/suas-synth-50k/README.md`): the test split is VisDrone-heavy — filter `vd_*` filenames for synthetic-only eval; VisDrone boxes dominate mannequin counts, so the trainer downweights `vd_*` images (`vd_weight` in anet.yaml).
