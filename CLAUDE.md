# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

SUAS 2026 UAV target detection: find mannequins and tents in nadir aerial frames (150 ft AGL) and run at ≥30 FPS on a Raspberry Pi 5 + AI HAT+ (Hailo-8). Python via the repo-root venv: `.venv/bin/python` / `.venv/bin/pip` (root `requirements.txt` covers ANetV1 *and* gen2; install PyTorch separately).

Git: repo `main` → https://github.com/AvnehSBhatia/UAVImaging2627.git. Commit convention since 2026-07-07: `ANetV1: <imperative summary>` with a symptom → root cause → fix body. Training runs happen on a remote MI300X box that pulls from this remote — code must be committed/pushed to reach it.

Three sibling projects:

- **`ANetV1/`** — the custom ~21k-parameter per-cell detector (v9), a YOLO baseline, and a distillation pipeline. `ARCHITECTURE.md` is the authoritative design record: design decisions **D1–D48**, §3–§5 describe the v6–v8 model, **§14 is the current v9 delta** (`V9_CHANGES.md` is the quick summary). Read it before touching model code; changes to model semantics must stay consistent with it or update it.
- **`datasetgen-2026/gen2/`** — the current synthetic dataset generator (CLIP-gated OpenAerialMap backgrounds, Blender-rendered assets, sun-consistent shadows, Reinhard harmonization, sensor sim). Deterministic per-index seeds, resume-safe. Its living documentation is `datasets/suas-synth-50k/README.md` (the `datasetgen-2026/README.md` describes only the legacy gen1 tool — don't extend gen1).
- **`datasets/`** — generated data (gitignored blobs, ~17 GB; only yaml/README in git). `suas-synth-50k/` is the active YOLO-format dataset; `gen-assets/` holds generator inputs; `VisDrone/` the raw source for the `vd_*` subset.

There is **no pytest/ruff/pre-commit/CI anywhere** — `ANetV1/scripts/smoke_test.py` is the only automated check. Run it after any model change.

## Commands

```bash
.venv/bin/pip install -r ANetV1/requirements.txt
cd ANetV1        # ALWAYS run scripts from inside ANetV1/ — running from the
                 # repo root silently writes a stray runs/ tree at the root

python scripts/smoke_test.py      # param budgets, fwd/bwd, v9 path parity, dataset

# training pipeline, in order
python scripts/train_yolo.py                                          # 1. YOLO baseline + teacher
python scripts/train_anet.py                                          # 2. ANetV1 v9 from scratch
python scripts/cache_teacher.py --weights runs/yolo/baseline/weights/best.pt
python scripts/train_anet_distill.py                                  # 3. distilled ANetV1

# three-way comparison on the test split (identical cell metrics)
python scripts/evaluate_all.py \
  --yolo runs/yolo/baseline/weights/best.pt \
  --anet runs/anet/best.pt \
  --anet-distill runs/anet_distill/best.pt

# MI300X (remote box): ./run_mi300x.sh (YOLO), ./run_anet_mi300x.sh (ANet v9)
# fast local inference: anet.metal.MetalANet.from_checkpoint (v8 ckpts, ~7.4 ms/img)
python scripts/export_onnx.py --ckpt runs/anet/best.pt --bench   # portable ONNX (~21 ms/img)
```

**There is no yaml config.** Hyperparameter defaults live in `ANetV1/anet/train/presets.py` (`anet_cfg()`, device-aware: MI300X vs Mac) and are overridden inline in each train script's cfg block. Env knobs: `ANET_BATCH/ACCUM/LR/EPOCHS/COMPILE/FUSED/FUSED_BWD/CACHE/CONF/INIT_FROM/LOSS_MODE`, `DATA_ROOT`.

Training runs on MPS locally (`PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is unsupported) and on ROCm remotely. ROCm quirks are load-bearing: dataloader workers must be 0 (spawn deadlocks on fork'd MIOpen mutexes — a background prefetch thread hides the loader cost instead), `TORCHINDUCTOR_COMPILE_THREADS=1` (host-OOM guard), `MIOPEN_FIND_MODE=FAST`, and inductor MISCOMPILES some v15 (pixel_unshuffle) tier shapes to step-0 NaN — v15 defaults `compile=False` (presets), `ANET_COMPILE=1` opts a known-good tier back in. The run scripts set all of this.

Dataset generation (from `datasetgen-2026/`, must be the cwd — package-relative imports):

```bash
.venv/bin/python -m gen2.run --workers 10      # generate/extend (resume-safe)
.venv/bin/python -m gen2.run --preview-only    # labeled preview frames → previews/
```

Generator config: `datasetgen-2026/gen2/config.yaml` — also the ground truth for the GSD/object-size math used in the architecture spec.

## Architecture big picture

**ANetV1 v9** (`ANetV1/anet/`): a pipeline classifying every 10×10 cell of a 960×540 frame as {nothing, mannequin, tent} — region marking, not boxes.

1. **Stem** (`model/blocks.py` `EdgeDQStem4`): raw colour + 4 oriented Sobel-init 7×7 edge branches behind dual-quaternion colour rotations → 15 channels.
2. **Stage 1** (`model/tile_encoder.py`): one shared 20×20-window encoder (stride 10 → 53×95 windows) — 3 cosine-gated mixing rounds, fc1+gated pool per token, fc2 per window → 32-d embeddings. ~96% of FLOPs. Three equivalent implementations, parity-asserted: windowed tokens (reference), dense 4-phase (PyTorch, any device), and **fused Triton kernels** for training on ROCm/CUDA (`train/fused.py` — startup parity checks demote automatically: triton bwd → chunked-autograd bwd → dense).
3. **Neck/context** (`model/neck.py`, `model/pyramid.py`, `model/context.py`): residual depthwise conv neck on the embedding grid, Path A k3/7/11 per-channel maps, SlimContext (gated global pools → multi-cosine weave → one context vector).
4. **Head** (`model/head.py` `RegionHeadV9`): split local/context streams → Linear(68→24)→SiLU→Tanh→Linear(24→3), overlap-averaged into 54×96 cells. A train-only aux probe (1×1 conv on the embedding map) gives the encoder a direct gradient path.

**Normalization is DeployNorm** (`model/norm.py`, D39): running-stat affines with detached-EMA updates — the training forward IS the deploy forward. Do not "fix" it back to BatchNorm; batch-stat coupling is what made training unfusable and OOM-prone (MIOpen int32 overflow at batch ≥ 44).

**Loss** (`train/losses.py` `focal_norm_loss`, D47): per-class positive-normalized focal — ONE smooth term. The Focal-Tversky/anchor stack (kept for v8 ablation) is a documented oscillation machine; don't reintroduce multi-term class-balance tugs-of-war.

**Hardware constraints shape everything.** The Hailo-8 compiler is conv-centric: no QK attention, no data-dependent normalization at deploy, all cosine arguments bounded to one period for int8 LUTs; sigmoid-gated pooling instead of softmax, affine-foldable norms only. Deployment-safe forms are used during training from step 0 — do not simplify gates to softmax or add attention. The v8 code paths remain behind `arch="v8"`; `ANetV1.from_state_dict` shape-sniffs v8 vs v9 checkpoints.

**Known-good checkpoint:** `ANetV1/runs/anet/good.pt` (v8, mannequin recall 0.573) is force-added into git as the fine-tune base for the v8 lineage. v9 cannot warm-start from it (different encoder); it still loads for evaluation. History lesson encoded in it: a "loss problem" that was actually silent architecture drift — always diff a loaded state_dict against expectations before blaming the loss.

**Evaluation**: accuracy is never reported (99.9% of cells are background). The decision metric is worst-decile mannequin object-level recall; per-class cell P/R/F1, object recall, and object FP/image are the reported numbers. The go/no-go bar vs YOLO26n is in ARCHITECTURE.md §10. `best.pt` selects on `mannequin_synth + 0.5·tent` and stores the weight-EMA model (D48).

**Dataset caveats** (`datasets/suas-synth-50k/README.md`): test split is VisDrone-heavy (1,267 vd vs 449 synthetic) — filter `vd_*` filenames for synthetic-only eval. VisDrone frames keep native mixed resolutions (everything is letterboxed to 960×540 by `anet/data/`); synthetic frames are uniformly 1920×1080. VisDrone boxes dominate mannequin counts (both raw classes 0 and 1 remap to mannequin), so the trainer downweights `vd_*` images (`vd_weight`). Instances <25% visible are unlabeled; ~9.5% of synthetic frames are deliberately background-only. Underscore-prefixed dirs in `gen-assets/` (`_rejected/`, `_removed/`, `_oblique/`) are excluded staging pools, not usable assets.
