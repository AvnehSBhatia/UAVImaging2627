# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

SUAS 2026 UAV target detection: find mannequins and tents in nadir aerial frames (150 ft AGL) and run at ≥30 FPS on a Raspberry Pi 5 + AI HAT+ (Hailo-8). Python via the repo-root venv: `.venv/bin/python` / `.venv/bin/pip` (root `requirements.txt` covers ANetV1 *and* gen2; install PyTorch separately).

Git: repo `main` → https://github.com/AvnehSBhatia/UAVImaging2627.git. Commit convention since 2026-07-07: `ANetV1: <imperative summary>` with a symptom → root cause → fix body. Training runs happen on a remote MI300X box that pulls from this remote — code must be committed/pushed to reach it.

**Project status (2026-07-17, §10 decision executed): the SUAS 2026 flight model is YOLO26n** (`scripts/train_export_yolo26n.py` → ONNX → Hailo DFC). ANetV1 continues as the research track: the §16.2 capacity verdict measured zero train/test generalization gap — the 25k model *underfits its own training data*, so more data/distillation/training cannot close the worst-decile gap. **§19 (D82–D84) is the most recent and most load-bearing correction — read it before proposing any mechanism**: the metric the working loop steered on was empty, §18.5's "≤40k is closed" verdict is withdrawn, and the measured defect is object-appearance sim-to-real.

Three sibling projects:

- **`ANetV1/`** — the custom tiny-detector family (**best proven: v22g (D88, 102,781 params, `runs/anet/v22g_best.pt`) — at 0.5 synthetic fp/img, test mannequin 0.868 / tent 0.959 / worst-quartile 0.772 / worst-decile 0.619, dominating v22's 78k at every operating point and v13's by +0.10 to +0.18 recall, §22.2. Rank checkpoints ONLY with the §21.2 threshold sweep — the per-epoch `mannequin_r` is one threshold on val and systematically under-reports capacity gains (§22.2)**; the prior best, v13 at 25,212 params, is still the small-budget option), a YOLO baseline, and a distillation pipeline. `ARCHITECTURE.md` is the authoritative design record: decisions **D1–D71**; §3–§5 the legacy v6–v8 model, §14 the v9 training-stack rebuild, **§15 the v12/v13 model (current best)**, **§16 the v14–v21 experiment record** including the capacity verdict (§16.2) and the active v20/v21 lines. `OBSERVATIONS.md` records the standalone side probes (P1/P2 — owner-abandoned, code stays runnable). Read the relevant sections before touching model code; changes to model semantics must stay consistent with the record or update it.
- **`datasetgen-2026/gen2/`** — the current synthetic dataset generator (CLIP-gated OpenAerialMap backgrounds, Blender-rendered assets, sun-consistent shadows, Reinhard harmonization, sensor sim). Deterministic per-index seeds, resume-safe. Its living documentation is `datasets/suas-synth-50k/README.md` (the `datasetgen-2026/README.md` describes only the legacy gen1 tool — don't extend gen1).
- **`datasets/`** — generated data (gitignored blobs, ~17 GB; only yaml/README in git). `suas-synth-50k/` is the active YOLO-format dataset; `gen-assets/` holds generator inputs; `VisDrone/` the raw source for the `vd_*` subset.

There is **no pytest/ruff/pre-commit/CI anywhere** — `ANetV1/scripts/smoke_test.py` is the only automated check. Run it after any model change.

## Commands

```bash
.venv/bin/pip install -r ANetV1/requirements.txt
cd ANetV1        # ALWAYS run scripts from inside ANetV1/ — running from the
                 # repo root silently writes a stray runs/ tree at the root

python scripts/smoke_test.py      # param budgets, fwd/bwd, path parity, dataset

# main pipeline
python scripts/train_yolo.py                                          # YOLO baseline + teacher
python scripts/train_anet.py                                          # ANetV1 (default arch v13; ANET_ARCH=v14..v20 opts in)
python scripts/cache_teacher.py --weights runs/yolo/baseline/weights/best.pt
python scripts/train_anet_distill.py                                  # distilled ANetV1

# active research line: v21.x two-stage chunk detector (standalone model + trainer)
python scripts/train_twostage.py              # train (runs/twostage/)
python scripts/train_twostage.py --smoke      # fwd/bwd/detect sanity
python scripts/train_twostage.py --eval runs/twostage/best.pt [--split test]

# evaluation
python scripts/evaluate_all.py --yolo runs/yolo/baseline/weights/best.pt \
  --anet runs/anet/best.pt [--anet-distill ...] [--split train]   # --split train = capacity check
python scripts/benchmark_paper.py --anet runs/anet/best.pt \
  --yolo runs/yolo/yolo26n/weights/best.pt --out runs/paper_bench  # paper-grade: bootstrap CIs, decile curves, latency

# flight model export
python scripts/train_export_yolo26n.py        # YOLO26n → ONNX (Hailo DFC compile is downstream)

# MI300X (remote box): ./run_mi300x.sh (YOLO), ./run_anet_mi300x.sh (ANet)
python scripts/export_onnx.py --ckpt runs/anet/best.pt --bench   # portable ONNX
```

**There is no yaml config.** Hyperparameter defaults live in `ANetV1/anet/train/presets.py` (`anet_cfg()`, device-aware: MI300X vs Mac) and are overridden inline in each train script's cfg block. Everything is env-knob driven (`ANET_*` — ~40 knobs defined in presets.py and the trainers; the load-bearing ones: `ANET_ARCH/BATCH/LR/EPOCHS/COMPILE/INIT_FROM/FREEZE_TRUNK/POS_W/VD`, `DATA_ROOT`). The side probes and the two-stage trainer follow the same "family protocol": `ANET_*` knobs, `pick_device`, gated selection score, `best.pt`/`last.pt` under `runs/<name>/`.

Training runs on MPS locally (`PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is unsupported) and on ROCm remotely. ROCm quirks are load-bearing: dataloader workers must be 0 (spawn deadlocks on fork'd MIOpen mutexes — a background prefetch thread hides the loader cost instead), `TORCHINDUCTOR_COMPILE_THREADS=1` (host-OOM guard), `MIOPEN_FIND_MODE=FAST`, and inductor MISCOMPILES some v15 (pixel_unshuffle) tier shapes to step-0 NaN — v15 defaults `compile=False` (presets), `ANET_COMPILE=1` opts a known-good tier back in. The run scripts set all of this.

Dataset generation (from `datasetgen-2026/`, must be the cwd — package-relative imports):

```bash
.venv/bin/python -m gen2.run --workers 10      # generate/extend (resume-safe)
.venv/bin/python -m gen2.run --preview-only    # labeled preview frames → previews/
```

Generator config: `datasetgen-2026/gen2/config.yaml` — also the ground truth for the GSD/object-size math used in the architecture spec.

## Architecture big picture

**ANetV1 v13** (`ANetV1/anet/`, D57/D58) — the proven best of the family, and `train_anet.py`'s default:

- **Backbone** (`model/backbone.py` `V13Backbone`): a plain multi-scale conv pyramid — stem s2 → dw-sep s4 stage → dw-sep s20 stage with 3 residual blocks → 1×1 head. It replaced the v6–v12 window-token encoder after the pooling-before-features ceiling was measured (~0.05 embedding separation, two full-scale runs pinned at soft-p ≈ 0.09). **Kaiming init is a functional requirement**, not a nicety — DeployNorm seeds from running stats, and default init puts the cascade ~300× off its fixed point (measured: 1e23 logits). Deliberately absent, each for a measured reason: coordinate channels, global-context vector, aux probe.
- **Readout** (D57, since v12): CenterNet-style **center heatmaps** on the 27×48 stride-20 grid — two independent per-class sigmoid heatmaps (no softmax competition), class-agnostic sub-cell (dx,dy) offset, Gaussian targets from `data/rasterize.py`, RetinaNet prior 0.01 init. This replaced per-cell {nothing, mannequin, tent} classification (v6–v11).
- **Loss** (`train/losses.py` `center_focal_loss`): penalty-reduced pixel focal, `pos_weight` 3, ONE smooth term. The older per-cell losses (`focal_norm_loss` D47, the Focal-Tversky/anchor stack) remain for the v8/v9 lineage — the multi-term class-balance stacks are documented oscillation machines; don't reintroduce them.
- **Normalization is DeployNorm** (`model/norm.py`, D39): running-stat affines with detached-EMA updates — the training forward IS the deploy forward. Do not "fix" it back to BatchNorm; batch-stat coupling made training unfusable and OOM-prone. Corollary from the v14 adapter collapse (§16.1): **frozen weights require frozen stats** whenever anything trainable sits upstream (`DeployNorm.frozen`; `ANET_FREEZE_TRUNK=1` handles both).

**Version map** (all in `model/anet.py`; `ANetV1.from_state_dict` shape-sniffs the arch from checkpoint keys, v8→v22): v8/v9 = legacy window-token pipeline (stem `EdgeDQStem4`, `tile_encoder.py`, fused Triton path in `train/fused.py` — ROCm-only, does not apply to v13+ which is pure convs). v14–v19 = valved extensions over v13, mostly falsified (verdicts in §16 — read them before re-proposing similar mechanisms). Active lines: **v20** re-render cycles (owner-directed, §16.8, run pending), **v21.x two-stage** (owner-directed, `model/twostage.py` + `scripts/train_twostage.py`, §16.9 — a separate standalone model whose docstring carries the v21.2–21.5 per-rev history), and **v22** peak-augmented full-rank funnel growth (§17, D72–D75): v13_best grown via a tanh-valved parallel funnel branch (fused `Conv2d(32,64,5,s5)` + max-pool peak path) with a bit-exact full-identity warm start — built, gated, MI300X run pending (`ANET_ARCH=v22 ANET_INIT_FROM=runs/anet/v13_best.pt ANET_BG_W=0`, run-1 isolation per §17.4).

**Hardware constraints shape everything.** The Hailo-8 compiler is conv-centric: no QK attention, no data-dependent normalization at deploy, cosine arguments bounded to one period for int8 LUTs, sigmoid gates instead of softmax, affine-foldable norms only. Deployment-safe forms are used during training from step 0. v13 needs no workarounds (single-shot NPU graph, host-side peak-finding only); the v8/v9 mechanisms (DQ rotations, cosine gates, gated pooling) were Hailo-legality workarounds for attention-like compute.

**Known-good checkpoint:** `ANetV1/runs/anet/good.pt` (v8, mannequin recall 0.573) is force-added into git as the v8-lineage fine-tune base; later archs load it only for evaluation. History lesson encoded in it: a "loss problem" that was actually silent architecture drift — always diff a loaded state_dict against expectations before blaming the loss.

**Evaluation**: accuracy is never reported (99.9% of cells are background). The decision metric is **`mannequin_recall_smallest_decile_synthetic`** — worst-GT-area-decile mannequin object recall *within the synthetic distribution*. **Never read the pooled `mannequin_recall_smallest_decile` (D82, §19.1): 98.5% of test mannequin boxes are VisDrone, the pooled decile is 100% VisDrone at a 13.1 px² cutoff (~3.6×3.6 px vs the synthetic median of 1365 px²), and it reads 0.000 for every model in the family** — an empty metric, not a hard one. Likewise the unsliced `mannequin_recall` is 98.5% VisDrone-weighted; the numbers quoted throughout this file (v13 mannequin 0.835) are the *synthetic* slice. Reported numbers are per-class object recall and FP/image, sliced synthetic vs VisDrone (`train/metrics.py` `CenterObjectMetrics`; `CellConfusion`/`ObjectMetrics` for the per-cell lineage). `best.pt` selects on `mannequin_synth + 0.5·tent` (weight-EMA, D48) — synthetic-only, so unaffected; the two-stage trainer gates that score to −1 above 25 FP/img. The measured open problem is **object-appearance sim-to-real** (§19.3): on real web scenes v13's objects respond 15% weaker *and* background fires 2.6× more often than on synthetic, and since gen2 composites rendered objects onto *real* aerial backgrounds, the gap is the objects — a dataset/compositing problem before an architecture one.

**Dataset caveats** (`datasets/suas-synth-50k/README.md`): test split is VisDrone-heavy (1,267 vd vs 449 synthetic) — filter `vd_*` filenames for synthetic-only eval. VisDrone frames keep native mixed resolutions (everything is letterboxed to 960×540 by `anet/data/`); synthetic frames are uniformly 1920×1080. VisDrone boxes dominate mannequin counts (both raw classes 0 and 1 remap to mannequin), so the trainer downweights `vd_*` images (`vd_weight`). Instances <25% visible are unlabeled; ~9.5% of synthetic frames are deliberately background-only. Underscore-prefixed dirs in `gen-assets/` (`_rejected/`, `_removed/`, `_oblique/`) are excluded staging pools, not usable assets.
