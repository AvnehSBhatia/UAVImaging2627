# Observations — side probes

Standalone micro-experiments that live NEXT TO ANetV1 (same folders, same
train protocol: `ANET_*` env knobs, device pick, best.pt on a gated
selection score) but are not part of the detector. ARCHITECTURE.md records
ANet design decisions; this file records what the probes measure.

Motivating hypothesis (owner, 2026-07-17): *"the biggest issue is the conv
blocks"* — i.e. the v13 trunk's `_DWSep` stacks may be doing work that
simpler, more legible machinery could do. Each probe isolates one slice of
that machinery and measures it in isolation on object-centred crops
(mannequin → 40×40, tent → 100×100, synthetic-only by default; `ANET_VD=1`
opts VisDrone in).

Shared data plumbing: `anet/probes/patches.py` (`PatchCrops`) — one decode
per image, all object crops + 2 deterministic background crops per size,
masks = union of GT box rects (white) on black.

## P1 — WhiteboxDQ: is colour algebra alone separable? (PENDING)

`anet/probes/whitebox.py`, trainer `scripts/train_whitebox.py`,
checkpoints `runs/whitebox/`.

- **Spec** (owner): learn a series of dual-quaternion transforms +
  non-linear thresholding/activations until GT boxes render white and
  background pure black.
- **Model**: 4 × (DualQuaternionRGB → LearnedAct) → 1×1 conv → per-pixel
  logit. ~120 params, strictly pointwise — zero spatial context by design.
  Reuses D5 (DQ folds to 1×1 conv) and D69-B (bounded LearnedAct = one
  Hailo LUT).
- **Measures**: how much of the separation the conv blocks provide is
  actually colour-space separation. Metric: mean IoU@0.5 on object crops,
  white-fraction on bg crops; sel = IoU − bg_white (over-painter guard).
- **Reading the result**: high IoU at near-zero bg_white ⇒ colour alone
  carries the classes and the trunk's spatial capacity is being spent on
  something else (or wasted). Low IoU ⇒ the conv blocks' texture/shape
  work is load-bearing, and P1 bounds how much of it colour can replace.
- **Result**: _pending first run._

## P2 — FiveStack: hand-picked filter bank + 10×10 window scorer (PENDING)

`anet/probes/fivestack.py`, trainer `scripts/train_fivestack.py`,
checkpoints `runs/fivestack/`.

- **Spec** (owner): duplicate the N×N patch into 5 images — 2 Sobel
  filters, 1 learned DQ, 1 Gaussian blur (10×10), 1 learned 10×10
  texture-removal kernel — then an embedding-space transform over the
  stack, scan with a 10×10 kernel emitting per-window class logits, sum
  the windows into the patch class.
- **Model**: fixed Sobel-x/y + fixed Gaussian(10,σ3) + learned 10×10 conv
  (blur-init) + learned DQ, all collapsed to 1ch → 1×1 conv 5→16 + SiLU →
  10×10/s10 conv → 3 logits per window → position-blind, window-count-
  normalized sum (a raw sum gives 40 vs 100 patches different logit
  temperatures under the shared softmax). ~5.1k params.
- **Measures**: whether filter DIVERSITY chosen by hand (edges / colour /
  low-pass / anti-texture) plus a tiny local scorer matches what the
  learned conv pyramid extracts. The sum is position-blind, so only local
  10×10 evidence can vote — same information regime as ANet's early
  stages.
- **Reading the result**: patch recalls near v13's cell-level recall ⇒
  the trunk's first stages are replaceable by 5 fixed-ish filters (big
  simplification lever, and a Hailo-cheap one). Well below ⇒ the learned
  blocks earn their keep before stride 4.
- **Result**: _pending first run._

## Protocol notes

- Both probes select on gated scores (bg-white penalty / bg-acc ≥ 0.8
  gate) — the max_sel_fp lesson applied: an over-predictor must never win
  the checkpoint.
- Background crops are rejection-sampled off every GT rect with a
  per-image fixed seed: identical crops every epoch, every eval, every
  machine.
- `--eval CKPT --split test` on either trainer gives the final-numbers
  pass; val (capped at 800 images) drives checkpoint selection only.
- **YOLO reference bar**: `scripts/benchmark_probes.py` runs YOLO26n
  full-frame and maps its boxes into the IDENTICAL crops — its boxes are
  rasterized through the same `rect_mask` geometry (P1's game) and its
  best-overlap box classifies each patch (P2's game), so all three columns
  are apples-to-apples. Dry run (12 test frames): YOLO mask_iou 0.934,
  bg_white 0.000, mann_r 0.929, tent_r 1.000 at 2.38M params — the bar a
  52-param colour probe / 5k-param filter bank is measured against.

  ```
  python scripts/benchmark_probes.py \
      --whitebox runs/whitebox/best.pt --fivestack runs/fivestack/best.pt \
      --yolo runs/yolo/yolo26n/weights/best.pt --latency
  ```
