# ANetV1 v9 — what changed, why, and what to expect

Companion to `ARCHITECTURE.md` §14 and D39–D48. This is the summary-of-changes
deliverable; the spec has the full rationale per decision.

## The three failures v9 fixes

**1. Recall collapse from scratch (mannequin AND tent → 0.000 by epoch 3).**
Root causes found in the code, not hypothesized:
- The Focal-Tversky loss stack is structurally oscillatory: set-level ratios
  make per-cell gradients depend nonlinearly on batch TP/FP totals, and the
  FT + focal-anchor pair is a documented tug-of-war. In your failing run it
  actively pushed tent soft-prob from its 0.1 prior init *down* to 0.003.
- The stem's "Sobel-7 init" (D33) was actually `weight.mul_(0.2)` on random
  kaiming noise — the model started with **no oriented-edge structure at
  all**, and mannequins differ from clutter by shape/texture, not colour.
- The decision path was starved: 60–84% of all parameters sat in three
  18→256 context expansions feeding one per-frame vector, while the per-cell
  classifier squeezed through an 8-d Tanh choke and the only cross-window
  context was fixed box averages.

**2. 517 s epochs at ~1% GPU utilization.** The dense path ran the per-token
MLP + BatchNorm + gate at full 540×960 resolution × 4 phases (2M positions),
and training-mode BN's batch statistics coupled every tile to every other
tile — unfusable by construction, thousands of launches per step.

**3. ~120 GB VRAM at batch 96.** MIOpen's BatchNorm kernel overflows int32
above batch ~44 on these shapes; the code silently fell back to a primitive
path that materializes fp32 copies of ~19 GB tensors, several times per step.

## The changes (D-numbers in ARCHITECTURE.md)

| # | Change | Effect |
|---|---|---|
| D39 | **DeployNorm** — normalize with running-stat affines (the deploy form), stats updated as detached EMA | encoder becomes tile-local (fusable); kills the MIOpen int32 / fp32-promotion bug class; no BN train/eval gap at all |
| D40 | **Fused Triton Stage-1** — whole per-token encoder in one kernel per direction, backward recomputes in registers | the 30 s/epoch lever; activations ~6–9 GB at batch 96; parity-checked at startup with automatic demotion (triton→chunked-autograd→dense) |
| D41 | **Sobel-init 4-orientation stem** (0/90/45/135°) | real edge detectors from step 0, no orientation gap for arbitrary-yaw mannequins |
| D42 | **fc2 moved after the pool** (per-window, not per-pixel) | −45% full-res FLOPs, largest activation tensor gone |
| D43 | **ConvNeck** — 2 residual dw5×5+pw rounds on the 53×95 embedding grid | trainable 50–110 px cross-window context (the "head can't resolve scale" fix) |
| D44 | **SlimContext** — Path-B 18→256 expansions deleted; d-width states + the multi-cosine weave kept | frees 14.6k params for per-region discrimination; keeps the signature CPU stage |
| D45 | **Head widened 8→24** (SiLU→Tanh kept) | the per-cell decision no longer dies in an 8-d choke |
| D46 | **Aux probe** — train-only 1×1 conv on the embedding map, weight 0.3 | direct encoder gradient a collapsed head can't block; 105 params, dropped at export |
| D47 | **focal_norm loss** — per-class positive-normalized focal (CenterNet-style), one term | size-invariant AND smooth; no limit cycles; recall-first dynamics at init (~40:1 fg:bg pull) |
| D48 | **Weight EMA** (0.998) evaluated + checkpointed | selection metric stops jittering; best.pt is a smoothed model |

Model: **20,706 deployed params** (+105 train-only aux) — under the 40k
budget. Every mechanism that defines the project (dual-quaternion colour
transforms, gaussian-lens cosine gates, sigmoid-gated pooling, multi-cosine
weave, cell overlap-averaging) is intact; all ops remain in the Hailo-legal
set (depthwise/1×1 convs, folded affines, bounded-cos LUT forms).

Training-support changes: DeployNorm stat seeding (8 no-grad batches before
step 0), background-thread batch prefetcher (hides the ~150 MB/step loader
cost the in-process ROCm loader was serializing), startup parity checks with
loud, layered fallback, prior-bias init p=0.05, cosine LR 3e-3 + 300-step
warmup, grad clip 10.0 (measured focal_norm grad norms 25–180), D24
regularizers rescaled to 3e-3 (matching the ~30× larger loss scale),
foreground-class loss floor 1 / background floor 8, 40 epochs.

## Verified locally (Apple Silicon, before you run on MI300X)

- Param count/budget, train fwd+bwd, every parameter receives grad.
- Dense path ≡ windowed token path: max delta **3.3e-16** (fp64).
- Chunked-backward mirror ≡ dense path forward: **~1e-9**; chunked backward ≡
  full autograd: **~1e-4** (the fp32 accumulation-order noise floor — two
  orderings of the same sums differ by 3e-4). This mirror is the same math
  the Triton kernels implement and the reference the Triton backward is
  parity-checked against on your box at startup.
- Overfit runs on real frames: loss falls, foreground probs rise from the
  0.05 prior (the v8 failure actively pushed them DOWN to 0.003), zero FP.
  Full per-cell discrimination needs full-data epochs, not a 4-frame toy —
  per image-presentation v9 moves ~40× faster than the failing v8 run did.

**A 10-agent adversarial review** then audited every new file (both Triton
kernels were independently re-derived: the forward was transcribed to PyTorch
and matched to 1e-7, the backward chain rule matched autograd to 1e-16 on a
small instance). It found and I fixed: a bf16-autocast downcast of two derived
bias tensors on the default fused path (invisible to the autocast-free parity
check), the fallback dataloader rebuild dropping the ROCm spawn context
(fork-deadlock class), chunked-backward VRAM ~26 GB at batch 96 (now
batch-sub-chunked to ~8-10 GB) plus an empty-band crash, a double stat
observation under gradient checkpointing, EMA shadowing stale norm buffers
(now parameters-only + cold-start debias ramp), the v8/distill path's fp32
promotion under bf16, a foreground-floor inversion in the loss for 1-4-cell
objects, ~30× diluted regularizers, a blocking per-step CPU sync, a missing
try/finally on the EMA swap, a prefetcher thread leak, and the dropped
`runs/.stages/anet.done` marker. All fixes re-verified green.

What could NOT be verified here: executing the Triton kernels (no ROCm on
this machine). That is exactly why startup runs forward and backward parity
checks on real frames and demotes automatically — you cannot silently train
on a broken kernel. Watch for the `fused Stage-1 ON (bwd=triton ...)` line.

## Expected improvements (realistic, not promised)

**Throughput (19,185-image epochs, MI300X):**

| path | epoch (train) | eval | VRAM @96 |
|---|---|---|---|
| fused, triton bwd | **~15–45 s** | ~5–15 s | ~6–9 GB |
| fused, chunked bwd | ~45–120 s | ~5–15 s | ~8–12 GB |
| dense fallback (batch 32) | ~3–6 min | ~30–60 s | ~12–16 GB |

First epoch adds one-time costs (memmap cache ~10 min if cold; triton/
inductor compile 1–5 min, disk-cached afterwards). The 30 s/epoch target is
inside the fused range but at its optimistic end — if the backward kernel's
register pressure forces spills on gfx942 it will land slower; the parity
harness still gives you a correct 45–120 s fallback the same day.

**Detection (val, conf 0.5):**
- Tent object recall: 0.85–0.95 — tents are large and were detectable even
  under the old stack; the neck + wide head mainly firm up boundaries.
- Mannequin object recall (synthetic): **0.65–0.85** expected band. The prior
  best from-scratch-lineage number was 0.573 (good.pt, after fine-tune
  rescue); real edge init + aux supervision + stable loss + 24-wide head each
  attack a separately-diagnosed cause of the gap.
- Object FP/image: ≤ 0.5 at conf 0.5, tunable down with `ANET_CONF`. A "0.1%
  of cells" budget is ~5 FP cells/image — expect to be well under it.

**On the 99% / 0.1% guarantee you asked for — no honest engineer can promise
that, and I won't.** The dataset deliberately contains mannequins occluded to
~6–7 px blurred fragments at 540p (ARCHITECTURE §1.2/D22 calls them
"physically invisible" in the worst decile); a 20k-param model will not find
what a 2.5M-param COCO-pretrained YOLO also misses. What IS realistic: the
spec's own decision bar (§10 — within ~5 points of the YOLO26n teacher on
worst-decile mannequin recall), 99%+ on *tents*, and 99%-class recall on
*unoccluded, typical-GSD* mannequins. If overall mannequin recall must go
higher, the pre-registered levers are the 1080p fine-tune (§8 risk 4 — warm
start, resolution-independent weights, 2× inference cost) and multi-look
fusion across the 30 FPS stream at mission level — not more epochs at 540p.

## How to run

```bash
cd ANetV1 && ./run_anet_mi300x.sh          # everything auto: seeding, parity, train
# watch: tail -f logs/anet.log
# knobs: ANET_FUSED=0 | ANET_FUSED_BWD=chunked | ANET_BATCH=64 | ANET_CONF=0.6
python scripts/smoke_test.py               # run once on the box first (30 s)
```
