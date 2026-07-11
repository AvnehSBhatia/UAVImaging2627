# ANetV1 — Full Architecture Specification & Design Record

**Status:** v9 (training-stack rebuild) · 2026-07-10 — v6 was the locked baseline spec; v7/v8 (D31–D38) fixed the recall collapse and MI300X throughput; v9 (D39–D48) rebuilds the training path (fused Triton Stage 1, DeployNorm, focal_norm loss) and rebalances the parameter budget (SlimContext, ConvNeck, wide head). §3–§5 describe the v6–v8 model; §14 is the v9 delta.
**Task:** per-region classification of UAV survey frames into {mannequin, tent, nothing} (SUAS-style search area)
**Deployment target:** Raspberry Pi 5 + AI HAT+ (Hailo-8, 26 TOPS int8) at ≥30 FPS
**Training target:** Apple Silicon (MPS), PyTorch
**Headline figures:** v9 default: 20,706 deployed parameters (~83 KB fp32 / ~21 KB int8) · ~3.5 GFLOPs/frame @ 960×540 (v6 locked spec was 17,037 params / ~2.5 GFLOPs) · est. 150–300 FPS on the HAT

---

## Table of contents

1. [Problem statement and constraints](#1-problem-statement-and-constraints)
2. [Design philosophy](#2-design-philosophy)
3. [Pipeline overview](#3-pipeline-overview)
4. [Stage-by-stage specification](#4-stage-by-stage-specification)
5. [Parameter budget](#5-parameter-budget)
6. [Compute, memory, and deployment analysis](#6-compute-memory-and-deployment-analysis)
7. [Design decision log](#7-design-decision-log)
8. [Known risks and pre-registered mitigations](#8-known-risks-and-pre-registered-mitigations)
9. [Ablation plan](#9-ablation-plan)
10. [Baseline comparison: YOLO11n / YOLO26n](#10-baseline-comparison-yolo11n--yolo26n)
11. [Training configuration](#11-training-configuration)
12. [Design version history](#12-design-version-history)
13. [Implementation notes](#13-implementation-notes)

---

## 1. Problem statement and constraints

### 1.1 Task

Mark every 20×20-pixel region (at capture resolution) of a 1920×1080 UAV frame as **mannequin**, **tent**, or **nothing**. This is region marking, not bounding-box regression — the mission needs "target is in this cell," not tight boxes.

### 1.2 Dataset facts (from `datasetgen-2026/gen2/config.yaml`)

These numbers drove several architecture decisions and are the ground truth for all pixel-size math:

| Property | Value |
|---|---|
| Frames | 50,000 synthetic, 1920×1080, 90/7/3 split |
| GSD | 0.012–0.025 m/px target (150 ft AGL), ±10% altitude jitter, crop-zoom up to 1.3× → effective range ≈ 0.009–0.028 m/px |
| Mannequin | body length 1.60–1.92 m → **57–208 px long / ~16–37 px wide at 1080p**; typical ≈ 98×25 px |
| Tent | larger (multi-meter footprint); never the limiting class |
| Occlusion | probability 0.55, coverage 0.10–0.55, labels kept down to 18% visibility and 9 px bbox → worst visible fragments ≈ 13×13 px at 1080p |
| Sensor model | grainy tier: motion blur up to 9 px, defocus up to 2.2 px, JPEG q≥42, Bayer roundtrip |
| Adversarial content | clutter deliberately scattered near objects; Reinhard color harmonization suppresses color shortcuts |
| Class balance | ~0–2 mannequins, 0–2 tents per frame; 8% background-only |

At the chosen **960×540** input: typical mannequin ≈ 49×13 px, worst case ≈ 29×8 px, worst occluded fragment ≈ 6–7 px.

### 1.3 Deployment constraints (Hailo-8, learned the hard way)

- **FLOPs are irrelevant.** 26 TOPS int8 vs our need of 0.075 TOPS at 30 FPS (0.3% of peak). The binding constraints are elsewhere.
- **Op support is everything.** The Hailo Dataflow Compiler (DFC) is conv-centric. Activation×activation matmuls (i.e., QK attention), data-dependent normalization (softmax over space, RMSNorm), and arbitrary elementwise functions are unsupported or fragile. Unsupported layers fall back to the Pi's CPU — fatal for any *dense* stage at 30 FPS.
- **int8 everywhere on-chip.** Unbounded-frequency cosines quantize to garbage; every cos argument must be bounded to ~one period so an 8-bit LUT is accurate.
- **PCIe Gen3 x1 ≈ 1 GB/s host↔HAT.** Pre-unfolded window tensors would need 1.2 GB/s; raw frames need 47 MB/s. The 4× window-overlap duplication must happen on-chip.
- **DRAM-free chip.** Small models with small activations are structurally favored — a 17 KB model is an ideal fit.

---

## 2. Design philosophy

1. **Aggressive weight sharing.** One window encoder for all 5,035 windows; one gate mechanism reused at every pooling site; one set of pyramid kernels. Total 17k parameters.
2. **Keep the signature mechanisms** (dual quaternion color transform, cosine gating, multi-cosine global mixing) — they are the point of the project — but place them where hardware can afford them: dense exotic ops become LUT-safe bounded forms; tiny-tensor exotic ops run on CPU in exact fp32.
3. **Match the mechanism to its consumer.** Anything feeding a global crush must be cheap (gated pooling, not attention). Anything feeding per-region decisions keeps full resolution (Path A).
4. **Positional information is explicit, not architectural.** Coordinates ride as token features at both levels; every pooling/attention-like op is otherwise permutation-invariant.
5. **Train what you deploy.** The deployment-safe forms (sigmoid gates, BatchNorm, bounded cos) are used during training, not swapped in afterward.

---

## 3. Pipeline overview

```
960×540×3 frame  (bilinear downscale of 1920×1080 capture)
│
├─ STAGE 0 · stem + windowing:
│     EdgeDQ stem (D33, default): raw ∥ DQ→7×7 vert-Sobel ∥ DQ→7×7 horiz-Sobel,
│       each group re-framed by its own dual-quaternion transform → 9 channels
│       (ablation variant "highpass", D32: DQ-RGB + depthwise 3×3 high-pass → 6 ch)
│     → 20×20 windows, stride 10 → 53 rows × 95 cols = 5,035 windows
│     pixel tokens: (r, g, b, e₁…e₆, u, v)   u,v = window-relative coords ∈ (0,1)
│
├─ STAGE 1 · window encoder (shared weights, 5,035×):
│     3 × mixing round:
│         BN → 3 shared dim-11 dots → Gaussian-blur s1,s2 maps (learned σ)
│         → score = ⟨s1⟩·cos(π·tanh(⟨s2⟩·s3) + φ) → sigmoid gate
│         → gated window mean added to RGB channels → SiLU (edges + coords frozen)
│     → per-token MLP 11→16→16 (SiLU) → BN → cosine-gated pool → 16-d
│     → concat global window-center (x,y) → 18-d embedding
│
├─ STAGE 2 · embedding map (B,18,53,95):
│     Path A (local, full res): 3×3 / 7×7 / 11×11 scalar-kernel maps → kept per-window
│     Path B (global): per level: BN → 1×1-conv scorer 18→8→4→1 → sigmoid ⊙ map
│                       → global avg pool → Linear 18→256 → three 256-d states
│
├─ STAGE 3 · global mixing (CPU, exact fp32, 771 floats):
│     per state i: (s1ᵢ,s2ᵢ,s3ᵢ) = Uₖ·vᵢ
│     wᵢ = Σⱼ s1ⱼ·cos(π·tanh(s2ⱼ·s3ᵢ) + φ)   → softmax(3) → mixed 256-d
│     → split into 16 tokens × 16-d → pad with learned 2-vector → (16,18)
│
└─ STAGE 4 · per-window head (5,035×), two pooled streams (D31):
      local stream: own embedding + own 3 Path-A vectors (4 × 18-d)
        → BN → cosine gate → sigmoid-weighted mean → 18-d
      context stream: 16 global tokens (once per frame)
        → BN → cosine gate → sigmoid-weighted mean → 18-d
      → concat (36-d) → Linear 36→8 → SiLU → Tanh → Linear 8→3
      → overlap-average window logits into 10×10 cells (54×96 grid) → argmax
```

Output granularity: 10×10 px cells at 540p = 20×20 px at capture = 0.24–0.56 m ground.

---

## 4. Stage-by-stage specification

### 4.0 Stage 0 — windowing and tokens

- **Stem (before windowing):** the EdgeDQ stem (D33, default) produces a 9-channel full-frame feature map: colour + 3 vertical-edge + 3 horizontal-edge channels (7×7 oriented Sobel init, learnable, bracketed by dual-quaternion colour transforms). The `highpass` ablation variant (D32) produces 6 channels. Everything is a plain conv at export; because the stem runs on the full frame, the windowed and 4-phase dense paths stay bit-identical.
- `F.unfold(feat, kernel=20, stride=10)` → `(B, 9·400, 5035)`. Grid: `(540−20)/10+1 = 53` rows, `(960−20)/10+1 = 95` cols. Both axes tile the frame exactly (20 + 52·10 = 540; 20 + 94·10 = 960).
- Token order inside a window is row-major `(row·20 + col)`; windows are ordered `(row·95 + col)`. The coordinate buffers are constructed to match this exactly.
- **Window-relative coords** `u = (col+0.5)/20`, `v = (row+0.5)/20` are concatenated to every pixel token (dim 9→11). Like the edge channels, they are *frozen channels*: no round update, no activation ever touches them.
- **Global coords** `x = (10·i+10)/960`, `y = (10·j+10)/540` (window centers ≡ mean of the center-4 pixels) are concatenated to embeddings (dim 16→18).

**Why two coordinate frames:** global coords vary by only ~0.001 between adjacent pixels inside a window — numerically invisible after normalization, so within-window shape needs window-relative coords. Global position (horizon bias, runway location) belongs at the embedding level where 0.01-scale steps are meaningful.

### 4.1 Stage 1 — window encoder (1,050 params incl. stem)

**Dual quaternion RGB transform (8 params each).** Real part `q_r` (4) → rotation matrix R ∈ SO(3); dual part `q_d` (4) → translation `t = 2·vec(q_d ⊗ q_r*)`. Output: `R·rgb + t`. A *rigid* transform of color space: rotation + offset, norm-preserving, 6–7 effective DOF. At export this is a constant 3×3 conv + bias — zero deploy risk, zero extra inference cost. The stem uses five instances.

**EdgeDQ stem (334 params, D33).** Triplicate the frame: one copy stays raw colour; the other two pass through a learned DQ colour rotation then a learnable 7×7 oriented edge conv (depthwise, Sobel-7 init — vertical and horizontal). Each 3-channel group is then re-framed by its own DQ → 9 channels. Frozen through the rounds like (u,v) (only the first 3 colour channels get the residual update). Rationale: without it the tokens are single-pixel colours and Stage 1 has no edge/texture operator at all — measured mannequin-vs-clutter separability in the trained v6 embeddings was barely above chance (linear-probe lift ×1.4 vs tent ×2.0) while mannequins differ from clutter by shape/texture, not colour. The `highpass` variant (D32: DQ-RGB + depthwise 3×3 zero-DC high-pass, 35 params, 6 channels) is kept as the minimal-stem ablation.

**Mixing rounds ×3 (35 params each).** Round r on tokens `x ∈ (400, 11)`:

1. `x̂ = BN(x)` (per-channel BatchNorm; folds into the following dot products at export).
2. Three shared dot products: `s_k = x̂ · V_k`, `V ∈ ℝ^{3×11}` → maps `s1, s2, s3 ∈ ℝ^{400}`.
3. Gaussian blur of `s1` and `s2` over the 20×20 token grid: 9×9 kernel built from learned σ (`σ = softplus(σ_raw) + 0.5`, init ≈ 3.1 px). ⟨s1⟩, ⟨s2⟩ are neighborhood amplitude/frequency fields; `s3` stays per-token.
4. `score = ⟨s1⟩ · cos(π·tanh(⟨s2⟩·s3) + φ)`, φ learned per round, init π/2.
5. `gate = sigmoid(score)`; `pooled = mean(gate ⊙ x)` over 400 tokens (unnormalized gated mean — no softmax).
6. `rgb ← SiLU(rgb + pooled_rgb)`; u,v unchanged.

Semantics: a smooth local field defines the lens `a·cos(b·x+φ)`; each token's own s3 is read through its neighborhood's lens — tokens that deviate from their surroundings score differently from tokens that conform (emergent edge/blob detection). σ interpolates between per-token gating (σ→0) and whole-window-context gating (σ→∞).

**Per-token MLP (464 params).** `Linear(11,16) → SiLU → Linear(16,16) → SiLU`. This plus the pool is a 300:1 compression (4,800 capture-res values → 16 floats) — the model's tightest capacity point.

**Cosine-gated pool (49 params + BN 32).** `CosineGate(16)`: 3 shared 16-d dots → `score = s1·cos(π·tanh(s2·s3)+φ)` → sigmoid → gated mean over 400 tokens → 16-d embedding. BN(16) before scoring.

### 4.2 Stage 2 — pyramid (15,458 params)

**Path A — multi-scale local maps (179 params shared / ~4.5k per-channel).** For k ∈ {3, 7, 11}: a k×k neighborhood map. Two forms (D13 / D37): the *shared-scalar* spec form is **one weight per position, shared across all 18 channels** (init 1/k², depthwise conv with an expanded single-channel kernel — 179 params); the *per-channel* form (default, D37) gives each channel its own k×k kernel, box-filter-initialised so it starts bit-identical to the shared form and specialises from there (~4.5k params at d=26). Same-resolution output (padding k//2). Context spans at 540p: 40 / 80 / 120 px ≈ 0.7–3.4 m ground — 3×3 ≈ mannequin torso + surround, 11×11 ≈ full tent + surround. These maps are **kept per-location** and fed to the head (see D14), each re-framed by a learned per-scale 1×1 conv (D36) before Path B and the head consume it.

**Path B — gated global pooling, per level (5,093 params each).** `BN2d(18) → Conv1×1 18→8 → SiLU → Conv1×1 8→4 → SiLU → Conv1×1 4→1 → sigmoid` produces a relevance gate; `pooled = mean(gate ⊙ map)` over all 5,035 positions; `state = Linear(18→256)(pooled)`. Pooling and expansion commute (both linear), so the expansion runs *after* the crush — one 18→256 matmul per level instead of 5,035.

### 4.3 Stage 3 — multi-cosine global mixing (771 params, CPU fp32)

Given states `v₁,v₂,v₃ ∈ ℝ^{256}` and shared `U ∈ ℝ^{3×256}`:

```
(s1ᵢ, s2ᵢ, s3ᵢ) = (U₁·vᵢ, U₂·vᵢ, U₃·vᵢ)          for i = 1..3
wᵢ = Σⱼ s1ⱼ · cos(π·tanh(s2ⱼ·s3ᵢ) + φ)             cross-vector cosine weave
g  = softmax(w)                                     softmax kept — CPU has no op limits
mixed = Σᵢ gᵢ·vᵢ  ∈ ℝ^{256}
tokens = reshape(mixed, 16×16) ∥ learned 2-d pad → (16, 18)
```

Each state contributes an (amplitude, frequency) pair; each state's third scalar is evaluated under **all three** lenses. This is the most exotic block in the model and it survives deployment untouched because it operates on 771 floats — it runs on the Pi CPU in exact fp32 in ~microseconds.

**Note (spec correction):** the original spec said "split 256 into 8×16" — 8×16 = 128 ≠ 256. Resolved as 16 tokens × 16-d (D18).

### 4.4 Stage 4 — head (505 params)

Two pooled streams per window (D31):

- **Local stream** (per window): `{own 18-d embedding, own 3 Path-A vectors}` (4 × 18-d) → BN(18) → `CosineGate(18)` → sigmoid → gated mean → 18-d.
- **Context stream** (once per frame): 16 global tokens → BN(18) → `CosineGate(18)` → sigmoid → gated mean → 18-d.

Concat (36-d) → `Linear(36,8) → SiLU → Tanh → Linear(8,3)`.

Token-set semantics: *what's here* (own embedding) + *what's around here at 3 scales* (Path A) + *what frame is this* (context stream). The streams are pooled separately because the 16 global tokens are identical for all 5,035 windows: pooled jointly (v6) they capped per-window evidence at 4/20 of the vector and the shared BN's variance was dominated by cross-image variation, collapsing the head into an image classifier (D31).

**Cell averaging.** Window logits `(B,3,53,95)` → grouped `conv_transpose2d` with a 2×2 ones kernel → `(B,3,54,96)`, divided by a precomputed coverage-count map (1/2/4 at corners/edges/interior). Every 10×10 cell's logits are the average of the ≤4 windows covering it — a free 4-view ensemble, differentiable, so the training loss applies at cell level directly.

---

## 5. Parameter budget

| # | Block | Computation | Params |
|---|---|---|---|
| 1 | Dual quaternion | 4 + 4 | 8 |
| 2 | High-pass stem | depthwise 3×3×3, no bias | 27 |
| 3 | Mixing rounds ×3 | 3 × (3·8 V + φ + σ) | 78 |
| 4 | Round BNs ×3 | 3 × BN(8) | 48 |
| 5 | Per-token MLP | 8·16+16 + 16·16+16 | 416 |
| 6 | Encoder pool BN + gate | BN(16) + 3·16 + φ | 81 |
| 7 | Path A kernels | 9 + 49 + 121 | 179 |
| 8 | Path B BNs ×3 | 3 × BN2d(18) | 108 |
| 9 | Path B scorers ×3 | 3 × (18·8+8 + 8·4+4 + 4+1) | 579 |
| 10 | Path B expands ×3 | 3 × (18·256+256) | **14,592** |
| 11 | Global mix | 3·256 U + φ + pad 2 | 771 |
| 12 | Head gates + BNs ×2 streams | 2 × (3·18 + φ + BN(18)) | 182 |
| 13 | Head classifier | 36·8+8 + 8·3+3 | 323 |
| | **Total** | | **17,392** |

84% of the model is row 10. Deliberately kept unshared across levels (capacity should live in how global evidence is shaped); sharing one expansion drops the model to ~7.3k if ever needed.

The table is the locked shared-Path-A spec. Current training defaults add: edge_dq stem (D33, +307 over the 27-param high-pass), hidden-24 encoder (§8.2), the per-scale 1×1 convs (D36), and **per-channel Path A** (D37, row 7 → ~4.3k) — landing around ~24–25k params. All are box/identity-initialised or frozen-channel additions, so they warm-start at the spec model's behavior; none touch the deploy op set.

---

## 6. Compute, memory, and deployment analysis

### 6.1 FLOPs per frame (960×540)

| Stage | FLOPs | Share |
|---|---|---|
| Stage 1 (5,035 × ~0.48 M) | ~2.4 G | ~96% |
| Path A maps | ~32 M | 1.3% |
| Path B | ~7 M | — |
| Stage 3 (CPU) | ~10 k | — |
| Head | ~12 M | — |
| **Total** | **~2.5 G** | |

At 30 FPS: 0.075 TOPS ≈ **0.3% of the Hailo-8's 26 TOPS peak**.

### 6.2 Throughput estimate (Raspberry Pi 5 + AI HAT+)

Anchor: YOLOv8n (8.7 GFLOPs, conv-friendly) measures ~431 FPS on this hardware at 640×640 int8. ANetV1 is 3.5× fewer FLOPs but with 5–10× worse expected NPU utilization (channels of 16–18 vs 64–256, LUT activations, elementwise-heavy stages, 4-phase graph): **estimate 170–350 FPS; worst plausible ~100 FPS.** The 30 FPS target clears with ≥3× margin — *conditional on full NPU residency* (§8, risk 1).

### 6.3 Host/NPU split and PCIe

- **In:** 960×540×3 int8 = 1.56 MB/frame → 47 MB/s at 30 FPS (~5% of PCIe Gen3 x1).
- **NPU:** Stage 1 (4-phase dense form, §13), Path A, Path B, head.
- **CPU:** frame downscale (~1–2 ms), Stage 3 (µs), cell argmax. No NMS, no box decode.
- **Out:** window logits ~15 KB + 3×256 states 3 KB per frame — trivial.
- **Never** ship unfolded windows over PCIe: 41 MB/frame = 1.2 GB/s > link capacity. Overlap duplication must be on-chip (4-phase formulation).

### 6.4 Memory

- **Model:** 68 KB fp32, ~17 KB int8.
- **Training (MPS):** dominant activation is `(B·5035, 400, 16)` ≈ 130 MB fp32 per image per stored copy. With gradient checkpointing on the encoder: batch 4 + grad-accum 4 on 16 GB unified memory; batch 8–16 on 32 GB.
- **Inference (HAT):** all activations stream on-chip; host buffers < 30 MB.

---

## 7. Design decision log

Every decision, its alternatives, and why. Numbered in rough pipeline-then-history order.

**D1 — 20×20 windows, stride 10, at 960×540.** Alternatives: stride 20 (no overlap, ¼ compute, loses overlap ensemble), 40×40/stride 20 (same 4× overlap factor → *identical* stage-1 compute; 4× fewer embeddings but 2× coarser labels — was v4, reverted), stride 1 ("pure sliding" — 204k windows, downstream infeasible). Overlap factor = area/stride² = 4 in both 20/10 and 40/20; the pixel-token count (and hence stage-1 FLOPs) is invariant to that choice. 20/10 chosen for finer output granularity.

**D2 — Vertical-strip + horizontal-slice dual pass: removed.** Original design processed 20×1080 strips and 20×1920 slices "sharing all weights." Since the encoder never uses context outside its own 20×20 chunk, both passes evaluate the same per-chunk function — at stride 20 they produce *bit-identical* embeddings (2× compute for nothing); at stride 10 they sample offset grids `(10i,20j)` and `(20i,10j)` while double-computing the intersection. Replaced by one dense pass at stride 10 in both axes: strictly more coverage, no duplicates.

**D3 — Window-relative (u,v) on every pixel token.** The original architecture had *no positional information anywhere*: every op was permutation-invariant/equivariant, making the model a provable bag-of-pixels (any pixel shuffle inside a window → identical embedding). Coordinates as token features fix this in-family (no bolt-on positional encodings). Global coords would be numerically invisible at within-window scale (Δ≈0.001) — hence window-relative here.

**D4 — Global (x,y) on embeddings** (= mean of the center-4 pixel positions, per original spec). Gives Path B's attention-replacement and the head absolute position (horizon/runway bias) at the scale where it's legible. Side effect: coordinates riding inside the 18 dims gave the (since-removed) map-level attention positional awareness for free.

**D5 — Quaternion → dual quaternion → baked constant.** A single quaternion = pure rotation of RGB (4 params); the dual part adds translation (color offset). Still strictly weaker than a full 3×3+bias affine (12 params, adds scale/shear) — kept for the norm-preserving inductive bias and because it's a signature piece. Deploy cost is zero: exported as a constant 3×3 conv + bias.

**D6 — Cosine gate form: `s1·cos(π·tanh(s2·s3) + φ)`, φ init π/2.** Raw `s1·cos(s2·s3)` has three defects: (a) **dead at init** — with small weights s2·s3≈0, cos≈1, and ∂/∂s2 ∝ sin(s2·s3)≈0, so the frequency path gets no gradient and the gate trains as plain `s1`; (b) unbounded frequency → oscillatory loss surface and int8-fatal quantization; (c) cos is even → sign of s2·s3 invisible. Fixes: φ=π/2 init makes the gate ≈ −sin at init (SIREN-style, live gradients); tanh bounds the argument to one period (also exactly what makes an 8-bit LUT accurate); π scaling uses the full period. L2 penalty on the s2/s3 vectors additionally bounds learned frequencies (D24).

**D7 — Gaussian-by-location gate parameters (replaces "random pixel's cosine function").** The random-pairing idea degenerates: one random pixel per window → all 400 scores equal → softmax uniform → gate collapses to a plain mean; per-token random partners → weight statistically decoupled from content → expectation again ≈ plain mean, plus inference nondeterminism. The Gaussian version — blur the s1, s2 *maps* with a learned-σ kernel, keep s3 per-token — is the deterministic realization of the same intent: neighborhood defines the lens, token supplies the probe. σ learns the neighborhood scale and interpolates per-token ↔ whole-window gating. Cost ≈ 11k MACs/round/window (separable blur of two 20×20 maps).

**D8 — Gated mean added to RGB channels only; coordinates frozen.** Adding the pooled 5-d mean to all channels would drift every token's (u,v) each round, corrupting position. The rank-1 update touches only the 3 feature channels; SiLU likewise applies to features only.

**D9 — SiLU replaces ReLU.** On 3-channel feature vectors, ReLU zeroing a channel destroys ⅓ of the representation permanently. (User decision, v2.)

**D10 — Sigmoid-gated pooling replaces spatial softmax everywhere on-NPU.** Spatial softmax needs exp + activation÷activation normalization — fragile-to-unsupported in the DFC and quantization-hostile. `mean(sigmoid(score) ⊙ x)` keeps content-dependent weighting with only supported ops; the lost normalization (output scale varies with active-token count) is absorbed by the following BN. Applied at: mixing rounds, encoder pool, Path B, head. Stage 3 keeps true softmax because it runs on CPU. Trained in this form from step 0 (train-what-you-deploy).

**D11 — BatchNorm replaces RMSNorm.** Both stabilize the scale-sensitive gates (cos frequency, sigmoid sharpness). RMSNorm is data-dependent at inference → a Hailo pain point. BatchNorm uses frozen statistics at inference and **folds into adjacent convs at export** — zero deploy cost. Placement: before every scoring computation (each round, encoder pool, each Path B level, head).

**D12 — Per-token MLP 5→16→16; 16-d embeddings.** The deliberate bottleneck: 4,800 window values → 16 floats (300:1). Pre-registered upgrade if mannequin-window training loss plateaus: 5→24→24 / 24-d embeddings (+~900 params) — first knob to turn, before anything else (§8, risk 2).

**D13 — Path A kernels: one scalar per position, shared across channels.** 179 params for three context scales. Upgrade path if tent boundaries are mushy: per-channel kernels (×18 params). Receptive spans 40/80/120 px at 540p bracket torso→tent scales (dataset-derived, §1.2).

**D14 — Path A feeds the head (per-region context).** Original design crushed each level to one global vector and gave every region identical context — a tent spanning 12 windows couldn't pool evidence with neighbors except through a 768-float image-wide bottleneck, and per-location multi-scale features were computed then discarded. Fix: each window's own 3 pyramid vectors join its head token set. ~Zero params; this is the difference between "is this patch tent-colored" and "is this patch inside a tent-shaped blob."

**D15 — Path B: self-attention removed.** History: full-res self-attention over 20,437 tokens (v3 grid) = 418M pairwise scores = ~10 GB fp32 materialized and ~90–160 GFLOPs — for a branch whose entire output is 3×256 floats. Interim fixes explored: stride-2 learned downsample (v3), 40×40 windows (v4), "flash attention" (rejected: flash reduces *memory traffic*, not FLOPs, and doesn't exist on Hailo/CPU targets). Terminal reason: QK matmuls (activation×activation) are the DFC's weakest op class. Replacement: gated global pooling (D10 form). The information argument: attention enriched tokens moments before a global average — quadratic work feeding a 768-float pipe.

**D16 — Pool-then-expand.** `mean(gate⊙map)` then `Linear(18→256)` ≡ expanding every location then pooling (both linear, they commute), at 1/5,035 of the matmul cost.

**D17 — Stage 3 on CPU in fp32, exactly as designed.** The multi-cosine weave + true softmax operate on 771 floats — the one place the most exotic math costs nothing and faces no compiler or quantization risk. NPU→CPU→NPU hop ships ~3 KB.

**D18 — 256 splits into 16×16 tokens (spec bug fix).** Original: "split the 256 into 8 16-dim" — 8×16=128≠256. Resolved to 16 tokens × 16-d (head gets 20 tokens); alternatives (project 256→128, or 8×32-d tokens) rejected as adding params/complexity for no benefit.

**D19 — Head: QK attention → cosine-gated pooling.** Same DFC reason as D15, at awkward per-region shapes (5,035 × 20×20 score matrices). The replacement keeps content-dependent token weighting via the signature gate, all in supported ops. Bonus: the two blocks hardware forced out (D15, D19) were the most *conventional* parts of the design.

**D20 — Classifier `18 → SiLU → 8 → Tanh → 3`** per original spec arrow-chain, read as: Linear(18,8) → SiLU → Tanh → Linear(8,3). Tanh bounds the pre-logit representation — mildly helpful for int8 calibration of the final layer.

**D21 — Cell-level output with overlap averaging.** Each 10×10 cell (540p) averages the logits of its ≤4 covering windows: free ensemble, differentiable, and the *loss is applied at cell level* so training and deployment optimize the same quantity. Effective label granularity: 20×20 px at capture ≈ 0.24–0.56 m ground.

**D22 — 960×540 input.** The dataset math (§1.2), not YOLO convention, sets resolution. At 540p the typical mannequin is 49×13 px (comfortably detectable); the cost is the tail (29×8 px worst case; ~6 px occluded fragments under ~4 px downscaled motion blur). 640-class input was rejected: typical 33×8, worst 19×5 — the worst decile of the dataset becomes physically invisible. Operational mitigations for the 540p tail: 30 FPS gives many looks per target per pass; and all weights are resolution-independent, so 1080p fine-tuning is a warm start if worst-decile recall collapses (§8, risk 4). Bonus: ~4× faster training iteration.

**D23 — Class imbalance: focal loss (γ=2, α=[1,8,4]) + balanced image sampling (weight 1 + 2·has_tent + 4·has_mannequin) + per-class metrics only.** ~5,180 of 5,184 cells are "nothing" in a typical frame; plain CE converges to the constant background predictor with 99.9% accuracy. Accuracy is never reported; the primary metric is object-level recall (§11).

**D24 — Targeted regularization, not blanket weight decay.** L2 penalty *specifically on s2/s3 gate vectors* (mechanistically bounds cosine frequencies — the one place decay has a justification); L1 on Path A kernels (sparse, interpretable neighborhoods); AdamW weight_decay=0 otherwise. Coefficients 1e-4.

**D25 — 4-phase dense formulation for the NPU graph.** Stride-10 20×20 windows = four phase-shifted non-overlapping 20×20 tilings (offsets {0,10}²). Each phase is a plain dense conv/pool graph (per-window ops become 20×20-window pooled ops at stride 20). Avoids unfold (unsupported) and avoids 1.2 GB/s PCIe (D-constraint §1.3). Training uses `F.unfold` (simpler, autograd-friendly); the export graph uses phases.

**D26 — Quantization-aware training planned, not post-hoc calibration only.** The cascaded gates (cos LUT → sigmoid → mean → BN, three rounds deep) accumulate quantization error; bounded arguments (D6) make each LUT accurate but QAT is expected to be needed for the cascade.

**D27 — Distillation from YOLO26n (experiment 3).** ANetV1 has no pretraining (D29); YOLO26n fine-tuned on the same 50k frames provides per-box soft evidence, rasterized to per-cell soft labels (prob = conf × coverage) and distilled via KL (T=2, weight 0.7) mixed with hard focal (0.3). Cheap to cache (one inference pass over the train split), directly comparable (same label space).

**D28 — Path B expansions kept unshared across levels** (86% of params). The three levels see different context scales; sharing the 18→256 would save 9.7k params the deployment doesn't need saved.

**D29 — No pretraining, 50k synthetic frames from scratch.** Nothing exists to pretrain a 17k-param non-standard architecture on; the balanced sampler + focal loss + distillation (D27) are the compensations. Acknowledged as YOLO's structural advantage in the comparison.

**D30 — MPS training specifics.** Gradient checkpointing on the encoder (recompute in backward; halve BN momentum to compensate for double stat updates); fp32 default (model is tiny; MPS fp16 autocast available behind a flag); `PYTORCH_ENABLE_MPS_FALLBACK=1` as safety net; batch 4 × grad-accum 4 default.

**D31 — Head: split-stream pooling (v7 fix for 0.000 mannequin recall).** The v6 head gated-pooled all 20 tokens in one mean. The 16 global tokens are *identical for every window of a frame*, so per-window evidence (own embedding + 3 Path-A) could contribute at most 4/20 of the pooled vector, and the shared BN normalized against variance dominated by cross-image global-token variation. Measured failure: head logits constant per frame regardless of GT cell class (mean mannequin logit −0.75 on background, mannequin, and tent cells alike); 0/3,443 mannequin cells predicted while a linear probe on the same per-window features recovered signal. Fix: pool the 4-token local stream and the 16-token context stream separately (each BN → cosine gate → gated mean), classify from the 36-d concat. Per-window signal now owns half the classifier input unconditionally. Same op classes as v6 (BN folds, sigmoid gates) — Hailo-neutral; the context stream is computed once per frame, not per window.

**D32 — High-pass texture stem (v7 fix for weak mannequin separability, risks 2/6).** Stage 1 tokens were single-pixel `(r,g,b,u,v)`; the only spatial op in the encoder was a Gaussian blur of gate scores, so nothing in the model could see edges or texture. Mannequins (13 px wide typical) differ from near-object clutter by shape/texture, not color — the pre-registered capacity bump (D12 upgrade, hidden 24) added width but no spatial features and did not recover recall; risk 6 named this ceiling. Fix: depthwise 3×3 zero-DC conv on the quat RGB (27 params) appends 3 local-contrast channels to every token, frozen through the rounds like (u,v). Placed on the full frame before windowing: windowed and 4-phase dense paths remain bit-identical, and the export graph gains exactly one standard conv (Hailo-native). Kept as the minimal-stem ablation (`stem="highpass"`) after D33 superseded it as the default.

**D33 — EdgeDQ stem (v7 default).** Oriented-edge dual-quaternion front-end, validated by probing raw pixels + fixed 7×7 Sobel maps before building it (`probe.py --edges`). Triplicate the frame: one copy stays raw colour; the other two pass through a learned DQ colour rotation then a learnable depthwise 7×7 oriented edge conv (Sobel-7 init, vertical / horizontal); each 3-channel group is re-framed by its own DQ → 9 stem channels, 11-d tokens. The block-diagonal DQs keep the colour/edge grouping clean: the encoder's residual update still touches only the 3 colour channels and reads the 6 edge channels as frozen evidence. All plain convs at export (D5-style, Hailo-native); 334 params.

**D34 — Eval/deploy graph algebraically restructured for memory bandwidth (exact math, training path untouched).** The eager eval graph was launch/bandwidth-bound (~92 ms/img on MPS; the arithmetic itself is sub-millisecond). Rewrites, each bit-equivalent to the reference windowed path (asserted by the dense-vs-windowed test) and gated on `self.training` where they rely on frozen BN statistics: (1) einsum → 1×1 conv / matmul everywhere (the ORT CoreML EP has no Einsum builder; einsums forced CPU partitions); (2) 4-phase merge via pad + `pixel_shuffle` instead of strided ScatterND; (3) all 4 phases ride the batch dim through ONE encoder pass (`_map_dense_batched`) — legal because every op is tile-local and eval BN is affine; garbage tiles from padding land exactly on the cropped row/col; (4) eval-BN affines fold into the following 1×1 convs (rounds, encoder pool); (5) mixing rounds run on the 3 RGB channels only (`dense_round_rgb`) — the 8 frozen channels' score contributions for all 3 rounds are precomputed in one conv, and the gated pool never needed the frozen channels (only `pooled[:, :3]` was ever used); (6) the per-tile Gaussian blur is a banded 20×20 matrix applied to each contiguous 20-slice (`reshape(-1,20) @ K`) instead of 5,184 tiny per-tile convs. Result: 92 → 21 ms/img (hidden=24) via ONNX Runtime + CoreML EP on an M-series GPU — single partition, exact argmax parity on real frames. Rejected after measurement: fp16 (1.4× slower on Apple GPU — fp32-rate ALUs plus cast overhead; ANE mangles the op mix), coremltools direct conversion (mistranslates the graph: argmax agreement 0.45 even at fp32), multi-stream concurrency (zero gain — GPU already bandwidth-saturated at ~11 ms in the full-res per-token MLP).

**D36 — Per-scale 1×1 conv after Path A (`path_dq`).** Each Path-A scale (k3/k7/k11) is followed by a learned d→d 1×1 conv (identity-init, so it starts as a no-op and bakes to a constant 1×1 conv at export — Hailo-native, D5-style). Lets each scale recombine its channels before *both* Path B and the head read it, instead of forwarding the raw neighborhood sum. Zero deploy risk, ~2k params at d=26.

**D37 — Per-channel Path A kernels (viz-driven capacity bump, ARCH §8.2 step 2).** The `runs/viz/000008` dump showed the failure the ablation plan predicted for mushy tents: cell tent-recall ~½ (48/98 predicted), a low-contrast (dark green) tent almost entirely missed in the tent logit while its encoder embedding and all three Path-A scales clearly fired on it, and the mannequin channel lighting up on a car (a large distractor) — i.e. the *features* separate the objects but the head can't resolve class at the right *scale*. The shared-scalar Path A (D13) gave every embedding channel the same 3/7/11-px neighborhood weighting, so the head could not learn "this channel matters at tent scale, that one at mannequin scale." D37 promotes each scale to a full **depthwise per-channel** kernel (one k×k filter per channel), box-filter-initialised so training starts exactly at the D13 model and only adds resolving power. It is the pre-registered §8 risk-2 *second* step (after the hidden-24 width bump), L1-regularised like the shared form (penalty normalised by channel count so the per-kernel pressure is unchanged, D24), and reversible via `path_a_per_channel=False`. Deploy-neutral: still a depthwise conv, the Hailo DFC's favourite op. Param cost ~4.3k (d=26); the model stays ≪ the Path-B expansion's 84%.

**D35 — Fused Metal kernel (`anet/metal.py`) breaks the graph-runtime bandwidth wall.** Any graph runtime materializes every intermediate full-res map to DRAM (~2.5 GB/frame → the D34 21 ms floor). But the architecture is threadgroup-shaped: one 20×20 window = one 400-thread Metal threadgroup, so `torch.mps.compile_shader` runs the ENTIRE encoder — all 3 mixing rounds (folded score dots, in-tile separable blur via shared memory, cosine gate, `simd_sum` gated tile means, SiLU residual), the per-token MLP, and the final cosine-gated pool — in registers and ~3 KB of threadgroup memory, reading the 9-channel stem map once and writing the phase-interleaved (hidden, 53, 95) grid directly. The stem itself folds into ONE dense 7×7 conv over `[img, ones]` — the constant ones channel (zero-padded like the image) reproduces the DQ translation's valid-tap sum at borders exactly, so even the padding semantics are bit-faithful. DRAM traffic: ~2.5 GB → ~90 MB/frame. Measured (M-series, hidden=24): **7.4 ms/img, 134 img/s**, max logit delta ~4e-6, cell-argmax agreement 1.0 on real val frames — vs YOLO26n on the same machine at 13.6–16.5 ms wall / ~9.5 ms pure inference. Eval-only wrapper around a trained checkpoint (`MetalANet.from_checkpoint`); training and the Hailo export are untouched.

**D38 — Training throughput on CUDA/ROCm (launch-bound, not compute-bound; core arch untouched).** The MI300X sat at ~1% util spending ~430 s/epoch on a 20k-param net — the wall was kernel *dispatch*, not FLOPs. Two fixes, neither changing model semantics: (1) **single batched Stage-1 pass in training** — the training map used to launch the whole 96%-FLOP encoder four times in a Python phase loop (`_map_dense`); now all four stride-2 phases ride the batch dim through ONE `encoder.forward_dense` (unified `_map_dense_all`), exactly the D34.3 trick already used at eval. The only subtlety is BN: eval folds frozen stats so padding is free, but training BN reads *batch* stats, so the phase crops are **edge-replicate**-padded (not zero) — the ~3% padded tiles are then in-distribution and don't skew the (now larger, joint 4-phase) BN batch the valid tiles normalise against. Bit-identical to the windowed reference at eval (asserted); training-equivalent, ~4× fewer Stage-1 dispatches. (2) **`torch.compile` on for CUDA/ROCm** (`mode="default"`): inductor fuses the elementwise chains — cos/tanh/sigmoid/SiLU, the folded BN affine, the gated-pool multiplies — into a handful of Triton kernels, the actual "fused kernels/ops" win. `reduce-overhead` (HIP-graph capture) is *not* the default: it aliases the compiled output buffer and fights grad-accum + the on-GPU loss accumulation (`loss_win += loss.detach()`), which crashed the build; `default` fuses without that hazard. Host-OOM while compiling the backward graph (inductor forks one full-torch worker per thread) is fixed by `TORCHINDUCTOR_COMPILE_THREADS=1` (set before the first lazy compile and in the run script), with a warm on-disk cache. Any compile error — setup or first-step — degrades to eager instead of dying; `ANET_COMPILE=0` is the fast off-switch. Also `set_float32_matmul_precision("high")` for the fp32 GEMMs outside autocast. The Hailo export path and model math are untouched.

**D39 — DeployNorm: deploy-form normalization (v9).** BatchNorm's training mode normalizes with *batch* statistics, which (a) couples every 20×20 tile to every other tile in the batch — the single reason the encoder could not be fused into one kernel (each round forced a full-resolution HBM round trip so the next round could see batch stats); (b) triggered the MIOpen int32 overflow at batch ≥ 44, silently dropping to the primitive-op path that materializes fp32 copies of ~19 GB tensors (the observed ~120 GB at batch 96); (c) double-updated stats under checkpointing. DeployNorm normalizes with the RUNNING statistics — exactly the affine the deploy graph uses after BN folding — and updates them as a detached EMA of observed batch stats (momentum ramp seeds from the first batches; the trainer additionally runs 8 no-grad seeding passes before step 0). Consequences: the training forward IS the deployment forward (train-what-you-deploy with no BN train/eval gap left at all); normalization is a constant per-channel affine within a step, so the encoder is tile-local and fusable; no gradient flows through statistics. At batch 96 the stats are averages over ~10⁸ tokens per step — the EMA is glassy smooth and the one-step lag is noise. Folds at export identically to BN (same buffers).

**D40 — Fused Triton Stage-1 training kernels.** The D35 Metal kernel proved the architecture is threadgroup-shaped at eval; v9 does the same for *training* on ROCm/CUDA: one Triton kernel per direction runs the entire per-token Stage 1 — 3 mixing rounds (per-tile blur as a banded-matrix `tl.dot`, cosine gates, gated tile means, SiLU residuals), fc1, and the cosine-gated pool — in registers, reading the 15-channel stem map once (phase offsets computed in-kernel; the 4-phase crops are never materialized) and writing the (B, 48, 53, 95) pooled grid. The backward kernel saves nothing but the stem map: it recomputes each tile in registers and emits d_feat (atomics — phases overlap) plus all parameter grads (slot-spread atomics; σ grads are projected onto ∂K/∂σ in-kernel so the blur-kernel gradient is one scalar per round). DeployNorm batch stats are accumulated by the forward kernel via slotted atomics and EMA-applied after the step. **Verification is layered and automatic at startup** (`Trainer._setup_fused`): fused forward is parity-checked against the PyTorch dense path on real frames; the Triton backward is parity-checked against a chunked-autograd backward (autograd through a pure-torch mirror of the identical folded-parameter math, `pool_from_params` — asserted to 1e-8 against the dense path in the smoke test); any mismatch demotes one level (triton bwd → chunked bwd → PyTorch dense at a VRAM-safe batch), loudly. Fused-path activations at batch 96 are ~6–9 GB vs ~120 GB before.

**D41 — Sobel-init 4-orientation stem.** Two bugs/gaps in one: the D33 stem's "Sobel-7 init" was actually `weight.mul_(0.2)` on default kaiming noise — the stem started with NO oriented-edge structure and had to discover edges through the encoder's weak early gradients (a measured contributor to the mannequin cold start); and the v/h pair leaves 45°-oriented limbs at ~0.7× response while mannequins lie at arbitrary yaw. v9: four depthwise 7×7 edge convs genuinely initialised to oriented Sobel-7 operators at 0/90/45/135° (×0.5), each behind its own DQ colour rotation, all learnable → 15 stem channels (3 colour + 12 frozen edge evidence). The v8 `EdgeDQStem` init is also fixed to true Sobel for reproducibility of the ablation.

**D42 — fc2 after the pool.** The per-token stage becomes fc1 (17→48) + gate + pool; the 48→32 layer runs on the 5,035 pooled windows instead of 2·10⁶ full-resolution positions. Removes ~45% of full-res FLOPs and the largest activation tensor. Capacity argument: h1=48 > hidden=32 keeps the pre-pool width, and the cosine gate is still a second data-dependent nonlinearity applied at token level before the 300:1 crush; the aux probe (D46) watches whether the encoder still separates classes.

**D43 — ConvNeck: cross-window context on the embedding grid.** Two residual depthwise-5×5 + pointwise rounds on the (53×95, d=34) embedding map (~4.2k params, Hailo's favourite ops, near-identity init so the cold start is undisturbed). The v8 head saw other windows only through fixed-shape Path-A box averages; the 000008 viz showed features firing on objects the head couldn't resolve at scale. The neck gives every window a *trainable* 50–110 px receptive field before Path A / the head read it.

**D44 — SlimContext replaces the Path-B expansions.** The three 18→256 expansions were 14.6k params (60–84% of the model) feeding one per-frame vector that D31 showed was actively diluting per-window evidence. v9 keeps the signature pieces — per-scale gated global pooling and the multi-cosine state weave (still CPU-sized: three d-dim states) — but states stay at width d and the mixed vector feeds the classifier directly. ~1.3k params. The freed budget went to the neck, the wider head, and hidden=32.

**D45 — Head classifier widened 8 → 24.** Linear(2d→24) → SiLU → Tanh → Linear(24→3), prior-bias init (RetinaNet §4.1) at p=0.05. The 8-d Tanh choke was the narrowest point in the network — everything the encoder discriminates had to survive 8 dims. Split local/context streams (D31) kept.

**D46 — Aux deep-supervision probe (train-only).** A 1×1 conv (d→3, 105 params) on the pre-neck embedding map, overlap-averaged to cells, added to the loss at weight 0.3. Direct gradient path to the encoder that a collapsed head cannot block — the linear-probe experiments repeatedly showed signal in the embeddings that the head lost; now that probe trains *with* the model. Dropped at eval/export: zero deploy cost.

**D47 — focal_norm loss.** The Focal-Tversky stack failed structurally, twice over: set-level ratios make each cell's gradient depend nonlinearly on batch TP/FP totals (spiky; the documented source of the fp 0.3↔28 and mannequin 0↔overshoot limit cycles), and the FT + focal-anchor pairing is a two-term tug-of-war (re-created every time the anchor was re-tuned). In the observed failure it actively pushed tent soft-prob from its 0.1 prior init down to 0.003. v9 uses ONE smooth per-cell term with CenterNet/FCOS-style size invariance: per-class summed focal normalized by that class's positive-cell count in the batch (background normalized by total foreground count, boundary-band cells masked). Every positive cell of a rare class carries O(1) gradient regardless of rarity; at prior init the foreground pull dominates ~40:1 so recall rises first and precision pressure grows as predictions appear. Class weights (1, 2, 1).

**D48 — Weight EMA (decay 0.998) for eval and checkpoints.** The object-recall selection metric is noisy epoch-to-epoch; the EMA weights are what get evaluated, selected, and saved (best.pt/last.pt), so what's flown is a smoothed model, not the last optimizer step. Raw weights keep training.

---

## 8. Known risks and pre-registered mitigations

Ranked. Each has a trigger and a pre-agreed response — decided now so results don't get rationalized later.

1. **DFC rejects an op (cos-LUT, gated pooling pattern, 4-phase reshape).** *Trigger:* compile spike failure. *Response:* plan-B gate `s1·tanh(s2·s3)` (pure supported ops, loses oscillation); restructure pooling as avg-pool of pre-multiplied maps. The compile spike runs **before training** so the trained model is the deployed model.
2. **Encoder under-capacity** (508 params carrying all texture discrimination against deliberate near-object clutter + color harmonization). *Trigger:* mannequin-window train loss plateaus high / train-val gap near zero with poor recall. *Response:* widen 5→24→24, 24-d embeddings (+~900 params); second step: per-channel Path A kernels.
3. **int8 drift through cascaded gates.** *Trigger:* fp32-vs-int8 eval gap > a few recall points. *Response:* QAT fine-tune (D26); worst case: freeze σ, φ and re-calibrate.
4. **Worst-decile recall at 540p** (29×8 px occluded mannequins). *Trigger:* the GSD/occlusion-sliced eval (§11). *Response:* 1080p fine-tune from 540p weights (resolution-independent params); accept 2× inference cost — still >30 FPS.
5. **Early false-positive floods** (sigmoid gates + background-dominated cells before BN settles). *Trigger:* object FP/image not collapsing within first epochs. *Response:* warmup with higher background α; verify balanced sampler weights.
6. **Rank-1 mixing ceiling** — three rounds of add-one-shared-vector is context *conditioning*, not token routing; if shape discrimination underperforms, this is the architectural suspect behind risk 2.

---

## 9. Ablation plan

One training run each at 540p (cheap). Keep/kill by object-level recall on val.

| Ablation | Question |
|---|---|
| cos gate vs `s1·tanh(s2·s3)` | does the oscillatory gate earn its LUT risk? |
| sigmoid vs softmax pooling (GPU-only) | what did D10 cost? |
| σ learned vs frozen 3 px | is the Gaussian interpolation doing work? |
| Path A tokens removed from head | value of per-region context (D14) |
| 3 mixing rounds vs 1 | depth of the context conditioning |
| dual quaternion vs plain 3×3+bias affine | inductive bias vs expressivity |
| global tokens 16 vs pooled-8 (project 256→128) | D18 resolution |
| shared vs unshared Path B expansions | is 86% of the params pulling weight? |

---

## 10. Baseline comparison: YOLO11n / YOLO26n

Reference specs: YOLO11n = 2.6M params, 6.5 GFLOPs @640. YOLO26n ≈ same scale, natively end-to-end (no NMS), DFL removed for edge/int8 friendliness, STAL small-target-aware loss, ~31–43% faster CPU inference than 11n. Measured anchor: YOLOv8n (8.7 GFLOPs) ≈ 431 FPS on Pi 5 + Hailo-8 @640 int8.

| Axis | ANetV1 @960×540 | YOLO11n @960×544 | YOLO26n @960×544 |
|---|---|---|---|
| Params | **17k** | 2.6M (~153×) | ~2.5M |
| Model int8 | **~17 KB** | ~3 MB | ~3 MB |
| GFLOPs | **~2.5** | ~8.3 | ~8 |
| Est. HAT FPS | 170–350 | 250–400 | 250–400, no host NMS |
| Post-processing | argmax (µs) | decode + NMS on CPU | none |
| Small-object machinery | native windows, unproven | proven | proven + STAL |
| Pretraining | none | COCO | COCO |
| Output form | per-cell grid (task-native) | boxes → rasterized | boxes → rasterized |
| Deploy risk | compile spike pending | model-zoo path | community pipeline exists |

**Honest framing:** everyone clears 30 FPS — speed is not the differentiator. ANetV1's real edges: 150× fewer params / 3× less compute (power + thermals on the airframe), task-native output, full understanding of every parameter. YOLO26n's edges: pretrained backbone, proven small-object recall, mature int8 path — and it specifically neutralizes ANetV1's NMS-free and quantization-friendly talking points. **Decision metric:** worst-decile mannequin object-recall (max-GSD × occluded × grainy slices). Within ~5 points of the YOLO26n teacher → fly ANetV1. 15+ points behind after risk-2 mitigations → fly YOLO26n.

---

## 11. Training configuration

Three experiments, same data, same eval:

| | Exp 1: YOLO26n | Exp 2: ANetV1 | Exp 3: ANetV1-distilled |
|---|---|---|---|
| Init | COCO pretrained | scratch | scratch |
| Labels | YOLO boxes | hard cell grids (coverage ≥ 0.3) | 0.3·focal(hard) + 0.7·KL(teacher soft, T=2) |
| Teacher | — | — | Exp 1 checkpoint, cached per-image `.npz` |

ANetV1 defaults: AdamW lr 3e-3 (cosine schedule), weight_decay 0, focal γ=2 α=[1,8,4], L2(s2/s3)=1e-4, L1(kernels)=1e-4, batch 4 × accum 4, ~30 epochs, balanced sampler, device MPS.

**Metrics (all three models, test split):** per-class cell P/R/F1; object-level recall (GT box found if ≥1 of its cells predicted with its class); object FP/image (connected components matching no box); the same numbers on the worst-GSD/occluded/grainy slices via gen2 metadata. YOLO boxes pass through the identical rasterizer so every number is apples-to-apples.

---

## 12. Design version history

| Ver | Change | Driver |
|---|---|---|
| v1 | Original spec: strips+slices, softmax attention pooling, `s1·cos(s2·s3)`, quaternion, 3-dot tricks, self-attn ×3 levels, 12-token head attention | — |
| v2 | Positional coords both levels; Gaussian gate (replacing random-pixel idea); φ; L1/L2; SiLU; dual quaternion; 5→16→16; 4 heads; stride 10 | first review: bag-of-pixels proof, dual-pass redundancy, dead-at-init gate |
| v3 | Unified dense windowing; Path A→head; stride-2 downsample before attention; per-site φ; frozen coords; pad vector | second review: global-context bottleneck, attention cost |
| v4 | 40×40 @ stride 20, full-res flash attention | (detour) — compute wash, coarser labels |
| v5 | Back to 20×20 @ 10; **Hailo mapping**: attention → gated pooling (both sites), softmax → sigmoid, RMSNorm → BN, bounded cos for LUTs, Stage 3 → CPU, 4-phase plan | 30 FPS on Pi 5 + Hailo-8; op support > FLOPs |
| v6 | **960×540** (dataset-derived target sizes); 16×16 global token split fix; final param/compute lock; 3-experiment plan with YOLO26n teacher | gen2 config measurements; implementation |
| v7 | **Split-stream head** (local vs context pooled separately, D31); **high-pass texture stem** (depthwise 3×3, tokens 5→8, D32); 17,392 params | trained v6 hit 0.000 mannequin cell recall: head logits constant per frame (global-token dilution), linear-probe lift ×1.4 (no edge/texture features) — risks 2/6 triggered |
| v8 | **EdgeDQ stem default** (D33); **per-scale 1×1 conv after Path A** (D36); **per-channel Path A** (D37, ~24–25k params); **CUDA/ROCm training throughput** (batched Stage-1 pass + `torch.compile`, D38) — model math and Hailo export unchanged | `runs/viz/000008`: tent cell-recall ~½, low-contrast tent missed, mannequin/car scale confusion (features fire, head can't resolve scale); MI300X launch-bound at ~1% util |
| v9 | **DeployNorm** (D39); **fused Triton Stage-1 train kernels** (D40); **Sobel-init 4-orientation stem** (D41); **fc2 post-pool** (D42); **ConvNeck** (D43); **SlimContext replaces Path-B expansions** (D44); **24-wide head** (D45); **aux deep-supervision probe** (D46); **focal_norm loss** (D47); **weight EMA** (D48). ~20.7k deployed params. See §14. | from-scratch v8 runs: mannequin AND tent recall 0.000 at epoch 3 (tent soft-prob actively pushed 0.1→0.003 by the FT loss), 517 s epochs, ~120 GB VRAM at batch 96 (MIOpen int32 → primitive-BN fp32 fallback), stem "Sobel init" was actually random noise ×0.2 |

---

## 13. Implementation notes

- **Unfold ordering:** `F.unfold` flattens channel-major `(c, kh, kw)` and orders blocks row-major; `uv`/`xy` buffers are built to match. Reshape `(B, 5035, 18) → (B, 18, 53, 95)` is valid because window index = `row·95 + col`.
- **Cell averaging:** grouped `conv_transpose2d(ones 2×2)` + precomputed count map (1/2/4) — exact overlap mean, no gather ops.
- **Checkpointing:** per-segment `torch.utils.checkpoint` (`use_reentrant=False`) inside `WindowEncoder.forward_dense` — each mixing round and the MLP tail is its own segment, so the backward peak is one segment's rematerialized set, not the whole encoder's; BN momentum halved (stats update twice per step under recompute).
- **Training memory (measured 2026-07-08, hidden=24/edge_dq, batch-1 fwd+bwd, bf16):** saved-for-backward was 2.87 GiB/img: 1.35 GiB fp32 `ManualBatchNorm` intermediates (fixed — fused `F.batch_norm` everywhere except MPS, same math/running-stat convention) and ~0.5 GiB from the fp32 `uv_tile` type-promoting the whole Stage-1 stream under autocast (fixed — uv cast to stream dtype). Now 1.54 GiB/img eager, ~0.65 compiled, 0.44 + largest-segment with checkpointing.
- **RGB-only rounds (train, non-MPS):** the 3 mixing rounds carry only the 3 RGB channels, not all `in_dim` — the frozen edge/texture + (u,v) channels never change across rounds, so `MixRound.forward_dense_rgb` reads them for the gate score only. The joint `in_dim`-wide BN splits into `F.batch_norm` on the RGB slice + the frozen slice of the same running buffers (per-channel BN ⇒ bit-exact: verified fp64 forward 2e-16, all grads ≤1e-17 incl. the stem edge kernels, identical running-stat updates). ~3× lighter round conv/blur/pool work → 1.54→1.21 GiB/img and the main Stage-1 throughput lever. Same trick the eval fast path (`dense_round_rgb`) already used; MPS keeps the `in_dim`-wide path. **Model math, export graph, and parameter count all unchanged.**
- **Export:** `export_onnx()` bakes quaternion → 1×1 conv, σ → fixed 9×9 depthwise kernels; BN folding left to the DFC. The Hailo graph proper is the 4-phase dense variant (separate builder, after the compile spike).
- **Fast local inference:** fastest path is the fused Metal kernel (D35): `anet.metal.MetalANet.from_checkpoint(...)` → **~7.4 ms/img / 134 img/s** (hidden=24, M-series GPU), exact parity; needs MPS + the edge_dq stem. Portable path: `scripts/export_onnx.py --ckpt runs/anet/best.pt` produces a self-contained ONNX (sidecar weights re-inlined — the ORT CoreML partitioner can't read `.onnx.data`); `anet.onnxrt.OnnxANet` runs it at ~21 ms/img (hidden=24) / ~17 ms (hidden=16) via the CoreML EP (D34), vs ~47 ms eager MPS and ~436 ms eager CPU. ONNX profile (copy-free): stem 2.5 ms, 3 rounds ~7.3 ms, per-token MLP + pool ~11 ms (activation-bandwidth floor — the wall D35 removes), tail ≈ 0. fp16, ANE, and multi-stream all measured slower or flat (D34).
- **Determinism note:** the deployed model is fully deterministic; no runtime randomness anywhere (D7 removed the only stochastic proposal).

---

## 14. v9 — training-stack rebuild (D39–D48)

Driver: from-scratch v8 runs on MI300X hit **0.000 mannequin AND tent recall by epoch 3** (the FT loss actively pushed tent soft-prob from its 0.1 prior init to 0.003), at **517 s/epoch** and **~120 GB VRAM** at batch 96 (MIOpen int32 overflow silently dropped BN to a primitive path materializing fp32 copies), and the D33 stem's "Sobel init" turned out to be random noise ×0.2. v9 keeps every signature mechanism (DQ colour transforms, gaussian-lens cosine gates, cosine-gated pooling, multi-cosine weave, cell overlap-averaging) and rebuilds what carried them.

### 14.1 Pipeline delta (vs §3)

```
960×540×3
├─ STAGE 0 · EdgeDQStem4 (D41): raw ∥ 4× (DQ → 7×7 Sobel-init edge conv at
│    0/90/45/135°), each group re-framed by its own DQ → 15 ch; tokens 17-d
├─ STAGE 1 · TileEncoder (D39/D40/D42), per 20×20 window:
│    3 × mixing round (DeployNorm affine, gaussian-lens cosine gate,
│      gated tile mean → RGB residual, SiLU)                — unchanged math
│    → fc1 17→48 + SiLU per token → DeployNorm affine
│    → cosine-gated pool over 400 tokens → 48-d
│    → fc2 48→32 + SiLU per WINDOW (D42)  → (B, 32, 53, 95)
│    [trains as ONE Triton kernel per direction (D40); PyTorch dense path
│     and windowed token path kept, parity-asserted]
├─ concat global (x,y) → 34 ch → ConvNeck ×2 (D43): residual dw5×5 + pw
├─ Path A k3/7/11 per-channel + per-scale 1×1 (D13/D36/D37)  — unchanged
├─ SlimContext (D44): 3 × gated global pool → 34-d states → multi-cosine
│    weave → one 34-d context vector (Path-B 18→256 expansions REMOVED)
└─ RegionHeadV9 (D45): local stream {emb, 3×PathA} cosine-gate-pooled ∥
     context vector → Linear(68→24) → SiLU → Tanh → Linear(24→3)
     → cell overlap-average (D21)                            — unchanged
   [+ train-only aux probe: 1×1 conv 34→3 on the pre-neck map (D46)]
```

### 14.2 Parameter budget (v9 defaults: hidden=32, h1=48, d=34)

| Block | Params |
|---|---|
| EdgeDQStem4 | 660 |
| TileEncoder (3 rounds + fc1 + pool + fc2) | 2,934 |
| Path A per-channel + path_dq | 9,656 |
| ConvNeck ×2 | 4,216 |
| SlimContext | 1,270 |
| RegionHeadV9 | 1,970 |
| **Deployed total** | **20,706** |
| aux probe (train-only) | 105 |

Under the 40k budget with headroom; the deleted Path-B expansions paid for the neck, the wide head, and hidden 24→32.

### 14.3 Training configuration (v9 defaults)

AdamW lr 3e-3, cosine + 300-step warmup, bf16 autocast, batch 96 × accum 1 (fused) / 32 (dense fallback), grad clip 10.0 (focal_norm grad norms run 25–180; a 1.0 clip would bind every step), **focal_norm** loss (D47, γ=2, weights 1/2/1, fg floor 1 / bg floor 8) + 0.3 × aux (D46), D24 reg coefficients rescaled to 3e-3 (the new loss is ~30× larger than the per-cell-mean focal the old 1e-4 was tuned against), prior-bias init p=0.05, boundary band ignore (band_lo 0.05), balanced sampler + vd_weight 0.4 (unchanged), weight EMA 0.998 with cold-start debias ramp, parameters only (D48), 40 epochs, early stop patience 12 / min 25, DeployNorm seeding 8 batches (D39). Startup runs fused parity checks and demotes automatically (D40).

### 14.4 Compatibility and deployment split

- `runs/anet/good.pt` and all v8 checkpoints still load via `from_state_dict` (shape-sniffed) for evaluation; they cannot warm-start a v9 model (different encoder layout).
- Export: the v9 eval forward is plain convs/matmuls + constant affines — the Hailo-legal op set (depthwise convs, 1×1 convs, sigmoid/tanh/cos LUT forms) — with ONE exception, inherited from v8: **SlimContext's weave softmax (3 scalars per frame) is the D17 CPU stage.** The Hailo graph builder must split at the three gated-pool states exactly as v8 split at Path B's states: NPU computes the pooled states, the weave + softmax + mix run on the Pi CPU in fp32 (microseconds on 3×34 floats), and the mixed context vector re-enters the head's context matmul on-CPU too (the head fc1 is per-frame for the context half — trivially CPU). The monolithic ONNX export (one Softmax node) is for ORT/local eval, not the DFC.
- The v8 code paths are intact behind `arch="v8"` for ablation. `anet/metal.py` (D35) and `scripts/profile_step.py` remain v8-only.

### 14.5 v10 — training fixes (D49–D53)

The first from-scratch v9 run learned features (mean mannequin softmax prob climbed 0.03→0.29) but the argmax ran an all-foreground↔all-background limit cycle: `argmax_fg` swung 0↔185k every 1–2 epochs while the loss slid down monotonically ("cheating the loss"). An 8-agent static audit (single-step CPU gradient probes, no training) found the causes:

**D49 — focal_norm background normalizer (THE oscillation).** `focal_norm_loss` normalized each foreground class by its own cell count (a stable per-class mean) but the background term by `n_fg` — the batch's *foreground* count. Background is ~99.9% of every grid, so that one class-foreign, batch-varying denominator was the entire fg-vs-bg balance. Measured on two prediction-identical batches (a 3-cell mannequin vs a 600-cell tent): background loss swung **79.6×** and per-bg-cell gradient **~75×** purely from which object the sampler drew. A big-object batch muted the corrective background pushback ~75×, an over-prediction excursion ran unchecked, then a small-object batch swung the correction ~75× harder and collapsed the head. Fix: normalize **every** class (background included) by its own cell count — `L = Σ_c w_c · Σ_{t=c} FL / max(N_c, floor_c)`. Measured swing after the fix: **1.00×**. This restores the RetinaNet/CenterNet property (fg:bg pull batch-invariant) while keeping per-class size-invariance.

**D50 — peak LR (amplitude).** The cosine schedule stretched over 40 epochs leaves LR at ~100% of peak for the whole early window where the swing lived; with AdamW the clip value is provably irrelevant when it always binds (scale-invariance), so LR is the real step-amplitude knob. Peak 3e-3 → 1.5e-3.

**D51 — aux probe dropped.** The train-only aux linear probe (D46) was measured to contribute **0.02%** of the encoder gradient (the hard-loss path dominates ~4000× because focal gradient doesn't vanish on a confidently-wrong class), so it never achieved its "gradient path a collapsed head can't block" goal — while its private weights fitting themselves were ~19% of the logged loss, decoupled from detection (part of why loss fell as metrics oscillated). Off by default (`ANET_AUX=1` re-enables).

**D52 — ctx_norm noise.** `RegionHeadV9.ctx_norm` normalizes one d-vector per image, so it sees only `B` samples/channel — ~20,000× fewer than every other DeployNorm — and at momentum 0.05 its running stats random-walk ~4% every step from pure sampling noise, folded into a scale/shift added identically to all ~5,035 windows (a globally-coherent logit wobble, the exact shape of the 0↔185k flip). Its momentum alone → 0.01; seeding raised 8→24 to cover the cumulative-average ramp.

**D53 — speed.** The primary lever is the fused Triton kernel itself (it runs ~96% of the compute — the encoder); the fixes below make it compile on ROCm and keep the residual work cheap. (1) **Fused kernel on ROCm:** `tl.math.tanh` is absent on gfx942's Triton build — replaced with `2·σ(2x)−1` (exact, sigmoid is supported); the `tl.dot(allow_tf32=False)` blur calls already compile (the tanh error surfaced after them). A fused *backward* crash now demotes only the backward to chunked-autograd (keeping the fast forward and large batch) instead of collapsing to dense/batch-32, with full-traceback diagnostics. (2) `EdgeDQStem4`'s 4 separate `groups=3` depthwise 7×7 convs (88.6% of grouped-conv FLOPs at full 540×960) fused into one `groups=12` conv — bit-identical (max Δ 0.0), one MIOpen dispatch. (3) `samples_per_epoch` was uncapped on ROCm (full ~13.5k → ~60 min/epoch); `ANET_SAMPLES=6000` gives feedback ~2.3× faster (pure sampler lever). (4) `MIOPEN_FIND_MODE=NORMAL` was tried to avoid FAST's workspace-starvation fallback but reverted — this container's MIOpen can't open/write its SQLite perf DBs, which NORMAL hard-requires (`miopenStatusInternalError`); once the fused kernel handles the encoder, MIOpen's algorithm choice on the cheap residual convs is largely moot and the freed VRAM lets FAST get its workspace.
