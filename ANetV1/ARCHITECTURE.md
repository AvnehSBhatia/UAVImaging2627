# ANetV1 — Full Architecture Specification & Design Record

**Status:** v13 final · 2026-07-17 — **the §10 decision is executed: YOLO26n flies at SUAS 2026; ANetV1 is the research track** (§16.2: train-split eval proved the 25k-param model *underfits its own training data* — a capacity ceiling, not a data/loss/training problem). v6 was the locked baseline spec; v7/v8 (D31–D38) fixed the recall collapse and MI300X throughput; v9–v11 (D39–D56) rebuilt the training path and losses; v12 (D57) replaced per-cell classification with an object center-heatmap readout; v13 (D58) replaced the window-token encoder with a plain multi-scale conv backbone — the best model of the family (test mannequin 0.835 / tent 0.967 at 25,212 params, 1,132 img/s); v14 (D59–D63) structured priors were falsified against the capacity ceiling. §15 is the v12/v13 model; §16 is the v14 record and the closing verdict. **§17 is the v22 redesign (2026-07-19, D72–D75): peak-augmented full-rank funnel growth of v13_best** — the full-record redesign (evidence audit → design panel → red team), built and locally validated (identity 0.0, both gates PASS, 1.051× v13 latency at 3.1× params); MI300X run pending.
**Task:** object center detection of {mannequin, tent} in UAV survey frames (SUAS-style search area; per-cell region classification through v11, center heatmaps since v12)
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
14. [v9 — training-stack rebuild](#14-v9--training-stack-rebuild-d39d48)
15. [v12/v13 — center-heatmap detector and the conv backbone](#15-v12v13--object-center-heatmap-detector-and-the-conv-backbone-d57d58)
16. [v14 — structured priors as a monotone extension](#16-v14--structured-priors-as-a-monotone-extension-d59d63)
17. [v22 — grown, not retrained: peak-augmented full-rank funnel growth](#17-v22--grown-not-retrained-peak-augmented-full-rank-funnel-growth-d72d75)

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
| v11 | **weighted FP/TP loss** (D54, replaces focal_norm); **dynamic-box pooling** (D55, gate-mass-normalized Stage-1 readout); **metric-prototype head + `proto_metric_loss` pretraining** (D56, deploy-folds to one conv, +1 param). See §14.6. | v10 fixed the argmax limit cycle but exposed class asymmetry: tent recall 0.999 while mannequin stayed 0 and its softmax prob *decayed* 0.37→0.20 — small objects averaged away by the 400-px pool, and the FP/TP ratio never forces the true class to win the argmax |
| v12 | **Object center-heatmap readout** (D57): single-phase stride-20 Stage 1 (27×48 grid), CenterHead with two independent per-class sigmoids + sub-cell offset, CenterNet penalty-reduced focal + offset L1, peak/object metrics, soft-signal selection. See §15.1. | the per-cell softmax formulations kept re-creating the same tug-of-war failures (D47→D49→D54 lineage); the task is object-level, and independent sigmoids remove class competition structurally |
| v13 | **Plain multi-scale conv backbone replaces the window-token encoder** (D58): stem s2 → dw-sep s4 → dw5×5 s5 → 3× dw5×5 residual blocks → 1×1 head; DeployNorm + SiLU throughout; Kaiming init; ~25.2k params. Same D57 readout/targets/loss/metrics. See §15.2. | v12 on MI300X plateaued at soft p(center)≈0.09 at BOTH 1.5e-3 (monotonic crawl) and 3e-3 (oscillation) — pinpoint: tile-local pooling caps object-vs-background embedding separation at ~0.05; the identical targets are learned by bare logits (16/16) and by a small plain CNN in seconds |
| v14 | **Identity-init structured priors** (D59–D63): residual dw7×7 noise filter; 5× dual-quaternion channel-group shifts (fold to grouped 1×1); texture-energy sigmoid masking as a **bounded** weighted sum (unbounded draft collapsed the first run — see D61); max-pool s4→s20 detail skip; zero-gamma 4th block. 36.1k params; a v13 checkpoint warm-starts v14 to bit-exact v13 (smoke-asserted). See §16. | trained v13 vs corrected YOLO26n baseline: worst-decile mannequin 0.643 vs 0.857, fp/img 2.15 vs 0.018; viz decomposition: FPs cluster on canopy texture at prob 0.30–0.60, misses are diluted (heat 0.2–0.3) not absent, clutter hits under-confident |
| v15 | **SPD/full-rank funnel + capacity tiers** (D64–D65): `pixel_unshuffle(5)`+1×1 (≡ full k5s5 conv — see the D64 honesty note) replaces the depthwise funnel; tiers 74k/170k; per-layer LR for the funnel (μP-style); compile off on ROCm (miscompiles to step-0 NaN). See §16.3. | §16.2 underfit verdict + YOLO26n weight anatomy: 73.7% of its params deep, never strides >2×, and v13 funnels all fine evidence through one 2,048-param strided pipe |
| v16 | **Auxiliary cosine-weave texture channel** (D66, user-directed): v13 trunk + 1,800-param spatial multi-cosine weave over s4 texture energy, bounded gate, D24-bounded frequencies. 27,012 params. Identity-at-init vs pinned v13_best = 0.0. See §16.4. | single-variable test of the texture hypothesis; falsifier = fp/img at held recall. **Verdict: falsified** (test fp 2.082 vs donor 2.147, inside scatter) |
| v17 | **PowerBlend A^v injectors** (D67, user-directed): learned 3×3 exponent-rate matrix over chromaticity — `out_j = Σ_i relu(exp(W_ij·v_i) − τ_j)` — injected at four stage boundaries through zero-gamma valves; any D65 channel plan (scaled-v13 donors warm-start bit-exactly). +~900 params. See §16.5. | owner-directed; attached to the BIG tier per the §16.3 reopening condition. Judged strictly on the delta vs the plain tier at matched training |
| v18 | **Exposure-mask + bg-mask aux heads** (D68, user-directed): dual-exposure front (shared weights, +1.5 stops) blended at s4 by a state-vector-driven mask through a tanh-bounded valve; train-only background head with a smoothness prior. +603 params (25,815). See §16.6. | from-scratch scaling dead (0-for-6); owner thesis: capacity via auxiliary heads. Falsifiers: worst-decile (exposure mask) and fp at held recall (bg head) vs v13_best |
| v19 | **The attribution build** (D69): every mechanism valved — A bias injectors (the autopsy-endorsed worker + built-in §16.5 control), B owner's LearnedAct (bounded learned-SiLU + Gaussian bump, one LUT/site), C owner's 4-bump exposure head from a stolen micro-latent (+1.5-stop cap, D17 CPU micro-stage), D QuatShift + bg-aux. 27,111 params. Post-training per-valve ablation assigns credit. See §16.7. | v17 autopsy: wackiness unused, bias recalibration + training signals are the measured winners — v19 tests owner asks and evidence-endorsed mechanisms under one attribution harness |
| v20 | **Re-render cycles** (D70, owner-directed): both v13 stage transitions become embed→unembed pairs — lossless `pixel_unshuffle` + 1×1 funnel into an E=16 latent (LearnedAct) then cheap 1×1 expansion to a fresh full-width visual (identity-init QuatShift remix). Stem/block4/blocks/head keep v13 shapes → partial warm start (58 of 82 donor tensors land). 37,236 params. See §16.8. | owner pivot after v16–v19: stop bolting modules ONTO the trunk, restructure the trunk's transitions themselves — the two strided convs are where v14's diagnosis located evidence dilution (D62) and where v15 measured the funnel's leverage |
| v21 | **Two-stage filter front-end** (D71, owner spec): line-sampled mean RGB conditions three `A^chan` 11×11 kernels (thresholded, L1-normed, per-sample conv); mean-RGB MLP weights blend the 3 filtered images; smoothing quat (dedicated bg-TV loss) → edge quat + Sobel-init 7×7; saliency = pooled edge energy → center focal; peaks → **literal 100×100 crops** → 5.1k-param CropCNN → {BG, mannequin, tent}. 5,633 params total, no dense classifier (owner call). See §16.9. | owner direction — a detect-then-classify architecture with the feature extractor almost entirely hand-designed (~540 front-end params); evaluated on the same CenterObjectMetrics ladder as v13–v20 |

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

### 14.6 v11 — small-object rebuild (D54–D56)

v10's fixes stopped the argmax limit cycle but exposed a *class-asymmetric* failure: under the D54 loss below, **tent** recall reached 0.999 while **mannequin** stayed 0 and its softmax prob actively *decayed* (0.37→0.20) as tent sharpened — the two share one foreground detector and tent, being large and easy, wins every ambiguous cell. Two root causes, two mechanisms, both keeping the deploy-form (conv-only, affine-foldable, no attention/softmax) intact.

**D54 — weighted FP/TP loss (`fp_tp`, replaces focal_norm).** focal_norm's background term still coupled fg-vs-bg pull to batch composition and drove the head to predict-nothing (measured p(fg) 0.1→0.01). Replaced by one per-image, per-class **soft FP/TP ratio** `L = Σ_c w_c · mean_img[(FP_c + s)/(TP_c + s)]`, weights (bg 0.05, mann 0.8, tent 0.15). The `+s` in the *numerator* is the anti-collapse property: at the all-background point `dL/dTP = −1/s` (a real recall pull on true cells) and `dL/dFP = 1/s`, so predicting nothing costs ~Σw instead of 0 — collapse is no longer a zero-gradient fixed point. Per-image mean makes a 3-cell mannequin frame pull like a 600-cell tent frame. **Structural limit (why D56 is needed):** a soft-mass ratio never forces the *true class to win the argmax* — it is satisfied by p(mann)=0.2 as long as tent's own ratio is fine, which is exactly how the tiny mannequin lost.

**D55 — dynamic-box pooling.** The Stage-1 readout was `avg_pool2d(gate·h, 20)` — it divides the gated sum by the fixed 400-px window area, so a 2–8-px mannequin (0.5–2% of a window) is averaged ~50–90× into the background while a window-filling tent survives. The sigmoid cosine-gate (D10/D42) already computes a soft, data-dependent object mask per window; v11 reads out the **mean inside that soft box** — `Σ(gate·h)/(Σgate + ε)`, ε=0.5 in the sum domain — instead of the area mean. A mannequin covering 1% of the window is now recovered at full strength; a uniform-gate window is unchanged (mass-mean = area-mean), so tents and the cold start are undisturbed. Preserves the cosine-gated-pooling mechanism exactly (only the normalizer changes); stays two avg-pools + a divide (Hailo-legal). Applied bit-consistently across the token reference, dense path, chunked mirror, and the fused Triton fwd/bwd (`D = Σgate + ε`; `∂/∂gate_i = (A_i − Σ_j gate_j A_j /D)/D`), parity-checked at startup.

**D56 — metric-prototype head + pretraining.** The head's final `Linear(width,3)` is replaced by a **distance-to-prototype** readout in the bounded Tanh metric space z: `logit_c = scale·(2 z·p_c − ‖p_c‖²)`, which is exactly `−scale·‖z−p_c‖²` up to the class-independent `+scale·‖z‖²` (drops under softmax/argmax), so it **folds to one conv at export** — no runtime L2-norm, still affine-foldable. Net cost vs the linear head: **+1 param**. The point is not capacity (a linear layer has the same freedom) but that the classifier weights *are* the class prototypes, shaped by `proto_metric_loss`: a class-balanced prototype cross-entropy over `softmax(−scale·‖z−p_c‖²)` on 2×2-priority-pooled window labels, plus a prototype-separation push `mean exp(−‖p_i−p_j‖²)`. This supplies the missing "true class must win here" per-cell signal, in the same geometry the head decides in. Runs jointly (weight 0.5) from step 0 so the embedding clusters continuously; `loss_mode="metric_only"` runs a dedicated embedding-pretraining phase (detection off) to warm-start via `ANET_INIT_FROM`. All existing novel mechanisms — EdgeDQ oriented-edge stem, dual-quaternion colour rotations, cosine-gated mixing, DeployNorm, ConvNeck, Path-A, the multi-cosine SlimContext weave, per-cell region marking — are unchanged.

---

## 15. v12/v13 — object center-heatmap detector and the conv backbone (D57–D58)

Driver: through v11 every generation changed the loss or the head, and every from-scratch run kept converging to the same place — the rare tiny mannequin never wins. v12 changed **what is predicted** (object centers instead of per-cell classes) and v13 changes **what predicts it** (a plain conv pyramid instead of the window-token encoder). v13 is the current default; the D57 readout, targets, loss, and metrics carry over from v12 unchanged.

### 15.1 D57 — object center-heatmap readout (v12, kept by v13)

The per-cell {nothing, mannequin, tent} softmax is replaced by **CenterNet-style center detection** on the 27×48 stride-20 grid (`V12_H/W`; 540 = 20·27, 960 = 20·48):

- **Two independent per-class sigmoid heatmaps** (mannequin ch0, tent ch1) — no softmax competition, so a large easy tent can never eat the mannequin's gradient (the v11 D54/D56 failure, removed structurally instead of re-weighted).
- **Class-agnostic sub-cell (dx,dy) offset**, sigmoid-bounded to [0,1), supervised only at exact center cells (`offset_l1` + reg_mask).
- **Targets** (`rasterize.boxes_to_heatmap`): Gaussian center splats, σ=1.5 cells, max-merged across objects; the ring around a peak gets a reduced negative penalty via `(1−target)^β`.
- **Loss** (`center_focal_loss`, α=2, β=4): penalty-reduced pixel focal normalized by exact-peak count, `pos_weight` (default 3, `ANET_POS_W`) up-weighting the ~1-cell positive term against ~2,590 background cells.
- **Eval/selection** (`CenterObjectMetrics`): 3×3 local-max peak finding; object recall keys unchanged; best.pt/early-stop selection adds the threshold-free soft p(center-on-GT) signal so sub-0.5 learning is visible to selection.
- **Init**: RetinaNet prior p=0.01 on both center channels (measured: at 0.1 the shared bias sinks faster than the head can lift true centers).

### 15.2 D58 — plain multi-scale conv backbone (v13, replaces window-token Stage 1)

**The failure it fixes.** Every ANet generation v6–v12 pooled each 20×20 tile into ONE embedding vector with a tile-local encoder *before* any spatially fine learned feature extraction. A 15–30 px mannequin is 2–8% of a tile's 400 tokens, and the only pre-pool features were the stem's fixed-init edge channels — so its evidence was averaged into the tile summary almost untouched. The evidence chain, in order of discovery:

1. v12 pinpoint diagnostic: true-object windows separate from background by only **~0.05** in the normalized 32-d embedding; the deep head downstream is starved at the source.
2. The loss is exonerated: bare-logit optimization localizes 16/16 peaks; a small plain CNN learns the identical targets easily.
3. Two MI300X v12 runs hit the same **soft p(center) ≈ 0.09 ceiling** at two different LRs (1.5e-3: monotonic crawl 0.01→0.063 over 24 epochs; 3e-3: fast climb to 0.09 by epoch 5 then 10 epochs of oscillation with zero net gain) — an architecture ceiling, not a training-dynamics problem.
4. v13 overfit gate (12 real synthetic frames, 400 steps, **~13 s** on an M-series Mac): **19/21 GT centers past 0.5**, passing straight through the same ~0.09 level v12's full-scale runs never escape. Honest control: v12 *with the current training fixes* (pos_weight 3, σ 1.5, prior 0.01) also reaches 19/21 in this 12-frame harness — in **867 s** (~65× the wall-clock; the earlier constant-output stall was measured on the pre-fix config). So the overfit gate alone does not separate the architectures; the separation is (1)–(3) above — the ~0.05 embedding ceiling and the ~7,000-step full-scale plateau at two LRs — plus the 65× step cost of the token encoder for the same result.

**The fix** (`anet/model/backbone.py` `V13Backbone`) — learn features at fine stride first, summarize later:

```
960×540×3
├─ stem   conv3×3 s2  3→16   + DeployNorm + SiLU        (16, 270, 480)
├─ down4  dw3×3  s2 + pw 16→32  (DN+SiLU each)          (32, 135, 240)
├─ block  dw3×3  s1 + pw 32→32  residual                (32, 135, 240)
├─ down20 dw5×5  s5 + pw 32→64  (DN+SiLU each)          (64, 27, 48)
├─ 3×     dw5×5  s1 + pw 64→64  residual                (64, 27, 48)
└─ head   1×1 64→width + SiLU + 1×1 width→4             (4, 27, 48)
          channels: [center_mann, center_tent, dx, dy]  (D57 contract)
```

| Block | Params (width=24) |
|---|---|
| stem + DN | 464 |
| down4 | 752 |
| block@s4 | 1,440 |
| down20 | 3,040 |
| 3 × block@s20 | 17,856 |
| head | 1,660 |
| **Total** | **25,212** |

Design points, each load-bearing:

- **Kaiming (variance-preserving) init is a functional requirement, not a nicety**: DeployNorm normalizes with *running* stats (D39), so a net whose activations shrink ~10× per stage under torch's default init puts every norm's cold start ~300× off its fixed point — the 8 sequential seeding passes cannot relax a 10-norm cascade that far (measured: 1e23 logits on the first train step). With unit-variance propagation, seeding converges in a couple of passes.
- **No coordinate channels** — a center detector should be translation-equivariant; "what it looks like", not "where it is".
- **No global context path** — a per-frame global vector added identically to every cell is exactly the shortcut a collapsing head hides in (D52's noise mechanism, v12's constant-output basin). Receptive field is local-but-large: the three dw5×5 blocks alone give ±6 cells (±120 px) on top of the 100 px stride-5 window and the fine-stride stages — ~250–300 px per cell at 150 ft GSD.
- **No aux probe** — the gradient path is 8 convs deep; deep supervision was a workaround for the encoder starving the head, and the encoder is gone.
- **Deploy-legality strictly improves**: conv / affine-foldable DeployNorm / SiLU (single LUT, same as every YOLO the DFC compiles) / residual add. The D17/§14.4 CPU stage disappears — v13 is a single-shot NPU graph with host-side peak-finding only. The signature mechanisms v6–v12 preserved (DQ rotations, cosine gates, gated pooling, the weave) were Hailo-legality *workarounds* for attention-like computation; a conv pyramid needs no workaround.
- **What stays**: DeployNorm semantics and trainer contract (seeding + deferred EMA updates), the D57 readout/targets/loss/metrics, weight EMA + soft-signal selection, the <40k param budget (25.2k used).

### 15.3 Training configuration (v13 defaults)

Identical trainer path to v12 (`loss_mode="center"`): AdamW lr 1.5e-3 (3e-3 measured unstable — see the v12 LR history in §15.2 point 3), cosine over 80 epochs + warmup, `center_pos_weight` 3.0, σ 1.5, prior 0.01, weight EMA, soft-signal selection/early-stop. The fused Triton Stage-1 (D40) does not apply — every v13 op is a native cuDNN/MIOpen conv; there is nothing to fuse. Checkpointing is never engaged (activation footprint is small). `from_state_dict` sniffs v13 by the `backbone.` key prefix; v8/v9/v12 checkpoints still load for evaluation but cannot warm-start v13.

---

## 16. v14 — structured priors as a monotone extension (D59–D63)

Driver: the first trained v13 (25.2k params) measured against the corrected YOLO26n baseline (§10 decision metric): mannequin recall 0.837 vs 0.962, **worst-decile 0.643 vs 0.857** (21 pts behind the ≤15-pt bar), fp/img 2.15 vs 0.018. The 24-frame `runs/viz` stage dump decomposed the gap into three failure modes; v14 adds one targeted, deploy-foldable mechanism per mode — and reintroduces the project's signature dual-quaternion machinery in the one form that survived every prior post-mortem: constant-foldable, identity-initialized, and cheap.

**The D63 contract (the design rule that governs all of it):** every v14 module is identity- or zero-gamma-initialized, and v14 preserves v13's module names for the shared trunk — so a v14 warm-started from a v13 checkpoint computes *exactly* the v13 function at step 0 (`smoke_test` asserts max output delta < 1e-5; measured 0.0). Training can only move away from a proven optimum, never re-roll it. Zero-init is applied to *valves* (per-channel gains) rather than conv weights wherever a DeployNorm sits behind the branch: zeroing the conv would park the norm's running_var at ~0 and fold a ~√(1/ε)≈316× amplifier onto the branch exactly as it wakes (measured on the first draft); with zero-gamma valves every conv keeps Kaiming init and every norm observes real activation stats from the first forward.

| D | Component | Failure mode it targets (measured) | Cost |
|---|---|---|---|
| D59 | **Learned 7×7 noise filter** — residual depthwise conv on RGB before the stem, zero-init | FP band: sensor grain/texture aliasing enters the stem unfiltered | 147 |
| D60 | **5× dual-quaternion shift** (`QuatShift`) — per-4-channel-group Hamilton rotation + dual-part translation, one layer after each stage; folds to constant grouped 1×1 convs (the D5 bake, generalized) | clutter under-confidence: dw-sep stacks under-mix channels; norm-preserving rotation structure at 8 params/group | 416 |
| D61 | **Texture masking as a bounded weighted sum** (`TextureGate`) — learned high-pass (D32-init) → energy (square) → pooled to s20 → sigmoid mask g; trunk modulated `y·(1 + tanh(w_gate)·g)`, w_gate zero-init. **Bounded on purpose:** the unbounded first draft (`w_pass + w_gate·g`) gave the optimizer a whole-trunk multiplier, and the first from-scratch MI300X run used it as the global FP kill-switch — mann_r 0.52→0.009 across epochs ~13–20 with train loss near-flat, recovery only as LR decayed, early-stop at 24. The bounded factor ∈ (1−g, 1+g) ⊂ (0,2) keeps per-channel suppression expressible and trunk shutdown unrepresentable. | the fp/img 2.15: false peaks at prob 0.30–0.60 cluster on canopy texture — objects must beat the *local* texture floor, not an absolute bar | 2,032 |
| D62 | **Peak-preserving detail skip** — max-pool(5,5) of the s4 map → 1×1 → DN → zero-gamma gain, added to the s20 trunk | worst-decile misses: missed mannequins peak at heat 0.2–0.3 — evidence *diluted* by the strided-conv average over its 100-px window, not absent; max keeps the brightest 4-px response alive | 2,240 |
| D63 | **Zero-gamma 4th s20 block** + the identity-init contract itself | clutter discrimination capacity (hits at 0.35–0.55 in clutter vs 0.85–0.98 clear-ground) | 6,016 |

**Total: 36,063 params** (v13's 25,212 + 10,851), inside the <40k budget. Everything folds: quaternion algebra → constant grouped 1×1 convs (bake at export like `DualQuaternionRGB.to_conv`), gates → conv+sigmoid+mul (the D10 idiom), skip → max-pool+1×1, noise filter → one depthwise conv. No data-dependent normalization, no attention, no CPU stage.

Training: unchanged v13 recipe (§15.3). Two entry points: from scratch, or `ANET_INIT_FROM=<v13 ckpt>` — the trainer transfers all shared tensors and reports the new-at-identity count; warm-start begins at the donor's exact metrics. `from_state_dict` distinguishes v14 by the `backbone.noise.weight` key.

What would falsify each piece (pre-registered, in the §9 ablation spirit): D61 fails if fp/img does not drop at matched recall; D62 fails if worst-decile recall does not move; D60/D63 fail if the gains stay at ~0 (the valves report their own uselessness); D59 fails if the learned kernel stays ~0. Each is independently removable — they are additive valved branches, not rewires.

### 16.1 Run record and the frozen-stats correction (2026-07-17)

Three MI300X v14 runs, three distinct findings:

1. **From-scratch, unbounded gate**: collapse via the whole-trunk multiplier (fixed — D61 is now bounded).
2. **Full-tune warm-start, bounded gate, 5e-4**: epoch 0 reproduced the donor exactly (D63 held on hardware: sel 1.700), then val degraded monotonically to ~0.8 by epoch 14 while train loss *fell* 1.42→1.18 — the new capacity fits train in ways that do not generalize.
3. **Adapter (`ANET_FREEZE_TRUNK=1`), first attempt — INVALID**: sel 1.691→1.712 (ep2) then collapsed to ~0.26 with train loss *RISING* from epoch ~9 — impossible for overfitting, diagnostic for **function drift**: the freeze pinned donor *weights* but left donor DeployNorm *stats* live, and the trainable modules sit upstream of frozen norms (noise→stem_norm, qshift_i→next stage), so the stats chased the adapters' distribution shifts while the frozen weights could not re-adapt — a feedback loop with no restoring force. Fix: `DeployNorm.frozen` pins the stats; the freeze path now freezes donor weights *and* donor stats (verified: zero donor-buffer drift under adapter training). General DeployNorm lesson: **frozen weights require frozen stats whenever anything trainable sits upstream.**

Status: `train_anet.py`'s default arch is **reverted to v13** (the proven model); v14 is opt-in via `ANET_ARCH=v14`. The corrected adapter run is v14's remaining clean shot — with donor weights+stats pinned, the donor function is a true fixed point, and the run either beats the donor's sel or falsifies the D59–D62 priors on this data. (Superseded by §16.2: the capacity verdict makes the adapter test moot.)

### 16.2 The capacity verdict — §10 decision executed (2026-07-17)

The one measurement the v14 arc was missing was taken last: **object recall on the training split** (`evaluate_all --split train --limit 4000`):

| | train | test |
|---|---|---|
| mannequin recall | 0.828 | 0.835 |
| worst-decile | 0.586 | 0.595 |
| fp/img | 2.16 | 2.71 |

**The generalization gap is zero — the student underfits its own training data.** The tiny/occluded mannequins in the worst decile are seen thousands of times with exact labels and still cannot be fit at 25k params / stride 20. This closes every remaining mitigation in one stroke: more data cannot help (nothing to generalize better), distillation cannot help (GT already supervises perfectly — a teacher's soft target adds nothing the student isn't already failing to fit), further training cannot help (measured three times: fine-tunes shuffle the operating point, test recall is invariant at ~0.83). It also retro-explains v14: structured priors could not fix a representational ceiling, and their extra capacity had nowhere useful to go, so it went to train-specific fitting.

**§10 decision, per the pre-registered rule** (15+ points behind on worst-decile after mitigations → fly YOLO26n): final standing is 0.586–0.643 vs 0.857 — **the SUAS 2026 flight model is YOLO26n** (ONNX via `scripts/train_export_yolo26n.py`, Hailo DFC compile as the remaining step). ANetV1 continues as the research track: 25,212 params / 3.3 ms / 1,132 img/s at 0.835/0.95 recall is a legitimate efficiency-frontier artifact, and the one open, well-posed follow-up is a **capacity scaling curve** — relax the (self-imposed) 40k budget stepwise (50k → 100k → 200k) and measure where train-split decile recall lifts off; that curve is both the honest characterization of this architecture family and the natural spine of any write-up.

### 16.3 YOLO26n weight anatomy → v15, the scaling-curve architecture (D64–D65)

A weight-level study of the trained YOLO26n (the model that *does* fit this data) asked: where does its capacity live, and is either model saturated?

| finding | number | implication |
|---|---|---|
| YOLO backbone+neck params at stride ≤8 | 4.8% | small-object power is NOT fat early layers |
| …at stride 32 | **73.7%** | it is deep semantics… |
| …dedicated stride-8 head branch | ~21k params (P3 cv2+cv3) | …fused into a **fine detection grid**; YOLO never strides by more than 2× |
| effective rank (95%-energy / full), 1×1 convs | YOLO 0.67 vs v13 0.70 | *neither* saturated by this probe — blind width scaling has weak support |
| prunable norm gammas (\|γ\|<0.1) | YOLO 0.0%, v13 0.3% | no dead capacity anywhere; capacity must be *placed*, not just added |

Combined with §16.2 (underfit) and the §15.2 miss anatomy (worst-decile heat 0.2–0.3 = diluted, not absent), the indictment lands on one component: **v13's `down20` funnels every fine-scale feature through a 5×-strided depthwise average and a single 2,048-param 1×1** — a compression YOLO's architecture never commits (max stride step 2×, fine grid kept for detection).

**D64 — SPD projection.** `down20` is replaced by `pixel_unshuffle(5)` + learned 1×1: the s4 map (ch_mid, 135, 240) is rearranged **losslessly** to (25·ch_mid, 27, 48) — every s4 pixel's features arrive at the detection grid intact — and the 1×1 (51k params at tier S) learns what to keep. This is SPD-Conv (Sunkara & Luo, 2022: strided convs/pooling destroy small-object evidence), and it is Hailo-native (space-to-depth, the YOLOv5 Focus layer; ONNX SpaceToDepth).

**Honesty note (added after the first tier runs):** `pixel_unshuffle(5)+1×1` is *mathematically identical* to one full (non-depthwise) `Conv2d(ch_mid, ch_top, 5, stride=5)` — same weight count, reshuffled. The real content of D64 is therefore **full-rank vs depthwise-separable** projection at the funnel: v13's down20 constrained the s4→s20 map to (depthwise 5×5) ∘ (rank-ch 1×1); D64 lifts that rank constraint with ~25× the parameters in that one layer. The capacity claim stands; the "lossless" novelty framing was oversold. The same fact explains the tier runs' early turbulence — one layer holding ~70% of all parameters moves the function per step far more than anything in v13, hence the lowered LR, longer warmup, and the fp-gated selection.

**D65 — the pre-registered curve** (relaxed budget per §16.2; `ANET_CH`/`ANET_BLOCKS`/`ANET_PARAM_BUDGET`):

| tier | config | params | isolates |
|---|---|---|---|
| origin | v13 | 25,212 | — |
| v15-S | defaults (16,32,64)×3 | ~73.5k | the SPD projection alone |
| v15-M | ANET_CH=16,48,96 ANET_BLOCKS=4 | ~170k | + width/depth (YOLO's deep-heavy ratio) |

Verdict keys, committed in advance: (1) **train-split worst-decile recall** — where it lifts off is the capacity the task needs (if v15-M still underfits, the stride-20 grid itself is the binding constraint and the next move is a finer grid, i.e., a target-contract change); (2) **the canopy FP band** — if it persists once the model fits its training data, the texture-prior question (v14, falsified at 25k) honestly reopens, with capacity to spend. The auxiliary cosine-weave texture channel proposed at this stage is deferred on those grounds, not rejected: priors were falsified *at the capacity ceiling*; they get re-examined only after the ceiling moves.

### 16.4 v16 — the auxiliary cosine-weave texture channel (D66, user-directed)

Built at the project owner's direction ahead of the §16.3 reopening condition, as the **single-variable** test the texture hypothesis deserves: `arch="v16"` = the v13 trunk, bit-identical module names, plus ONE module (`CosineWeaveTexture`, **1,800 params** — total 27,012, inside the *original* 40k budget).

**Pre-registered expectations:** per §16.2, recall lift is *unlikely* (the worst-decile misses are objects v13 cannot represent, and 1.8k params don't change that); but the canopy FP band (false peaks at 0.30–0.60 on texture) is a decision-boundary problem, so **fp/img reduction at held recall is the plausible win and the falsifier**: if fp/img does not drop at matched recall, D66 is falsified and the texture hypothesis is closed at this capacity for good.

**Mechanism** — the project's signature multi-cosine weave (D44 idiom), spatial and deploy-legal: s4 texture energy (D32-init high-pass, squared, DN) → pooled to the s20 grid → `tanh`-bounded states → two-harmonic cosine bank (frequencies held to one LUT period by the existing D24 `l2_score_reg` hook via `reg_l2`) → sigmoid mask → **bounded modulation** `y·(1 + tanh(w_gate)·g)` (the D61 lesson: trunk shutdown unrepresentable). Every hard-won contract applies: identity-at-init (measured 0.0 against the pinned `v13_best.pt`), norms observing real stats from step 0, warm-start + frozen-trunk adapter support (v13→v16 transfer inherits scaled-v13 channel plans).

**Measured before shipping:** from-scratch overfit gate is ~2× slower to converge than v13 (2/21 at 400 steps) but passes at 800 with **the lowest background contamination of any arch in the harness** (max bg prob 0.227 vs v13's 0.54, v14's 0.41) — the exact behavioral signature the module was proposed to produce, in miniature. The recommended experiment is therefore the corrected adapter: warm-start from `v13_best.pt`, freeze trunk+stats, train the 1,800 weave params; the run either beats the donor (sel > ~1.72, or equal recall at lower fp) or D66 is cleanly falsified.

**Verdict (adapter run, 2026-07-17): falsified.** The corrected adapter harness worked flawlessly (epoch 0 = donor exactly, sel 1.712; fp drifted 2.26→2.18 over the first four epochs then flattened; train loss fell throughout — the weave fit *something*, but not the thing that matters). Final checkpoint vs donor: test fp/img **2.082 vs 2.147** (−3%, inside the 2.1–2.7 eval scatter), mannequin 0.832 vs 0.837, worst-decile 0.571 vs 0.643 (one object of fourteen), tent 0.943 vs 0.940; train split equally unmoved (2.126/0.828). The pre-registered falsifier fires: **at 25k capacity, texture-conditioned modulation cannot buy a meaningful fp reduction at held recall.** Interpretation consistent with §16.2: the gate can only reweight features the frozen trunk already computes, and separating canopy texture from mannequin texture evidently requires *representation*, not reweighting — even the FP band is capacity-coupled at this scale. The texture hypothesis is now closed at this capacity, measured three independent ways (v14 full-tune degradation, the §16.2 capacity argument, and this clean single-variable adapter); §16.3's reopening condition (a scaled model that fits its training data yet still shows the FP band) is the only path back.

### 16.5 v17 — PowerBlend A^v injectors (D67, user-directed)

The owner's op, from the broadcasting identity `A^v` (row i of a learned 3×3 matrix raised to the power of chromaticity component v_i), thresholded and column-summed: `out_j = Σ_i relu(exp(W_ij·v_i) − τ_j)` — a learned power-law activation over normalized RGB, sparsified by a learned threshold. Deploy-legal end-to-end: chromaticity bounds the exponent, W is D24-held (`reg_l2` → `l2_score_reg`) with a clamp(−4,4) saturation, threshold = bias+ReLU, column sum = constant conv. Injected at four stage boundaries (post-stem, post-s4, s20 entry, pre-head) through 1×1 projections behind zero-gamma valves; ~900 params on the big tier.

**Placement per the record:** D66 closed texture-style priors at 25k; §16.3's reopening condition permits testing priors on a model whose capacity ceiling has moved — so v17 attaches to the **big tier** (24,48,96)×4 ≈ 63k, and to keep the capacity-curve datapoint unconfounded the judgment is strictly **v17-at-tier vs the plain tier at matched training**: any recall/fp delta outside eval scatter, or D67 joins D66. Contracts verified before shipping: identity-at-init 0.0 against a scaled-v13 donor, PowerBlend init math (W=0 → A^v=1), exp-clamp overflow safety, valve wake-up (W sits at the reg minimum at exact identity — silent until the gains crack, live one step later), from-scratch gate PASS 21/21 at 800 steps (the familiar ~2× valved-injector slowdown).

**Verdict (adapter run, 2026-07-17): marginal pass — the first positive in the module ledger.** Same-protocol ladder (identical donor `v13_best`, identical frozen-trunk harness, identical 5e-4): donor test fp/img 2.147 → v16 2.082 (inside the sibling band, falsified) → **v17 1.955** — the first sub-2.0 checkpoint in the project and below the full sibling span (2.08–2.71) — at flat recall (mannequin 0.835 vs 0.837, tent 0.946 vs 0.940). Worst-decile 0.571 (−1 of 14 objects, within sibling span). Judgment per the pre-registered rule: the fp delta is outside checkpoint-selection variability, so **D67 stands, provisionally** — pending (a) the `peak_thresh` sweep (`evaluate_all --peak-thresh`) confirming the operating curve dominates the donor's rather than sliding along it, and (b) ideally a second seed. If confirmed, v17 becomes the ANet research-track deploy candidate; the §10 flight decision (YOLO26n, fp 0.018) is unaffected.

**Weight autopsy (2026-07-17, local; all 928 delta params read out).** Trunk verified bit-identical to the donor (82/82 tensors) — the entire fp gain lives in the injectors. Three findings:

1. **The A^v color function went unused.** Learned exponent rates are tiny (|W| ≤ 0.036 — the `exp` never leaves its linear regime), thresholds barely moved from init, and the injection magnitude is *flat across the entire chromaticity simplex* (dynamic range ×1.0 at every site). The module learned an (almost) input-independent, per-channel **bias injection** — ~176 effective bias adjustments to the frozen trunk at four depths — not color reasoning.
2. **The effect is real and well-targeted anyway**: on the 24-frame probe set, donor FP peaks moved **−0.022** on average (41 of 72 pushed down by >0.005) while donor TP peaks moved **+0.008** — suppression concentrated on false peaks with true peaks slightly *strengthened*. A constant bias achieves this differential effect through the frozen trunk's nonlinearity, not through the module's own selectivity.
3. **Site attribution (per-site gain ablation): pb1 (post-stem) carries half the FP suppression** (−0.011 of −0.022), pb2 a quarter, pb3/pb4 nearly nothing — despite pb3 having the largest raw injection norm. The earliest, highest-resolution site is where recalibration bites.

**Pre-registered control this demands:** a bias-only adapter (per-channel learned biases at the same four sites, no PowerBlend — ~176 params) trained in the identical harness. If it matches fp ≈ 1.95, the mechanism is "multi-depth recalibration of a frozen trunk" and the A^v machinery is ballast; if it falls short, the residual chromaticity linearity earns the credit. Either answer sharpens D67 into an attributable claim.
### 16.6 v18 — exposure-mask + background-mask auxiliary heads (D68, user-directed)

Two owner-directed auxiliary heads on the untouched v13 trunk (+603 params, 25,815 total), after two from-scratch big-tier runs failed to converge at any LR (0-for-6 for from-scratch-at-scale across this project; function-preserving widening remains the parked capacity path):

- **State-driven exposure mask** ("add ~1.5 stops to selected areas"): the trunk front (stem/down4/block4, shared weights) runs on the image and on a 2^1.5× brightened copy; a mask head — a GAP state vector (JEPA-*style* latent, explicitly not JEPA training) biasing a local 1×1 head — blends the two s4 maps through a tanh-bounded scalar valve, identity at init. Mechanistic target: the worst-decile canopy/shadow mannequins. The bright pass runs with the front DeployNorms frozen (`DeployNorm.frozen`) so its statistics never contaminate the deploy normalization — smoke asserts the front pendings equal a normal-branch-only pass.
- **Background-mask aux head** (train-only, dropped at eval/export like D46): 1×1 → bg logit per cell off the pre-head features, loss = BCE against 1−max-Gaussian + the owner's smoothness prior on the *predicted* background (deviation from a 3×3 local mean; `bg_aux_weight` 0.3 / `ANET_BG_W`, `bg_smooth_weight` 0.3). Mechanically distinct from the falsified D61/D66/D67 gates: a training signal that shapes trunk features, not inference machinery that reweights them.

Pre-registered falsifiers, judged against `v13_best` on test at matched fine-tuning: (a) worst-decile mannequin recall (the exposure mask's one job) outside ±1 object; (b) fp/img at held recall (the bg head's). From-scratch overfit gate: PASS 21/21 at 800 steps with max background prob **0.125 — a new harness record** (v16: 0.227, v13: 0.54); the bg-aux signature, unprompted.

**Verdict (full fine-tune from `v13_best`, 2026-07-17): split.** Test: mannequin 0.818, tent 0.932, **fp/img 1.722** — the family record (−20% vs donor, −36% vs the matched full-fine-tune control `v13_ft` at 2.708) — worst-decile 0.571. (a) **The exposure mask is falsified on its pre-registered axis**: worst-decile 0.571, unmoved — and notably this is the *third consecutive* checkpoint (v16, v17, v18) at exactly 0.571, i.e., the same ~8/14 immovable tail objects; the +1.5-stop mechanism did not recover a single shadowed mannequin. (b) **The bg-aux head's fp record comes with a recall dip** (0.837→0.818), so whether v18's operating curve *dominates* the donor's or just slid along it is exactly the `--peak-thresh` sweep question — unresolved pending that table (donor vs v17 vs v18 at 0.30–0.50). The training-signal-vs-gating distinction survives either way: the two largest fp movements in the family (v17's bias recalibration, v18's bg supervision) both came from mechanisms that alter *features or calibration*, not inference gates. Contracts verified: identity-at-init 0.0 vs the pinned `v13_best`, train/eval `aux_bg` contract, DN bright-pass isolation, blend valve live at identity, full gradient flow, sniff via `backbone.mask_out.weight`.


### 16.7 v19 — the attribution build (D69, owner + evidence co-designed)

Response to the owner's v19 directive, reconciled with the v17 autopsy (which showed the *opposite* of "wacky input manipulation helps": the A^v color function was unused; plain bias recalibration did the work). v19 therefore packages every mechanism behind **its own valve** so one training run + per-mechanism ablation — the autopsy method, now built in — assigns credit, and the §16.5 bias-only control is *inside* the model:

- **A — bias injectors** (176 params, zero-init) at four stage boundaries: the mechanism the autopsy identified. Ablating B/C/D post-training yields the pre-registered bias-only control for free.
- **B — LearnedAct** (owner): `x·σ(βx) + γ·exp(−(x−μ)²/2σ²)` per layer, parametric identity at init (β=1, γ=0 ≡ fused SiLU up to ~1e-5 kernel rounding, tolerance documented in smoke). One Hailo LUT per site; 4 scalars × 8 sites.
- **C — ExposureBumps** (owner, replacing v18's mask): a micro-encoder on the 8×-pooled image steals a latent; a small head emits **4 normalized (x,y) + per-bump exposure** (sigmoid-capped at +1.5 stops); Gaussian bumps applied to the *input*, tanh-valved, clamped to [0,1]. Deploy: D17-style CPU micro-stage (~1.6k params).
- **D — QuatShift post-stem** (owner invitation, reused from D60, identity-init) + the **v18 bg-aux head** (the family's fp-record earner; train-only). Cosine thresholding deliberately omitted: D66 closed the cosine machinery with a clean instrument.

27,111 params (v13 + 1,899), inside the original budget. Contracts: identity-at-init vs the pinned donor at 9.5e-6 (fused-vs-unfused SiLU rounding; asserted < 5e-5), all four mechanisms gradient-live at the identity point, ExposureBumps clamped under extreme valves, sniff via `backbone.bumps.head.weight`. Judgment: fine-tune from `v13_best` at 5e-4, then (1) test table vs the family ladder, (2) the built-in ablation — zero each valve, re-eval — to attribute whatever moves.


Gate record: the first v19 gate run NaN'd from scratch — at the identity point the new params see gradients in the 1e2–1e3 range (`qshift.qr` 2.1e3, `act` γ 3.5e2) and the original unbounded β/γ flew before any schedule could react. All LearnedAct parameters are now **bounded by construction** (β ∈ (0.5,1.5) via tanh, γ ∈ (−0.5,0.5), μ ∈ (−1,1), σ ≥ 0.25 — identity at init preserved exactly, and stricter LUT hygiene as a bonus); the re-run passes 21/21 at 800 steps. Same family lesson, third appearance: every scalar that multiplies or reshapes the trunk must be bounded by construction (D61 gate → D69 activation), not by hope.


**D69 verdict (adapter run, ANET_BG_W=0): falsified, informatively.** Same harness that produced v17's 1.955: v19 (A+B+C+D) scored test 0.809/0.951, fp **2.450**, decile 0.524 — worse than the donor on every axis and 0.5 fp worse than bias-alone. Mechanism interference is real: adding the LearnedAct, exposure bumps, and quat shift ON TOP of bias injectors undid the bias win. Family law, third confirmation: bias recalibration and training signals are the only measured positives; every additional input/feature-manipulation mechanism has measured zero or negative. The v19 attribution question answered itself — the stack lost to its own subset.

### 16.8 v20 — re-render cycles (D70, owner-directed)

**Owner direction (2026-07-17, after the v19 verdict):** "conv → embed →
then maybe unembed into a completely diff visual → conv, and repeat this
instead. embedding should be aided by successful components. unembed is
just expanding it back up cheaply."

**D70.** Both of v13's strided transitions (`down4` k3s2, `down20` k5s5)
are replaced by an explicit embed → unembed pair:

```
conv stage ──► EMBED: pixel_unshuffle (lossless, D64) → 1×1 funnel → E=16
              latent → DeployNorm → LearnedAct (bounded, D69-B)
          ──► UNEMBED: 1×1 E→C cheap expansion → DN → SiLU →
              identity-init QuatShift remix (D60/D63)
          ──► next conv stage … repeat
```

Cycle 1: s2→s4 (`embed1` 64→16, `unembed1` 16→32). Cycle 2: s4→s20
(`spd_proj` 800→16, `unembed2` 16→64) — the funnel keeps the `spd_proj`
NAME on purpose so the trainer's slow-LR group (0.2×, the measured v15
stability fix for exactly this fan-in-800 shape) matches it. 37,236 params
(budget-legal). ROCm: same pixel_unshuffle shape family as v15 → compile
defaults OFF for v20 too; warmup 600.

Why the bottleneck is the mechanism: E=16 ≪ 64 (cycle 1) and ≪ 800
(cycle 2) forces every transition to RE-ENCODE — the next stage sees a
freshly rendered visual, not a strided copy. This attacks the two
locations where the family's diagnostics actually pointed: D62 located
worst-decile evidence DILUTION at the s4→s20 stride, and v15 measured
that same funnel as the highest-leverage (and touchiest) tensor in the
net. Unlike v16–v19 this is not a module ON the trunk — it is the trunk's
transitions, rebuilt from measured-good parts only (D64 lossless descent,
D69-B bounded LearnedAct, D60/D63 identity QuatShift, D58 Kaiming, DN).

Warm start is PARTIAL by construction (no D63 identity contract): stem,
block4, the three s20 blocks and the head keep v13 shapes, so 58 of the
donor's 82 tensors land via strict=False; the 24 transition tensors are
dropped and the 36 new tensors start Kaiming. Full fine-tune only —
ANET_FREEZE_TRUNK would strand fresh transitions between frozen stages
(the v14 adapter failure mode, mirrored). Falsifier, same ladder as
v16–v19: test recall/fp vs v13_best 0.837/0.940/2.147 and worst-decile
vs the immovable 0.571–0.643 band — a transition rebuild that cannot move
the decile confirms the capacity verdict from yet another angle.

Run: `ANET_ARCH=v20 ANET_INIT_FROM=runs/anet/v13_best.pt python
scripts/train_anet.py` (lr auto-capped 1.5e-3, warmup 100 on warm start;
from-scratch fallback is legal — v13 itself trained from scratch at this
scale). **Status: built, smoke-passed (partial-transfer + sniff-order +
all-live-grads asserted); MI300X run pending.**

### 16.9 v21 — the two-stage filter front-end (D71, owner spec)

**Owner direction (2026-07-17):** mean-RGB from 20 random rows + 20 random
columns; three learned 11×11 matrices raised elementwise to the R/G/B
means (`A_k^chan`, colors normalized [0,1]) with a learned set-to-zero
threshold; triplicate the image through the three kernels; mean-RGB
through 3→8→SiLU→8→3 for blend weights → one composite; a quaternion
trained on a separate background-smoothing loss; a second quaternion as a
learned Sobel; find object centers on the filtered image (center loss),
expand **literal 100×100 crops**, classify each with a very small CNN
(BG/tent/mannequin). Dense-conv equivalent explicitly declined.

**D71 — implementation choices** (`anet/model/twostage.py`,
`scripts/train_twostage.py`, `runs/twostage/`):

- `A_k = exp(W_k)` so `A^chan = exp(chan·W)` is positive and
  differentiable; exponent clamped ±4 (D24 discipline applied to
  exponents — v17's exact parametrization). Threshold = `relu(· − τ)`
  (v17's form of "below n → 0"), then L1-normalized for scale stability.
  Per-image kernels run as one grouped conv with per-sample weights.
- Blend MLP last layer zero-init with bias 1/3: the composite starts as
  the plain average of the three filtered images.
- The smoothing quaternion is pointwise; its dedicated loss is mean
  |x − avgpool3(x)| of the composite on background cells (mask from the
  GT heat). It sits in the main path, so main-task gradients also reach
  it — a STRICTLY separate loss for an in-path module would require
  cutting the main gradient. Recorded, not hidden.
- A pointwise quaternion cannot BE a spatial Sobel (zero spatial
  extent), so quat #2 feeds a Sobel-init 7×7 depthwise kernel — the
  D5/D33 EdgeDQStem pattern: the quaternion picks WHICH colour axis the
  edge operator sees.
- Saliency = channel-L2 of the edge image, max-pooled 20×20 to the
  family 27×48 grid, affine-calibrated, trained with `center_focal_loss`
  (class-agnostic: max-over-class Gaussian targets).
- Stage 2 is the literal owner spec: 3×3 local-max peaks (top-12) →
  100×100 crops from the edge image (PatchCrops clamp geometry) →
  CropCNN (3→8→16→24 strided, GroupNorm — crop batches are small and
  variable — GAP → 3). Crop training set per step: GT-centered crops
  (their class), 2 random bg crops/img, ≤4 unmatched predicted peaks as
  hard negatives.
- Eval runs the REAL deploy path (peaks → crops → classify) and writes
  each detection's class prob at its peak cell into a family
  (heat, offset) pair, so CenterObjectMetrics and the v13–v20 ladder
  numbers are directly comparable.

Params: **5,633** total (front end ~540, CropCNN ~5.1k) — the smallest
model in the family by 4×. Deploy caveats recorded up front: per-image
kernels are dynamic conv weights (not Hailo-compilable as-is; the 16.8
basis-expansion `K0 + chan·K1` fix applies if it ever earns deployment)
and the crop gather is a CPU stage. Pre-registered falsifier, same ladder
as v16–v20: test recall/fp vs v13_best 0.837/0.940/2.147; the structural
risk is proposal recall — stage 2 can never recover an object the
~540-param front end fails to peak.

**Status: built; smoke + 1-epoch micro-run pass** (all-live grads, crop
CE cold-starts at ln 3, center focal 18→12 in 3 steps, detect contract
emits family tensors). MI300X run pending:
`python scripts/train_twostage.py` (knobs: ANET_LR 1.5e-3, ANET_EPOCHS
15, ANET_BATCH 16, ANET_SMOOTH_W 0.1, ANET_CACHE=1 recommended).

**v21.1 (owner-directed revision, same day):** the epoch-0 viz split the
blame cleanly — saliency peaks were landing ON objects (frame 000008's
single peak was the mannequin at CropCNN p 0.71) while the classifier
starved: it saw only the 3-channel edge image, discarding color (the
family's strongest class signal) and every other computed map. Owner:
"the issue is the crop messing up — try something better that uses all
of our info." CropCNN now takes the 9-channel window stack (raw RGB +
smoothed composite + edge) plus a 4-scalar context vector into the head
(the peak's saliency prob + the frame's mean RGB — stage 1's confidence
and the scene stats that conditioned the kernels). 6,077 params (+444).
Same viz also showed 18/24 frames peaking BELOW the 0.3 threshold →
center focal now uses pos_weight=3 (ANET_POS_W), the v12-measured fix
for exactly that slow positive climb.

---

## 17. v22 — grown, not retrained: peak-augmented full-rank funnel growth (D72–D75)

**Status: built, smoke-passed (full-identity contract 0.0), both overfit gates PASS, throughput falsifier fired-and-fixed pre-training (measured 1.051× v13) — MI300X run pending.** Produced 2026-07-19 by a full-record redesign campaign: a 200-finding evidence audit over D1–D71 + the probe/two-stage/runs/git record, a five-design panel (fine-grid FPN, unified two-stage, function-preserving growth, speed-first reflow, novel-signature peak descent) scored by four independent judges, and a six-lane adversarial red team (param math, MACs/latency, Hailo legality, falsified-collision, trainability, novelty-vs-literature) on the merged draft before a line of model code was finalized. One red-team blocker and four majors were found and resolved below — two of them by *measurement*, before any training was spent.

### 17.1 The thesis, from the record

Three measured facts compose v22, and none is new — the design is their intersection:

1. **§16.2**: v13 underfits its own training data (train 0.828/0.586-decile ≈ test 0.835/0.595). Data, distillation, and more training are measured-closed; *representational capacity, correctly placed*, is the only open lever.
2. **§16.3**: the indicted site is `down20` — all fine-scale evidence funnels through a depthwise-5×5 average + one 2,048-param 1×1, a rank constraint YOLO26n (worst-decile 0.881) never commits. The D64/D65 full-rank fix was pre-registered but **never successfully trained**: from-scratch-at-scale is 0-for-6 in this project, and no tier run ever completed.
3. **D62**: worst-decile misses peak at heat 0.2–0.3 — evidence *diluted by strided averaging*, not absent. A full-rank **linear** projection restores rank but cannot represent a **max** statistic, so the D64 fix is incomplete by construction; peak evidence needs its own nonlinear path.

v22 therefore **grows v13_best function-preservingly** instead of retraining at scale:

```
x20 = down20(x_s4)                                        [donor, bit-exact]
    + 2·tanh(spd_gain) ⊙ SiLU(DN( spd_proj(x_s4)          [D72: full-rank capacity]
                                + peak_proj(maxpool5×5s5(x_s4)) ))   [D73: peak path]
```

`spd_proj` = Conv2d(32→64, k5, s5) — by the D64 honesty note this **is** pixel_unshuffle(5)+1×1 over 800 channels, implemented as the fused conv so no 800-channel intermediate is ever materialized (a measured −0.5 ms/frame, and it removes v22 from the ROCm pixel_unshuffle inductor-miscompile family entirely: compile stays ON, unlike v15/v20). `peak_proj` (1×1 on the max-pooled s4 map) is algebraically the 32 extra concat columns, split out as its own tensor so the peak mechanism's post-training column autopsy is a single weight-norm readout. The valve is tanh-bounded ×2 — the third-time law (D61 gate → D69 activation → here), applied because this one gain gates 68% of all new capacity and an 80-epoch drift must be *unrepresentable*, not unlikely (red-team blocker, resolved). Plus v18's train-only `bg_head` (65 params, dropped at export; run-1 trains with `ANET_BG_W=0` — see 17.4).

**78,717 params (78,652 deployed)** = donor 25,212 + spd_proj 51,200 + peak_proj 2,048 + spd_norm 128 + spd_gain 64 (+ bg_head 65 train-only) — on the pre-registered D65 curve at the tier-S point. **216.7M MACs** (1.47× v13's 147.7M; ~0.05% of Hailo-8 int8 peak — [§1.3] compute headroom is not the constraint). Every op is from the sanctioned set: conv/dw-conv, max-pool, affine-foldable DeployNorm, single-LUT SiLU/tanh, residual add. Single-shot NPU graph; host-side peak-finding unchanged from v13.

### 17.2 D72 — capacity as valved growth (the warm-start law)

The 0-for-6 from-scratch record and the v14 full-tune failure bracket the training problem: capacity added from scratch never converges here, and capacity bolted on as small priors overfits. The untested corner is *function-preserving growth of the proven checkpoint*. Because the branch is parallel and zero-valved, `ANET_INIT_FROM=v13_best` lands **all 82 donor tensors bit-exact — weights AND every DeployNorm running-stat buffer** (legal precisely because no donor module's input distribution changes at step 0; contrast v20's partial 58/82 start, where copied stats downstream of a fresh funnel would have been stale). Step-0 output delta vs donor: **0.00e+00, smoke-asserted**. `spd_norm` is fresh and observes real branch activations from the first forward (valve after the norm — the D63 zero-gamma idiom, never a zeroed conv). Gains/valve get gradient at the identity point; branch weights wake one optimizer step later (smoke-asserted live).

Training contract: **full fine-tune only** — `ANET_FREEZE_TRUNK` is *refused* in `train_anet.py` for v22 (a frozen trunk around a fresh funnel is the measured 16.1 collapse). lr 7.5e-4 peak (funnel-dominant law, v15-measured), warmup 600 even on warm start, `spd_proj` auto-matches the 0.2× slow-LR group by name; `peak_proj` (fan-in 32) deliberately trains at full LR — the tiny new mechanism should not be slowed. From-scratch is legal-but-**discouraged**: the 800-step gate passes (below), but nothing licenses a from-scratch run at 79k params against the 0-for-6 record; it exists for the gate and for emergencies, not as a plan.

### 17.3 D73 — the peak side-channel, and what the red team removed

The draft carried the same peak idea at `down4` (tanh-gated maxpool blend) and four standalone bias-recalibration tensors (v17's D67 carry). Both were **removed by audit**, for independent reasons worth recording:

- **down4-peak was unlicensed**: D62/D64 indicted the s4→s20 funnel specifically; `down4` strides 2× — already YOLO-anatomy-compliant — and no finding flags it. Placement without a measured failure mode is how v14–v19 died.
- **"~0 MACs" ≠ free**: the red team *measured* the draft's elementwise adds over the s2/s4 maps (2.07M/1.04M elements) at **+19.7% wall-clock — more than the entire 69M-MAC funnel branch (+10.7%)** on the eager batch-1 protocol the throughput falsifier is judged on. Zero-MAC ops on big maps are the dominant hidden cost on a launch-bound model. (This generalizes D38: dispatches and memory passes, not MACs, are the budget.)
- **Bias sites are adapter-regime machinery**: v17's bias win was earned *on a frozen trunk*, where biases were the only degrees of freedom. In a full fine-tune every DeployNorm bias is already trainable — standalone bias tensors are redundant dof at real dispatch cost. **D74 protocol, not architecture**: the bias-recal experiment remains available post-training as a bias-only adapter phase (train only DN biases on the frozen result — zero new params), directly comparable to v17_best.

What remains of D73 is the funnel peak channel itself, with its pre-registered attribution: nearest prior art is **YOLOv9's ADown** (parallel avg/max downsampling branches, concat-fused) and the UAV small-object dual-pooling line (DFAS-YOLO/DPNet), with mixed/gated pooling (Lee et al. 2016) the older ancestor and SPD-Conv (Sunkara & Luo 2022) already in-family via D64. The architectural primitive is **not** novel and the record should never claim otherwise. The novel content is methodological: (a) *peak-vs-rank as separately falsifiable variables at a measured failure site* — a plain-SPD sibling (drop `peak_proj`) isolates whether max statistics buy anything beyond rank, the attribution every v14–v19 module lacked until post-hoc autopsy; (b) *capacity growth executed under the full D63 identity contract* — nearest prior art Progressive Networks (Rusu et al. 2016) / adapter-style PEFT (Houlsby et al. 2019), here inverted: the new capacity is grown to be *ablatable and bounded* inside a 79k-param deploy model.

### 17.4 D74 — proven-mechanism carry, isolation-first

Run-1 answers ONE question — does grown capacity + peak evidence move the immovable decile band? — so it runs `ANET_BG_W=0` (bg-aux off) and no bias adapters: the D69 interference law says proven single-variable wins do not compose additively, and the D65 tier design goal was "isolate the projection alone." The proven mechanisms then layer back in with their own controls: run-2 = +bg-aux 0.3 (judged against **v18_best** fp 1.722/recall-dip precedent), run-3 (optional) = post-training bias-only adapter (judged against **v17_best** 1.955). Both baselines already exist as checkpoints — attribution needs no extra runs.

### 17.5 The measured record (2026-07-19, local)

- **Smoke** (`scripts/smoke_test.py v22_checks`): 78,717 params; identity-at-init vs donor **0.00e+00**; donor-tensor accounting exact (82 land, 9 new tensors); valve-alive at identity; funnel+peak live after valve crack; ±1e6 valve collapse-safety; sniff roundtrip (v22 before v15 — both carry `spd_proj`; v22's unique key is `spd_gain`).
- **12-frame overfit gate** (from-scratch, 800 steps, lr 7.5e-4): 0/21 → 21/21 centers past 0.5 by step 700, max bg prob 0.262 at step 800 — PASS in the valved-arch class (v15-S 21/21@400, v16 pass@800; the ~2× wake-up lag is the documented valve pattern). Gate training throughput ~150 img/s MPS batch-12.
- **Warm gate** (v13_best growth, 300 steps): donor scores 12/21 on the gate set with max bg 0.866; v22 reaches **21/21 by step 100** and max bg **0.120** by step 300 — v18-record-class background suppression *with* recall rising, on 24 seconds of MPS fine-tuning. (Step-0 gate readout differs from the donor only because the harness's 8 seeding passes nudge donor DN stats toward the 12-frame distribution; the smoke test asserts exact identity without seeding.)
- **Throughput falsifier: fired, fixed, passed.** Draft architecture measured **1.30×** v13 batch-1 latency (both by this campaign's bench and the red team's independent reconstruction) — 3× over the pre-registered ~10% bound, before any training was spent. Cause (measured, not guessed): the 800-ch unshuffle+concat materializations and the big-map elementwise ops, not the 69M MACs. Fix: the fused-conv identity + the 17.3 removals. Final paired/interleaved bench (the protocol the falsifier is now defined by — naive sequential timing swings 2× thermally): **batch-1 v13 2.40 ms (412 img/s) vs v22 2.52 ms (393 img/s) = 1.051× [1.043–1.056]; batch-8 1.050×**. For scale: v22 remains **6.7× faster than YOLO26n** (16.88 ms) at 30× fewer params.

### 17.6 D75 — pre-registered falsifiers and the escalation ladder

1. **Capacity**: train-split worst-decile mannequin recall must clear the 0.586–0.643 band meaningfully (the §16.2 methodology: train split first). Stuck-in-band at fitted train loss → escalate to v22.1, per the honest §16.3 trigger.
2. **Peak-vs-rank control**: identical-tier sibling minus `peak_proj`. Indistinguishable → the peak channel is ballast; credit rank/capacity and say so.
3. **FP curve dominance**: any fp claim requires the 0.30–0.50 peak-thresh sweep vs donor — closing the open v16–v19 methodological gap; single-point comparisons don't count.
4. **Mechanism autopsy** (v17 method, designed-in): report |2·tanh(spd_gain)| per channel and ‖peak_proj‖/‖spd_proj‖ column norms at convergence; near-zero self-reports unused.
5. **Throughput**: paired/interleaved batch-1 bench within ~10% of v13 — **already measured PASS at 1.051×**; re-verify on the trained checkpoint (weights don't change latency, but the export path must stay clean).
6. **Generalization watch** (v14 run-2 lesson): val degrading while train falls = capacity landing wrong — stop and report, don't tune around it.

**Escalation ladder** (each staged, each with landed plumbing or named precedent): **v22.1** — per-class anisotropic readout: mannequin at stride-10 (54×96) via a raw pixel_unshuffle(10) tap + 2× **nearest** upsample (mode pinned now — DFC supports nearest only) of deep features, tent stays s20; the grid-parameterized `boxes_to_heatmap`/`SUASCells(center_grid=)`/shape-derived `CenterObjectMetrics` landed this session, so v22.1 is a model-only delta; triggered by falsifier 1 firing at fitted train loss. **v22.2** — cascade re-scoring on strip-pooled span/density features of the first-pass heat (the dense v21.5 chunk-shape lesson, from the panel's runner-up design); triggered by fp > 1.0 at matched recall after v22. **v22.3** — D65 tier-M growth of the same valved-branch form.

### 17.7 Findings disposition (the full-record audit, compressed)

| findings | disposition in v22 |
|---|---|
| §1/D22 input+grid physics; D10/D11/D15/D19/D24/D26 Hailo op law; §6 PCIe/DRAM | respected verbatim: 960×540 input, conv/pool/LUT/affine-fold graph, no attention/softmax/dynamic weights, raw-frame-only PCIe |
| D5/D33 bake idiom; D6/D24 bounded-argument law | carried: tanh-bounded valve folds to constants; no periodic args added |
| D23/D47/D49/D54/D57 loss lineage | untouched: center_focal + offset_l1, independent sigmoids, per-class n_pos normalization, pos_weight 3 |
| D31/D52/D58 context-dilution + no-global-context/no-coords | respected: no global path, no coord channels, branch is local conv/pool |
| D39/D48/D53 DeployNorm/EMA/infra contracts; 16.1 frozen-stats law | carried; freeze refused for v22; buffers transfer legally (17.2) |
| D46/D51 aux-probe falsification | respected: no linear-probe supervision; bg_head is the D68 training-signal class, not a probe |
| D58 conv base + Kaiming; D63 identity/valve contract | v22's foundation; full identity achieved (0.0) |
| D59–D63 v14 priors; D66 weave; D68 exposure; D69 stacking | all falsified machinery excluded; interference law → run-1 isolation |
| D64/D65 anatomy, SPD honesty, tiers, slow-LR, sel-gate | the capacity mechanism itself; honesty note *used* as the fused-conv optimization; tier-S sizing; slow-LR + max_sel_fp inherited |
| D67 bias autopsy; D68 bg-aux | demoted to protocol (adapter-regime insight) / staged to run-2 — with v17_best/v18_best as standing baselines |
| D70 v20, D71 v21.x (both untested), P1/P2 (abandoned) | not built upon; v22 is a third, independently-falsifiable line — v20/v21 remain the owner's open experiments (v21.2–21.5 history lives in `twostage.py`'s docstring, not yet in this file) |
| v21.4 attenuator; v21.4/21.5 crop cost; v21.5 threshold lessons | respected: no learned pre-filters ahead of evidence, no crops/gathers, no naive thresholds; max-pool cannot attenuate (parameter-free, parallel) |
| 16.2 capacity verdict; 0-for-6; paper_bench numbers | the design's premise, its training law, and its report protocol (synthetic-only slices, worst-decile with CI, curve sweeps) |

Run: `ANET_ARCH=v22 ANET_INIT_FROM=runs/anet/v13_best.pt ANET_BG_W=0 python scripts/train_anet.py` (80 epochs, cosine; selection/early-stop unchanged). Artifacts note: `runs/comparison.json` is a mislabeled v17 result and `runs/anet/log.csv` is from an abbreviated local run — judge v22 against `runs/paper_bench/` (the authoritative v13/YOLO26n record) and fresh evals only.

**Run record — MI300X run-1 (2026-07-19, killed at epoch 22 by owner): falsifier 6 FIRED.** Warm start verified on hardware (epoch 0 = donor-class: mann_synth 0.795 / tent 0.952 / fp 2.27 / sel 1.724; compile ON, 57–67 step/s, ~5.5k img/s train throughput at batch 16 × accum 6). sel peaked **1.736 at epoch 2 (mann 0.809), lr ≈ 5.4e-4, mid-warmup** — that is best.pt. From epoch 3 (lr ≥ 7.0e-4) val eroded monotonically-with-noise while train loss fell 1.43 → 0.98: mann to 0.61–0.67, **tent (the solved class) from 0.952 to 0.64–0.79**, soft probs falling in lockstep — real function degradation, the §16.1 v14-run-2 shape, exactly what falsifier 6 pre-registered. Two candidate diagnoses, one discriminator:

- **(A) capacity-overfit** (v14-run-2 redux). If confirmed, note the §16.2 nuance: "more data cannot help" was licensed by the *underfit* premise; a grown model that fits train but not val REOPENS data as a lever, and the v22.1 trigger's "at fitted train loss" clause must be re-read accordingly.
- **(B) LR protocol wrong for the warm-growth regime** — favored by the log: erosion onset is sharply LR-correlated (stable-to-up at ≤5.4e-4, eroding from the first ≥7e-4 epoch); the 7.5e-4 peak is the v15 *from-scratch* funnel law, while every successful post-donor run (v16/v17/v18 incl. v18's full fine-tune) used 5e-4; and the slow-LR group is INVERTED for this regime — the fresh branch trained at an effective 1.5e-4 (slowest in the model) while the converged donor took the full 7.5e-4 heat, the exact opposite of what the valve's function-space bound makes safe.

**Pre-registered discriminator (run-1b)**: `ANET_ARCH=v22 ANET_INIT_FROM=runs/anet/v13_best.pt ANET_BG_W=0 ANET_LR=4e-4 ANET_SLOW_MULT=1.0 python scripts/train_anet.py` — donor 1.9× gentler, branch 2.7× hotter, one variable (the LR protocol) changed. Same shape at 4e-4 → (A) confirmed at two LRs (the v14 evidentiary standard); write the falsifier-1/6 verdicts and escalate per the ladder. Holds/improves past 1.736 → (B): scope-correct the v15 LR law in this record ("from-scratch funnel dominance does not transfer to warm growth") and let the run continue to its real verdict. (Also confirm run-1 actually had ANET_BG_W=0 — if bg-aux was live at its 0.3 preset default, it is a second uncontrolled variable; v18 measured it recall-negative even when fp-positive.)

**Run record — run-1b, the discriminator (2026-07-19, early-stopped at epoch 25): hypothesis B FALSIFIED; A-vs-C pending.** At lr 4e-4 / ANET_SLOW_MULT=1.0 (donor 1.9× gentler, branch 2.7× hotter) the shape reproduced: epoch 0 donor-class (0.784/0.957/2.15, sel 1.707), best sel **1.710 @ epoch 4 ≈ donor — no val gain ever materialized in either run** — then the same both-classes erosion while train loss fell 1.41→0.99 (mann → 0.55–0.60, tent → 0.378 at ep21); early stop retired it at the min-epoch boundary. Two LRs × two slow-LR configs, same signature: the LR-protocol diagnosis is dead. Before the capacity-overfit verdict is written, a THIRD mechanism must be excluded, one the family has never probed:

- **(C) EMA-weights/live-stats mismatch under distribution drift.** ModelEMA shadows parameters only (D48); eval/checkpoints pair ~3.6-epoch-lagged EMA weights (0.998 ≈ 500-opt-step horizon) with LIVE DeployNorm buffers that chase the current raw-weight distribution at momentum 0.05. Sound when stationary — but v22's branch grows monotonically, so every eval runs old weights against new-distribution stats. Fits: local no-EMA gates improved cleanly while every EMA-evaluated run erodes; erosion is smooth and class-global (calibration-shaped); run-1b eroded LESS than run-1 at matched epochs early (ep9 0.761 vs 0.673, ep15 0.678 vs 0.618) — the LR×lag prediction. If confirmed, C also retroactively questions the v14-run-2 full-tune reading (same eval hybrid, same opening gates, never probed); the §16.2 verdict itself is safe (measured on a stationary converged v13).

**Pre-registered probes (on-box, minutes):** (1) train-split object eval of last.pt (§16.2 protocol) — A predicts high train recall/low val; C and loss-gaming predict both low; (2) the stat-reseed test — re-observe DN buffers under the checkpoint's own weights (~60 batches at momentum 0.05, then re-eval val): a substantial jump toward donor-level confirms C, and the fix is principled (re-seed stats before eval/checkpoint, or shadow the buffers in the EMA with the same debias ramp — a scoped D48 amendment for non-stationary growth regimes). No jump + high train recall → A confirmed at the v14 evidentiary standard: write falsifiers 1/6, note the §16.2 nuance (genuine overfit REOPENS the data lever), escalate v22.1 per the ladder. Also confirm both runs actually had ANET_BG_W=0 (the 0.3 preset default is a live confound otherwise).

**Fix record — the D48 amendment (2026-07-19, owner-directed "fix"):** `ModelEMA` now shadows every DeployNorm `running_mean`/`running_var` at the same decay + debias ramp as the parameters; `swap_in`/`swap_out` install/restore them symmetrically, so eval and `best.pt`/`last.pt` are internally consistent (weights and stats from the same ~500-opt-step window) even while the funnel valve shifts the trunk's distribution. `reset_buffers()` re-snapshots the shadows after `_seed_norm_stats` (the EMA is constructed pre-seeding). Preset `ema_norm_buffers=True`, escape hatch `ANET_EMA_BUFFERS=0` (pre-v22 parameters-only behavior); stationary regimes are first-order unaffected, so the family ladder's comparability survives. Validated: mechanics under synthetic valve drift (shadow diverges from live; swap/checkpoint/restore exact; off-switch) + an end-to-end micro-train through the real Trainer (seeding → epoch → eval → EMA-consistent checkpoint → `from_state_dict` round-trip). D48's original "parameters only, deliberately" note is amended in place (trainer.py) with the non-stationarity scope condition.

**Run-1c (the rerun that now discriminates A-vs-C):** same command as run-1b — `ANET_ARCH=v22 ANET_INIT_FROM=runs/anet/v13_best.pt ANET_BG_W=0 ANET_LR=4e-4 ANET_SLOW_MULT=1.0 python scripts/train_anet.py` — the fix defaults on. Erosion gone → C was the cause: the capacity/peak verdict (falsifiers 1–4) reopens on clean instrumentation, and the v14-run-2 full-tune reading deserves a retroactive footnote. Erosion persists → A at last measured trustworthily: write falsifiers 1/6, note the §16.2 data-lever reversal, escalate v22.1. The stat-reseed probe on the run-1/1b checkpoints remains the no-retrain shortcut to the same answer.

---

## 18. v23 — dual-grid anisotropy head: the mannequin-margin redesign (D76–D79)

Ground-up redesign inside the owner-chosen **≤40k envelope** (capacity explicitly off the table after v22's §17.5 erosion): fix the mannequin's discriminative margin through readout structure and feature type, not parameters. Driven by `runs/viz_web_scenes` — the trained v13/v22/YOLO26n run on realistic composites.

### 18.1 The diagnosis that forced it

The margin is **zero-to-inverted**. On the *easiest* case (`eval_open_easy_both`: a spread-eagle person on clean bare dirt, limbs plainly resolved, ~4×4 cells — not a resolution-starvation case) the person scores **0.10** while empty-corner background scores **0.33–0.36**. Elsewhere: v22 fires **0.50–0.58 on painted runway numbers** (worse than v13 there); a prone person in brush is missed while sagebrush fires; the raw heatmap never suppresses to zero (a low-level red field across the whole canopy). Tents meanwhile are fine (0.75–0.93, v22 0.93 beating YOLO's 0.47).

Two compounding causes: (a) at stride-20 a 49×13px person is a **1–2 cell point with no spatial support**, so a lone bright person-cell and a lone bright bush-cell are the same object to the head — whereas a tent is a 5×5-cell coherent blob whose neighbours co-activate, which is *exactly why tents work*; (b) the only evidence the trunk offers is brightness/edge **magnitude**, and every shape idea previously proposed is **2-way** (elongated vs round), which structurally cannot separate a person from a painted stripe because both are elongated.

### 18.2 D76–D79 — the design

- **D76 AnisotropyContrast**: two-scale structure-tensor coherence **contrast**, the family's first **3-way** shape feature. Fixed luminance+Sobel at s2 → J=[[Ix²,IxIy],[IxIy,Iy²]] → box-averaged at a limb-width (5) and a body-width (21) window → per scale (trace, eigen-gap). Coherent at fine but **not** coarse = person; coherent at **every** scale = paint/fence/shadow edge; coherent at **no** scale = canopy/brush. Eigen-gap via alpha-max-beta-min so **no sqrt and no divide** enters the deploy graph. DeployNorm(4) before the 4→8→1 MLP (the four J-statistics have wildly different natural scales; without it the sigmoid saturates at init — the D39/D58 cold-start law).
- **D77 per-class anisotropic grid**: mannequin read at **stride-10 (54×96)** off the s2 stem tap, *before* the s4→s20 funnel D62/D64 measured as diluting small-object evidence — a person becomes ~5×1.3 cells, so elongation is representable. Tent keeps stride-20.
- **D78 tent safety by construction**: trunk + tent head (**sliced from the donor's output rows [1,2,3]**) load from `v13_best` and freeze — weights **and** DeployNorm stats together (the D39/§16.1 law). Strictly stronger than v14's zero-init valve: a valve can drift under gradient pressure, a frozen parameter cannot.
- **D79 the margin metric**: `p(at GT centre) − max p(background)`, logged every epoch. Read at the GT centre rather than at a matched peak on purpose — it stays defined when the object is **missed**, which is the interesting case. This is the diagnostic recall/fp structurally hide.

33,119 params (25,187 frozen + 7,932 trainable), **6,881 under the cap**. All ops Hailo-legal; dual-grid output is structurally YOLO's own P3/P4 multi-scale head pattern. No pixel_unshuffle → **not** in the v15/v20 ROCm inductor-miscompile family.

### 18.3 Run record — run-1 (MI300X, early-stopped epoch 46): SPLIT verdict

| | epoch 0 | best | donor v13_best (val) |
|---|---|---|---|
| mannequin margin | **−0.178** | **+0.012** | ~−0.23 (easy case) |
| mannequin recall (synth) | 0.162 | **0.687** | ~0.795 |
| tent recall | 0.949 | **0.949 (byte-constant ×47 epochs)** | 0.949 |
| tent margin | +0.380 | **+0.380 (byte-constant)** | +0.380 |
| fp/img | 1.24 | 2.05 | ~2.15 |

**What passed.** (1) The margin **flipped sign**, monotonically, crossing zero at epoch 26: −0.178 → +0.012 is a **+0.19 improvement**, clearing the pre-registered ≥+0.10 falsifier. The 3-way coherence feature does move the quantity it was built to move. (2) **Tent safety is now measured, not argued**: recall and margin were byte-constant for 47 epochs — freeze-by-construction (D78) works exactly as specified, and this is the first mechanism in the family that provably cannot regress the working class.

**What failed.** Mannequin recall **0.687 vs the donor's ~0.795** — an ~11-point regression on the headline metric — with fp/img unimproved (2.05) and the absolute margin razor-thin (+0.012, versus YOLO's ~0.8 true / ~0 background separation). Directionally right, operationally worse.

**Diagnosis, from the log itself.** Soft p at GT centres saturates at **0.354** while best-background sits at ~0.342: nothing is being pushed *apart*, everything is being pushed *down*. Two causes, both cheap to test:

1. **The finer grid quadrupled the class imbalance.** s10 has 5,184 cells vs s20's 1,296, so the same ~1 positive cell now faces 4× more background in the negative term — while `pos_weight` was left at the family default 3.0. The design's own spec asked for 4.0 and even that is too timid against a 4× ratio change. This is the leading hypothesis and is a one-env-var test.
2. **The frozen 16-channel stem starves the branch** (the design's own pre-registered risk #1): 7,932 params must build a person detector from features trained for v13's objective, against a donor whose 0.795 came from a 25k trunk adapting end-to-end. Pre-registered fallback: unfreeze **stem+down4 only**, keeping every tent-critical downstream layer frozen with stats pinned per D39/§16.1.

**Pre-registered next runs** (one variable each, D69 law): **run-2a** `ANET_POS_W=8` — if soft-p and recall climb, the imbalance was the binding constraint and the mechanism is vindicated. **run-2b** (only if 2a is insufficient) unfreeze stem+down4. Still unmeasured and required before any verdict: worst-decile recall on `best.pt` (falsifier #2, ≥+0.03 over the 0.586–0.643 band), the peak-thresh sweep (falsifier #3), the mechanism autopsy, and — the falsifier that motivated the whole redesign — **falsifier #5: does the anisotropy map visibly separate person from paint/canopy on the 14 preserved `viz_web_scenes` inputs?** A metric win with no qualitative separation would mean the movement came from elsewhere.

### 18.4 Run-2a + the tail measurement — why margin is the wrong thing to optimize with features (D80)

**Run-2a (`ANET_POS_W=8`, early-stop ep31).** Mannequin recall **0.687 → 0.761** (soft-p 0.354 → 0.383) — so hypothesis 1 was real, positives *were* being outvoted 4× harder at s10. But the margin went **+0.012 → −0.037** (back negative) and fp/img **2.05 → 3.89**. Tent stayed byte-constant (0.949/+0.380) for a second run. Verdict: **pos_weight is a recall lever, not a margin lever** — it lifts foreground everywhere, background included. Runs 1 and 2a are two points on ONE operating curve, and v23 has ≈zero margin at either.

**Falsifier #5, measured locally with no checkpoint** (the coherence feature is fixed math — only the 4→8→1 MLP is learned, so the raw statistics bound what any downstream head can do). 168 mannequin centres vs background over 140 synthetic val frames:

| | mannequin | background | AUC |
|---|---|---|---|
| trace_fine | 0.246 | 0.080 | **0.731** |
| gap_fine | 0.062 | 0.018 | 0.693 |
| **D76 fine/coarse gap ratio** | **2.039** | **1.146** | **0.651** |
| 4-D logistic probe (ceiling for ANY downstream MLP) | | | **0.797** |

**The D76 premise is CONFIRMED directionally** — people genuinely are more fine-coherent-relative-to-coarse than background (2.04 vs 1.15), in the predicted direction, at AUC 0.65–0.80. (Methodological note, recorded because it nearly produced a false verdict: a first pass selected hard negatives *by* `gap_fine` and so reported AUC≈0.005 — an artifact of selecting on the tested variable. Unbiased random negatives give the table above.)

**D80 — the tail law, and why the mechanism still cannot work.** The margin is `GT-centre − max(background)`, i.e. an **extreme-value** statistic, while AUC is a **per-pair** statistic. Measured on the best single feature:

- **10.03%** of background locations exceed the *median* mannequin → **~13,000 locations per frame** out-score a typical person.
- Even at the 90th-percentile mannequin, 2.8% of background (~3,600 locations/frame) still scores higher.

A detector competes against the **max over ~130,000 background locations per frame**. To reach ~1 fp/frame the background tail must satisfy P(bg > object) ≲ 1e-5; the measured value is 1e-1 — **four orders of magnitude short**. This is why AUC 0.80 buys recall (run-2a) but zero margin, and it retro-explains the entire feature-mechanism ledger: D61 texture gate, D66 weave, D67 PowerBlend and now D76 all improved *average* separability and all failed to move fp/margin, because none of them touches the tail.

**The corollary is the tent result, inverted.** Tents work precisely because a 5×5-cell object requires many adjacent cells to agree, and requiring k-of-k co-activation suppresses the background tail multiplicatively. A mannequin at s20 is 1–2 cells and at s10 is ~5×1.3 — there is not enough spatial extent to buy the needed tail suppression from agreement alone. **So the binding constraint is not feature quality but evidence VOLUME per object**, and no single-cell feature — however clever, at any capacity ≤40k — can close a 4-order-of-magnitude tail gap. This is the legitimate negative result the redesign brief pre-authorized, now quantified rather than asserted.

**What this licenses next** (in order of evidence): (1) mechanisms that aggregate *multiple independent* evidence sources per object, since that is the only measured tail-suppressor in the family (the tent mechanism); (2) an honest write-up of D80 as the family's governing law — "per-pair separability is the wrong objective for a detector; report the background tail" — which is a stronger research contribution than another falsified module; (3) if the ≤40k envelope is ever reopened, the tail argument, not the AUC argument, is what sets the required capacity.
