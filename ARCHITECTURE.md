
### 20.5 Falsifier results — the numbers pass, the stated rationale does not

`d85_best.pt` (v13, 25,212 params — byte-identical arch to the donor) against `v13_best.pt`, test split, D82-corrected keys:

| falsifier | bar | v13 | D85 | verdict |
|---|---|---|---|---|
| 1. synthetic recall | ≥ ~0.80 | 0.837 | **0.882** | **PASS** — +4.5pt, family record |
| 2. real-scene gap closes | see below | — | — | **FAIL** |
| 3. throughput | within ~10% | — | 15–17 s/epoch, unchanged | pass (informal) |
| 4. worst decile (synth) | ≥ 0.643 | 0.643 | 0.571 | **inconclusive** |

**Falsifier 2 — measured, and it fails.** §20.3 pre-registered "object p should rise 0.482 → 0.570 and bg>0.30/frame fall 6.8 → 2.64." On the 14 real web scenes:

| | v13 | D85 | change |
|---|---|---|---|
| object peak (median) | 0.482 | 0.435 | **−10%** |
| **peak − p99 (separation)** | **0.270** | **0.241** | **−11%** |
| background p99 | 0.216 | 0.181 | −16% |
| background median | 0.026 | 0.015 | −44% |
| cells > 0.30 /frame | 6.79 | 6.50 | −4% |
| **cells > 0.50 /frame** | **0.79** | **1.21** | **+55%** |

Only the *quiet* part of the real-scene background improved. Separation did not — it fell 11% — and high-confidence background responses became **55% more frequent**. This is D80's own pattern recurring: **average separability improved and the tail did not.** The prediction was specific, it was wrong, and the mechanism story in §20 — that augmentation closes the sharpness tell — is **not supported**. D85's synthetic gain is most plausibly generic regularization of an under-regularized model, which is a real and valuable effect but not the one claimed.

D85 is kept on its measured merits (F1, fp, margin), with its rationale downgraded from *explanation* to *hypothesis that failed its own test*.

**Falsifier 4 is inconclusive, not a failure — and that is the bigger finding.** The synthetic worst decile holds **42 objects**; the D85–v13 gap is **3 of them**, Δ = −0.071 with bootstrap 95% CI **[−0.286, +0.143]**, straddling zero. The metric cannot resolve a 7-point difference, so it cannot distinguish any two revisions in this family.

That retroactively dissolves §16.6's "third consecutive checkpoint at exactly 0.571," which was read as ~8/14 immovable tail objects revealing something structural. It reveals nothing: it is a small-n metric quantized so coarsely that neighbouring revisions land on the same value by construction. **So D82's indictment extends — after correcting *which task* the decision metric measures, it turns out it also lacks the statistical power to compare revisions at all.** Any future verdict on the worst decile needs either a much larger synthetic test pool or a bootstrap CI reported alongside the point estimate.

**What remains open.** The real-scene gap is untouched. Augmentation was the cheapest lever aimed at it and it moved the wrong quantities. The objects in every training frame are still Blender renders composited onto real backgrounds, and no measurement in this session contradicts the §19.3 reading that object appearance is where the gap lives — augmentation simply does not fix that, because jittering a rendered object's exposure and sharpness does not make it a photograph of a person. The untried lever aimed directly at it is real object imagery.

### 21.2 The sweep, and D87 — fp/img had the D82 defect too

Running the sweep required looking at how fp is counted, and it has the same flaw D82 found in recall: **`fp_per_image` pools all frames**, and the test split is 1,267 VisDrone vs 449 synthetic. Measured at `peak_thresh=0.3`, v13: **pooled 11.995 vs synthetic 2.154** — a 5.6× difference, and the pooled figure is ~74% driven by a task the mission does not fly. (The record's historical "fp/img 2.147" matches the *synthetic* 2.154, so `benchmark_paper` was already reporting the right thing; it is `CenterObjectMetrics` — the trainer's per-epoch print and `evaluate_all` — that reports pooled.) **D87: `fp_per_image_synthetic` now ships alongside.**

**Operating curves, test split, synthetic fp** (10 thresholds, one forward pass per image with peak-finding re-run per threshold, written to `runs/thresh_sweep_test.json`):

| synthetic fp/img | v13 (25k) | d85 (25k) | **v22+D85 (78k)** | v22 − v13 |
|---|---|---|---|---|
| 0.5 | 0.715 | 0.817 | **0.842** | **+0.127** |
| 1.0 | 0.768 | 0.862 | **0.886** | **+0.118** |
| 2.0 | 0.830 | 0.900 | **0.924** | +0.093 |
| 3.0 | 0.855 | 0.916 | **0.938** | +0.082 |

Read the other way — **synthetic fp needed to hit a given recall**:

| recall | v13 | d85 | v22 | v13/v22 |
|---|---|---|---|---|
| 0.75 | 0.83 | 0.22 | **0.12** | **6.8×** |
| 0.80 | 1.37 | 0.42 | **0.24** | **5.6×** |
| 0.85 | 2.76 | 0.80 | **0.57** | **4.9×** |
| 0.90 | 13.00 | 1.99 | **1.27** | **10.3×** |

Worst-**quartile** synthetic recall (the powered key, §20.6) at matched fp: at 1 fp/img, v13 0.678 → **v22 0.789**. Tent at 1 fp/img: 0.895 → **0.962**.

**Dominance — the question §16.6 left open for v18 and never answered.** Interpolating each curve onto a common synthetic-fp grid:

- `d85` vs `v13`: recall delta min **+0.039**, median +0.045, max +0.116 → **dominates**
- `v22+D85` vs `v13`: min **+0.055**, median +0.073, max +0.182 → **dominates**
- `v22+D85` vs `d85`: min **+0.003**, median +0.019, max +0.067 → **dominates**, though thinly at the low-fp end

So both this session's changes are genuine curve *shifts*, not slides along a fixed trade-off — the first time any change in this family has been shown to dominate rather than trade. The total ordering **v22+D85 ≻ d85 ≻ v13** holds at every operating point measured.

**Session summary in one line:** at a fixed 1 synthetic fp/img, mannequin recall went **0.768 → 0.886** and worst-quartile **0.678 → 0.789**, from two changes — adding augmentation that had never existed, and re-running a capacity increase that had previously failed only because the augmentation was missing.
