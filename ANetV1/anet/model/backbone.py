"""V13Backbone — plain multi-scale conv backbone (v13, D58).

Why this replaces the window-token encoder (v6-v12 Stage 1), fundamentally:
every prior ANet generation pooled each 20x20 tile into ONE embedding vector
with a tile-local encoder BEFORE any spatially fine, learned feature
extraction happened. A ~15-30 px mannequin is 2-8% of a tile's 400 tokens,
and the only pre-pool features were the stem's fixed-init edge channels — so
the object's evidence was averaged into the tile summary almost untouched.
Measured consequence (v12 pinpoint diagnostic): true-object windows separate
from background by only ~0.05 in the normalized 32-d embedding; every deep
head downstream of that signal crawled or collapsed to constant output,
while the same targets/loss were learned easily by a small plain CNN and by
bare logits (16/16 peaks). The loss was never the problem; the encoder
destroyed the signal at the first pooling step.

The fix is the boring, proven shape: learn features at FINE stride first,
then downsample progressively, so a small object is many pixels wide when
the first learned filters see it and is only summarized after it has been
amplified into dedicated channels:

    (B, 3, 540, 960)
    stem   conv3x3 s2  3->16   + DeployNorm + SiLU      (B, 16, 270, 480)
    down4  dw3x3  s2  + pw 16->32   (DN+SiLU each)      (B, 32, 135, 240)
    block  dw3x3  s1  + pw 32->32   residual            (B, 32, 135, 240)
    down20 dw5x5  s5  + pw 32->64   (DN+SiLU each)      (B, 64, 27, 48)
    3x     dw5x5  s1  + pw 64->64   residual            (B, 64, 27, 48)
    head   1x1 64->width + SiLU + 1x1 width->4          (B, 4, 27, 48)

Output channels: [center_mannequin, center_tent, dx, dy] — exactly the v12
CenterHead contract, so losses (center_focal_loss/offset_l1), targets
(rasterize.boxes_to_heatmap) and metrics (CenterObjectMetrics) are reused
unchanged. 540/20=27, 960/20=48: the stride-20 output grid matches V12_H/W.

Deploy-legality is BETTER than the encoder it replaces: the whole network is
conv / DeployNorm (running-stat affine, folds into the adjacent conv) / SiLU
(single-LUT on Hailo, same as every YOLO the DFC compiles) / residual add.
No cosine gates, no gated pooling, no data-dependent anything. Translation
equivariance is kept on purpose: no (x,y) coordinate channels — a center
detector should respond to what an object looks like, not where it is.

Receptive field at the output grid: the three dw5x5 blocks alone give +-6
cells (+-120 px), on top of the 100 px window of the stride-5 downsample and
the fine-stride stages — roughly a 250-300 px neighborhood per cell, plenty
for mannequin/tent plus local context at 150 ft GSD without any global path
(SlimContext-style global vectors are exactly the shortcut a collapsing head
hides in; v13 deliberately has none).

25,212 params at the trained default head_width=24 — inside the <40k budget
with margin (see scripts/smoke_test and ARCHITECTURE.md section 15.2).

Training notes: DeployNorm's deferred-EMA contract applies unchanged — the
Trainer seeds stats before step 0 and calls apply_norm_updates() after each
backward; any external training loop must do the same. Activation maps are
small enough that no gradient checkpointing is used regardless of the
use_checkpoint flag.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import quat_mul
from .norm import DeployNorm


class _DWSep(nn.Module):
    """Depthwise-separable unit: dw kxk -> DN -> SiLU -> pw 1x1 -> DN -> SiLU.
    Residual only when it changes nothing about shape (stride 1, ch_in==ch_out).
    Padding: k//2 keeps H/W (stride 1) or halves it exactly on even sizes
    (stride 2: 270->135, 480->240); the stride==k case (dw5x5 s5) tiles the
    grid exactly with padding 0 (135/5=27, 240/5=48 — 540 = 2*2*5 * 27, no
    half-pixel alignment anywhere)."""

    def __init__(self, ch_in, ch_out, k, stride, zero_gain=False):
        super().__init__()
        self.residual = stride == 1 and ch_in == ch_out
        pad = 0 if stride == k else k // 2
        self.dw = nn.Conv2d(ch_in, ch_in, k, stride, pad, groups=ch_in, bias=False)
        self.dw_norm = DeployNorm(ch_in)
        self.pw = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        self.pw_norm = DeployNorm(ch_out)
        self.act = nn.SiLU()
        # zero-gamma valve (v14, D63): per-channel zero-init scale on the
        # branch so a NEW residual block is exact identity at step 0 while its
        # convs stay Kaiming-init and its DeployNorms observe REAL activation
        # stats from the first forward. Zeroing the pw weights instead would
        # park pw_norm's running_var at ~0 and fold a ~sqrt(1/eps)=316x
        # amplifier onto the branch just as it wakes up. v13 blocks never pass
        # zero_gain, so their state_dicts are unchanged.
        self.gain = nn.Parameter(torch.zeros(1, ch_out, 1, 1)) if zero_gain else None

    def forward(self, x):
        y = self.act(self.dw_norm(self.dw(x)))
        y = self.act(self.pw_norm(self.pw(y)))
        if self.gain is not None:
            y = self.gain * y
        return x + y if self.residual else y


class V13Backbone(nn.Module):
    """(B, 3, 540, 960) in [0,1] -> (B, 4, 27, 48) raw logits
    [center_mannequin, center_tent, dx, dy]."""

    CH_STEM, CH_MID, CH_TOP = 16, 32, 64  # spec defaults (v14 pins these)

    def __init__(self, head_width=24, prior_fg=None, n_blocks=3,
                 channels=(16, 32, 64)):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels  # scalable for the D65 curve
        self.channels = tuple(channels)
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = nn.SiLU()
        self.down4 = _DWSep(ch_stem, ch_mid, k=3, stride=2)
        self.block4 = _DWSep(ch_mid, ch_mid, k=3, stride=1)
        self.down20 = _DWSep(ch_mid, ch_top, k=5, stride=5)
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1)
              for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        # Variance-preserving init (Kaiming, ReLU gain ~ SiLU gain): torch's
        # default conv init loses ~10x activation scale per dw+pw stage on
        # this stack, which puts DeployNorm's cold start ~300x off per layer —
        # the trainer's 8 sequential seeding passes cannot relax a 10-norm
        # cascade that far (measured: 1e23 logits on the first train step).
        # With unit-variance propagation the running stats start near their
        # fixed point and seeding converges in a couple of passes.
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        if prior_fg:
            # RetinaNet-style prior on the two independent center sigmoids
            # (same rationale and value as v12's CenterHead: at p_init=0.01 the
            # ~2590 background cells barely push the shared bias down while the
            # positive -log(p) term strongly lifts true centers). dx/dy at 0.
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.head[-1].bias.zero_()
                self.head[-1].bias[0:2] = b

    def forward(self, img):  # (B, 3, 540, 960) -> (B, 4, 27, 48)
        x = self.act(self.stem_norm(self.stem(img)))
        x = self.block4(self.down4(x))
        x = self.blocks(self.down20(x))
        return self.head(x)


# ============================================================================
# v14 (D59-D63): v13 + structured, fold-legal priors — each tied to a failure
# mode measured on the first v13 MI300X model (comparison.json + the 24-frame
# runs/viz stage dump):
#
#   (B) false peaks in a narrow 0.30-0.60 band, clustered on canopy texture
#       (fp/img 2.15 vs YOLO26n 0.018)      -> NoiseFilter7 (D59) + TextureGate (D61)
#   (A) worst-decile mannequin misses with heat 0.2-0.3 — evidence DILUTED by
#       the s4->s20 strided average, not absent -> max-pool detail skip (D62)
#   (C) clutter hits under-confident (0.35-0.55 vs 0.85-0.98 on clear ground)
#       -> 4th s20 block + QuatShift cross-channel mixing (D60/D63)
#
# THE CONTRACT (D63, the load-bearing design rule): every v14 module is
# identity- or zero-initialized, so a v14 warm-started from a v13 checkpoint
# computes EXACTLY the v13 function at step 0 (asserted in smoke_test) and
# every new degree of freedom can only move away from a proven optimum under
# gradient pressure — a monotone extension, not a re-roll of the dice.
# Everything folds to Hailo-legal ops at export: the quaternion algebra
# evaluates to constant grouped 1x1 convs, the gates are conv+sigmoid+mul
# (the repo's D10 idiom), the skip is max-pool+1x1.
# ============================================================================


class QuatShift(nn.Module):
    """One layer of learned dual-quaternion shift (v14, D60).

    Channels are grouped in 4s; each group is treated as a quaternion field
    and transformed by a learned unit-quaternion rotation p (Hamilton
    p (x) q (x) p-bar — rotates the 3 imaginary channels, passes the scalar
    channel through) plus the dual-part translation t = 2 * qd (x) p-bar of a
    learned dual quaternion (qr=p, qd) — the same parametrization as
    DualQuaternionRGB (D5), generalized from the RGB 3-vector to 4-channel
    feature groups, and kept as a 4-d shift (the scalar component of t is
    meaningful for feature groups even though it vanishes for pure 3-vectors).

    Deploy form: the whole layer evaluates to ONE constant block-diagonal
    grouped 1x1 conv (4x4 per group) + bias — bake at export exactly like
    DualQuaternionRGB.to_conv(). Identity init (p=[1,0,0,0], qd=0) makes the
    layer a no-op at step 0 (D63). ~8 params per group: cross-channel mixing
    at 1/2 the parameter cost of a full per-group 4x4 + norm-preserving
    rotation structure on the imaginary subspace.
    """

    def __init__(self, ch):
        super().__init__()
        assert ch % 4 == 0, "QuatShift needs channels in groups of 4"
        self.groups = ch // 4
        self.qr = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.groups, 1))
        self.qd = nn.Parameter(torch.zeros(self.groups, 4))

    def _fold(self):
        """Constant (weight (C,4,1,1), bias (C,)) of the grouped 1x1 conv."""
        q = self.qr / self.qr.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = q.unbind(-1)
        one = torch.ones_like(w)
        zero = torch.zeros_like(w)
        m = torch.stack([
            torch.stack([one, zero, zero, zero], -1),
            torch.stack([zero, 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([zero, 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([zero, 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ], -2)                                            # (G, 4, 4)
        conj = q * q.new_tensor([1.0, -1.0, -1.0, -1.0])
        t = 2.0 * quat_mul(self.qd, conj)                 # (G, 4)
        return m.reshape(-1, 4, 1, 1), t.reshape(-1)

    def forward(self, x):  # (B, C, H, W) -> (B, C, H, W)
        weight, bias = self._fold()
        return F.conv2d(x, weight.to(x.dtype), bias.to(x.dtype),
                        groups=self.groups)


class TextureGate(nn.Module):
    """Learned texture masking as a weighted sum (v14, D61).

    Failure mode B: nearly all of v13's false peaks sit on high-frequency
    vegetation/canopy texture at probs 0.30-0.60 — the head scores object
    evidence against an ABSOLUTE bar, but in canopy everything is
    high-frequency, so texture must raise the local bar. This branch measures
    local texture energy at s4 (learned high-pass, init as the D32 isotropic
    high-pass kernel; energy = elementwise square), pools it to the s20 grid,
    and produces a per-channel sigmoid mask g. It modulates the trunk as a
    BOUNDED weighted sum

        y' = y * (1 + tanh(w_gate) * g),   w_gate init 0

    Exact identity at init (D63); the factor lives in (1-g, 1+g) subset of
    (0, 2), so per-channel texture-conditioned suppression/boost is
    expressible but GLOBAL TRUNK SHUTDOWN IS NOT. The first draft used an
    unbounded w_pass + w_gate*g, and the first from-scratch MI300X v14 run
    found the collapse channel: the cheapest way to satisfy the background
    focal term everywhere is to shrink the whole-trunk multiplier toward
    zero — measured mann_r 0.52 -> 0.009 across epochs ~13-20 with train
    loss near-flat, recovering only as the cosine decayed, early-stopped at
    24. v13 had no whole-trunk multiplier, hence no such channel; bounding
    the gate restores that safety while keeping the mechanism. tanh acts on
    a WEIGHT (constant at export), so the layer still folds to
    conv+sigmoid+mul; all ops remain depthwise conv, mul (square),
    DeployNorm affine, avg-pool, 1x1 convs, sigmoid, mul."""

    def __init__(self, ch_in, ch_out, hidden=16):
        super().__init__()
        self.hp = nn.Conv2d(ch_in, ch_in, 3, 1, 1, groups=ch_in, bias=False)
        with torch.no_grad():  # isotropic high-pass init (same as the D32 stem)
            self.hp.weight.fill_(-1.0 / 9.0)
            self.hp.weight[:, :, 1, 1] += 1.0
        self.norm = DeployNorm(ch_in)
        self.fc1 = nn.Conv2d(ch_in, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch_out, 1)
        self.act = nn.SiLU()
        self.w_gate = nn.Parameter(torch.zeros(1, ch_out, 1, 1))

    def forward(self, x_s4, y):  # texture stats from s4 modulate the s20 trunk
        e = self.hp(x_s4)
        e = self.norm(e * e)                              # local texture energy
        e = F.avg_pool2d(e, 5, 5)                         # 135x240 -> 27x48
        g = torch.sigmoid(self.fc2(self.act(self.fc1(e))))
        return y * (1.0 + torch.tanh(self.w_gate) * g)


class V14Backbone(nn.Module):
    """(B, 3, 540, 960) in [0,1] -> (B, 4, 27, 48) raw logits — the v13 conv
    pyramid extended by the D59-D63 structured priors. v13 module names are
    preserved verbatim (stem/down4/block4/down20/blocks/head) so a v13
    checkpoint warm-starts every shared weight; with the new modules at their
    identity inits the warm-started v14 IS the v13 function (asserted in
    smoke_test)."""

    CH_STEM, CH_MID, CH_TOP = V13Backbone.CH_STEM, V13Backbone.CH_MID, V13Backbone.CH_TOP

    def __init__(self, head_width=24, prior_fg=None, n_blocks=3):
        super().__init__()
        # D59: learned 7x7 noise filter — residual depthwise pre-filter on RGB,
        # zero-init (identity at step 0). Learns sensor-noise / high-frequency
        # suppression BEFORE the stem commits features. Stays one depthwise
        # conv at export.
        self.noise = nn.Conv2d(3, 3, 7, 1, 3, groups=3, bias=False)
        self.stem = nn.Conv2d(3, self.CH_STEM, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(self.CH_STEM)
        self.act = nn.SiLU()
        self.down4 = _DWSep(self.CH_STEM, self.CH_MID, k=3, stride=2)
        self.block4 = _DWSep(self.CH_MID, self.CH_MID, k=3, stride=1)
        self.down20 = _DWSep(self.CH_MID, self.CH_TOP, k=5, stride=5)
        # D62: peak-preserving detail skip. The strided dw5x5 AVERAGES a tiny
        # object's s4 evidence into its 100-px window (measured: missed
        # worst-decile mannequins peak at heat 0.2-0.3 — diluted, not absent);
        # max-pool keeps the brightest s4 response alive. Kaiming conv + real
        # DN stats from step 0; the zero-init per-channel gain is the identity
        # valve (see _DWSep.zero_gain for why the valve is NOT a zeroed conv).
        self.skip = nn.Conv2d(self.CH_MID, self.CH_TOP, 1, bias=False)
        self.skip_norm = DeployNorm(self.CH_TOP)
        self.skip_gain = nn.Parameter(torch.zeros(1, self.CH_TOP, 1, 1))
        self.blocks = nn.Sequential(
            *[_DWSep(self.CH_TOP, self.CH_TOP, k=5, stride=1)
              for _ in range(n_blocks)])
        # D63: extra s20 capacity for clutter discrimination, zero-gamma so
        # the residual passes through unchanged at step 0.
        self.block_extra = _DWSep(self.CH_TOP, self.CH_TOP, k=5, stride=1,
                                  zero_gain=True)
        # D60: five dual-quaternion shift layers, one after each stage.
        self.qshift1 = QuatShift(self.CH_STEM)
        self.qshift2 = QuatShift(self.CH_MID)
        self.qshift3 = QuatShift(self.CH_MID)
        self.qshift4 = QuatShift(self.CH_TOP)
        self.qshift5 = QuatShift(self.CH_TOP)
        # D61: texture masking (weighted sum) from s4 energy onto the trunk.
        self.tex = TextureGate(self.CH_MID, self.CH_TOP)
        self.head = nn.Sequential(
            nn.Conv2d(self.CH_TOP, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        # Kaiming for the v13-inherited stack (same requirement as v13: unit
        # variance keeps DeployNorm's cold start near its fixed point)...
        for mod in (self.stem, self.down4, self.block4, self.down20,
                    self.blocks, self.block_extra, self.head):
            for m in mod.modules() if isinstance(mod, nn.Module) else [mod]:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # ...then the D59 identity init (AFTER kaiming, order matters). The
        # noise filter is the one zero CONV: it has no norm behind it (the
        # stem_norm sees img+0 = img, real stats), so the 316x-fold hazard the
        # zero-gamma valves avoid does not apply here.
        with torch.no_grad():
            self.noise.weight.zero_()
        nn.init.zeros_(self.tex.fc1.bias)
        nn.init.zeros_(self.tex.fc2.bias)
        if prior_fg:
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.head[-1].bias.zero_()
                self.head[-1].bias[0:2] = b

    def forward(self, img):  # (B, 3, 540, 960) -> (B, 4, 27, 48)
        x = img + self.noise(img)                          # D59
        x = self.qshift1(self.act(self.stem_norm(self.stem(x))))
        x = self.qshift2(self.down4(x))
        x_s4 = self.qshift3(self.block4(x))
        x = self.down20(x_s4) + self.skip_gain \
            * self.skip_norm(self.skip(F.max_pool2d(x_s4, 5, 5)))  # D62
        x = self.qshift4(x)
        x = self.qshift5(self.block_extra(self.blocks(x)))  # D63
        x = self.tex(x_s4, x)                               # D61
        return self.head(x)


class V15Backbone(nn.Module):
    """v15 (D64/D65): the capacity-scaling architecture, sized by the YOLO26n
    weight-anatomy study (ARCHITECTURE.md section 16.3).

    Two findings drive it. (1) v13's train-split eval proved UNDERFITTING
    (train recall == test recall at ~0.83/0.59-decile) — the ceiling is
    representational. (2) YOLO26n's weights show where a model that CAN fit
    tiny objects spends parameters: 73.7% at stride 32 (deep semantics), a
    dedicated fine-grid head — and never a stride jump greater than 2. v13
    funnels ALL fine-scale evidence through one lossy 2,048-param strided
    pipe (down20.pw after a 5x-strided depthwise average); the measured
    symptom was worst-decile mannequins peaking at heat 0.2-0.3 — diluted,
    not absent.

    D64 — SPD projection (space-to-depth, SPD-Conv, Sunkara & Luo 2022):
    the s4 map is rearranged losslessly to the s20 grid via
    pixel_unshuffle(5) — (ch_mid, 135, 240) -> (25*ch_mid, 27, 48) — then a
    learned 1x1 selects what to keep. Every s4 pixel's features reach the
    detection grid intact; the projection IS the capacity spend, placed
    exactly where v13's evidence died. Hailo-native (space_to_depth, the
    YOLOv5 Focus layer), exports as ONNX SpaceToDepth.

    D65 — deep-heavy tiers, per the YOLO allocation (pre-registered curve;
    the self-imposed 40k budget is deliberately relaxed for these runs):
      tier S (defaults):        channels (16,32,64),  3 blocks  ~74k params
      tier M (ANET_CH/BLOCKS):  channels (16,48,96),  4 blocks  ~170k params
    v13 (25k) is the curve's origin. Verdict key: where train-split
    worst-decile recall lifts off; the FP band's response decides whether
    the texture-prior question (v14, falsified at 25k) reopens.
    """

    def __init__(self, head_width=24, prior_fg=None, channels=(16, 32, 64),
                 n_blocks=3, n_s4_blocks=1):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels
        self.channels = tuple(channels)
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = nn.SiLU()
        self.down4 = _DWSep(ch_stem, ch_mid, k=3, stride=2)
        self.s4_blocks = nn.Sequential(
            *[_DWSep(ch_mid, ch_mid, k=3, stride=1) for _ in range(n_s4_blocks)])
        self.spd_proj = nn.Conv2d(ch_mid * 25, ch_top, 1, bias=False)  # D64
        self.spd_norm = DeployNorm(ch_top)
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        for mod in self.modules():  # same cold-start requirement as v13 (D58)
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        if prior_fg:
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.head[-1].bias.zero_()
                self.head[-1].bias[0:2] = b

    def forward(self, img):  # (B, 3, 540, 960) -> (B, 4, 27, 48)
        x = self.act(self.stem_norm(self.stem(img)))
        x = self.s4_blocks(self.down4(x))          # (B, ch_mid, 135, 240)
        x = F.pixel_unshuffle(x, 5)                # lossless -> (B, 25*ch_mid, 27, 48)
        x = self.act(self.spd_norm(self.spd_proj(x)))
        x = self.blocks(x)
        return self.head(x)


class CosineWeaveTexture(nn.Module):
    """Auxiliary cosine-weave texture channel (v16, D66) — user-directed test
    of the texture-prior hypothesis on the UNSCALED v13, in its fairest form.

    Pre-registered expectations (ARCHITECTURE.md 16.4): per the 16.2 capacity
    verdict, a prior on v13 is unlikely to lift recall (the worst-decile
    misses are objects v13 cannot represent); but the canopy FP band
    (false peaks at prob 0.30-0.60 on high-frequency texture) is a
    decision-boundary problem, so fp/img reduction AT HELD RECALL is the
    plausible win and the falsifier this module is judged on.

    Mechanism — the project's signature multi-cosine weave (SlimContext/D44
    idiom), applied SPATIALLY and deploy-legally:

        e   = DN(highpass(x_s4)^2)         texture energy at s4 (D32-init hp)
        E   = avgpool5(e)                  -> the 27x48 grid
        u   = tanh(1x1: ch_mid -> K)       bounded states, |u| < 1
        W   = [cos(f1*u + p1), cos(f2*u + p2)]   2 harmonics x K states
        g   = sigmoid(1x1: 2K -> ch_top)   per-channel texture mask
        y'  = y * (1 + tanh(w_gate) * g)   bounded modulation (v14 D61 lesson:
                                           factor in (1-g,1+g) subset (0,2) —
                                           trunk shutdown unrepresentable)

    Deploy legality is the same LUT contract as every cosine in this repo:
    tanh bounds each state to (-1,1), so the cosine argument lives in
    (-|f|-|p|, |f|+|p|) with the frequencies L2-regularized through
    reg_l2() -> the existing D24 l2_score_reg hook keeps them inside one
    period for int8 LUTs. tanh/cos/sigmoid are single LUTs; everything else
    is conv/DN/mul/avg-pool.

    Every v14 safety lesson is load-bearing here: identity at init
    (w_gate=0), DN observes real energy stats from the first forward (no
    zero-valve behind a norm), bounded gate, and v13 module names untouched
    upstream so a v13 checkpoint warm-starts v16 bit-exactly (D63 contract,
    asserted in smoke). ~1.8k params — v16 stays inside the ORIGINAL <40k
    budget (~27k total)."""

    def __init__(self, ch_mid, ch_top, k=8):
        super().__init__()
        self.hp = nn.Conv2d(ch_mid, ch_mid, 3, 1, 1, groups=ch_mid, bias=False)
        with torch.no_grad():  # isotropic high-pass init (D32 stem kernel)
            self.hp.weight.fill_(-1.0 / 9.0)
            self.hp.weight[:, :, 1, 1] += 1.0
        self.norm = DeployNorm(ch_mid)
        self.state = nn.Conv2d(ch_mid, k, 1)
        # two harmonics per state, frequencies started inside one period and
        # held there by reg_l2 (D24); phases learnable, unregularized
        self.freq = nn.Parameter(torch.tensor([1.0, 2.0]).repeat_interleave(k))
        self.phase = nn.Parameter(torch.zeros(2 * k))
        self.mix = nn.Conv2d(2 * k, ch_top, 1)
        self.w_gate = nn.Parameter(torch.zeros(1, ch_top, 1, 1))

    def forward(self, x_s4, y):
        e = self.hp(x_s4)
        e = self.norm(e * e)                            # texture energy
        e = F.avg_pool2d(e, 5, 5)                       # 135x240 -> 27x48
        u = torch.tanh(self.state(e))                   # bounded states (B,K,27,48)
        uu = torch.cat([u, u], 1)                       # (B,2K,...) one per harmonic
        w = torch.cos(uu * self.freq.reshape(1, -1, 1, 1)
                      + self.phase.reshape(1, -1, 1, 1))  # the weave
        g = torch.sigmoid(self.mix(w))                  # (B, ch_top, 27, 48)
        return y * (1.0 + torch.tanh(self.w_gate) * g)

    def reg_l2(self):  # D24: cosine frequencies bounded to one LUT period
        return (self.freq ** 2).sum()


class V16Backbone(nn.Module):
    """v13 trunk (module names preserved verbatim — D63 warm-start contract)
    + the CosineWeaveTexture channel between the blocks and the head. The
    single-variable ablation the texture hypothesis deserves: ~1.8k new
    params, everything else bit-identical to v13."""

    def __init__(self, head_width=24, prior_fg=None, channels=(16, 32, 64),
                 n_blocks=3):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels
        self.channels = tuple(channels)
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = nn.SiLU()
        self.down4 = _DWSep(ch_stem, ch_mid, k=3, stride=2)
        self.block4 = _DWSep(ch_mid, ch_mid, k=3, stride=1)
        self.down20 = _DWSep(ch_mid, ch_top, k=5, stride=5)
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1) for _ in range(n_blocks)])
        self.weave = CosineWeaveTexture(ch_mid, ch_top)
        self.head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        for mod in (self.stem, self.down4, self.block4, self.down20,
                    self.blocks, self.head):
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.zeros_(self.weave.state.bias)
        nn.init.zeros_(self.weave.mix.bias)
        if prior_fg:
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.head[-1].bias.zero_()
                self.head[-1].bias[0:2] = b

    def forward(self, img):  # (B, 3, 540, 960) -> (B, 4, 27, 48)
        x = self.act(self.stem_norm(self.stem(img)))
        x_s4 = self.block4(self.down4(x))
        y = self.blocks(self.down20(x_s4))
        y = self.weave(x_s4, y)                          # D66
        return self.head(y)


class PowerBlend(nn.Module):
    """Learned power-law RGB activation (v17, D67) — the owner's A^v op.

    Per pixel, with v = normalized RGB (chromaticity, v_i = c_i / sum(c),
    bounded [0,1]) and a learned 3x3 exponent-rate matrix W (= ln A):

        M_ij  = exp(W_ij * v_i)          # A^v, row i powered by v_i
        M'_ij = relu(M_ij - tau_j)       # learned threshold zeroes entries
        out_j = sum_i M'_ij              # column sum blends all 3 channels

    A sum of exponentials with learned rates, sparsified by a learned
    threshold — a learned activation over chromaticity. Deploy-legal by the
    house rules: exp is one LUT with its argument bounded (|arg| <= |W|
    with v in [0,1]; W is L2-held through reg_l2 -> the D24 hook, plus a
    hard clamp(-4,4) as belt-and-braces saturation), threshold = bias+relu,
    channel repeat = concat, the fixed column sum = a constant 1x1 conv.
    12 params."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(9))     # W_ij, i-major; 0 -> A = 1
        self.tau = nn.Parameter(torch.full((3,), 0.5))

    def forward(self, rgb):  # (B, 3, H, W) raw RGB in [0,1]
        v = rgb / rgb.sum(1, keepdim=True).clamp_min(1e-4)   # chromaticity
        v9 = v.repeat_interleave(3, dim=1)                   # [v1 v1 v1 v2 ...]
        m = torch.exp((v9 * self.w.reshape(1, 9, 1, 1)).clamp(-4.0, 4.0))
        m = F.relu(m - self.tau.repeat(3).reshape(1, 9, 1, 1))
        return m.reshape(m.shape[0], 3, 3, *m.shape[2:]).sum(1)  # (B,3,H,W)

    def reg_l2(self):  # D24: keep the exp argument inside the LUT range
        return (self.w ** 2).sum()


class _PBInject(nn.Module):
    """One v17 injection site: PowerBlend of the (pooled) input RGB ->
    1x1 to the stage's channels -> zero-gamma gain -> added to the stage
    features. Identity at init (gain 0); the norms-free path means no
    cold-start or stat-drift hazards."""

    def __init__(self, ch, pool):
        super().__init__()
        self.pool = pool
        self.pb = PowerBlend()
        self.proj = nn.Conv2d(3, ch, 1)
        self.gain = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x, rgb):
        r = F.avg_pool2d(rgb, self.pool, self.pool) if self.pool > 1 else rgb
        return x + self.gain * self.proj(self.pb(r))


class V17Backbone(nn.Module):
    """v13 trunk (any D65 channel plan; module names preserved — a plain or
    scaled v13 checkpoint warm-starts v17 bit-exactly) + a PowerBlend
    injector between every stage (D67): after the stem (s2), after the s4
    stage, at the s20 entry, and before the head. ~0.6-1.3k added params
    depending on width.

    Pre-registered judgment (16.5): v17-at-tier vs the PLAIN tier at matched
    training — the injectors are judged on that delta alone, so the
    capacity-curve datapoint stays unconfounded."""

    def __init__(self, head_width=24, prior_fg=None, channels=(16, 32, 64),
                 n_blocks=3):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels
        self.channels = tuple(channels)
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = nn.SiLU()
        self.down4 = _DWSep(ch_stem, ch_mid, k=3, stride=2)
        self.block4 = _DWSep(ch_mid, ch_mid, k=3, stride=1)
        self.down20 = _DWSep(ch_mid, ch_top, k=5, stride=5)
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1) for _ in range(n_blocks)])
        self.pb1 = _PBInject(ch_stem, pool=2)    # after stem   (270x480)
        self.pb2 = _PBInject(ch_mid, pool=4)     # after s4     (135x240)
        self.pb3 = _PBInject(ch_top, pool=20)    # s20 entry    (27x48)
        self.pb4 = _PBInject(ch_top, pool=20)    # pre-head     (27x48)
        self.head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        for mod in (self.stem, self.down4, self.block4, self.down20,
                    self.blocks, self.head):
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if prior_fg:
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.head[-1].bias.zero_()
                self.head[-1].bias[0:2] = b

    def forward(self, img):  # (B, 3, 540, 960) -> (B, 4, 27, 48)
        x = self.pb1(self.act(self.stem_norm(self.stem(img))), img)
        x = self.pb2(self.block4(self.down4(x)), img)
        x = self.pb3(self.down20(x), img)
        x = self.pb4(self.blocks(x), img)
        return self.head(x)

    def reg_l2(self):
        return sum(p.pb.reg_l2() for p in (self.pb1, self.pb2, self.pb3, self.pb4))


class V18Backbone(nn.Module):
    """v13 trunk + the D68 auxiliary heads (owner-directed):

    (a) STATE-DRIVEN EXPOSURE MASK — "add ~1.5 stops of brightness to
        selected areas": the front of the trunk (stem/down4/block4, shared
        weights) runs on the image AND on a 2^1.5x brightened copy; a mask
        head — a global state vector (GAP of s4; JEPA-STYLE latent, not JEPA
        training) biasing a local 1x1 head — blends the two s4 maps:
            s4' = s4 + tanh(blend_gain) * m * (s4_bright - s4)
        Identity at init (blend_gain = 0). Mechanistic target: the
        worst-decile canopy/shadow mannequins — underexposed objects the
        shared trunk never sees well. Deploy-legal: the front runs twice
        (two subgraphs, shared weights), GAP + 1x1s + sigmoid + mul.
        The bright pass runs with the front's DeployNorms frozen so its
        activation statistics never contaminate the running stats the
        normal branch (and the deploy graph) normalize against.

    (b) BACKGROUND-MASK AUX HEAD (train-only, dropped at eval/export like
        the D46 probe): 1x1 -> 1 bg logit per cell off the pre-head
        features, supervised from the heat targets (bg = 1 - max Gaussian)
        plus the owner's smoothness prior on the PREDICTED background
        (penalize high-frequency bg — trainer side, bg_smooth_weight).
        Different mechanism from the falsified D61/D66/D67 gates: this adds
        TRAINING SIGNAL that shapes trunk features rather than inference
        machinery that reweights them.

    Trunk module names preserved verbatim — any v13/scaled-v13 checkpoint
    warm-starts v18 bit-exactly (D63 contract, smoke-asserted)."""

    def __init__(self, head_width=24, prior_fg=None, channels=(16, 32, 64),
                 n_blocks=3, stops=1.5):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels
        self.channels = tuple(channels)
        self.exposure = 2.0 ** stops
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = nn.SiLU()
        self.down4 = _DWSep(ch_stem, ch_mid, k=3, stride=2)
        self.block4 = _DWSep(ch_mid, ch_mid, k=3, stride=1)
        self.down20 = _DWSep(ch_mid, ch_top, k=5, stride=5)
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        self.mask_hidden = nn.Conv2d(ch_mid, 8, 1)
        self.mask_state = nn.Linear(ch_mid, 8)  # global state -> mask bias
        self.mask_out = nn.Conv2d(8, 1, 1)
        self.blend_gain = nn.Parameter(torch.zeros(()))
        self.bg_head = nn.Conv2d(ch_top, 1, 1)  # train-only aux
        for mod in (self.stem, self.down4, self.block4, self.down20,
                    self.blocks, self.head):
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if prior_fg:
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.head[-1].bias.zero_()
                self.head[-1].bias[0:2] = b

    def _front_norms(self):
        for mod in (self.stem_norm, self.down4.dw_norm, self.down4.pw_norm,
                    self.block4.dw_norm, self.block4.pw_norm):
            yield mod

    def _front(self, img, observe=True):
        if not observe:  # bright pass: never let its stats into the buffers
            saved = [(m, m.frozen) for m in self._front_norms()]
            for m, _ in saved:
                m.frozen = True
        x = self.block4(self.down4(self.act(self.stem_norm(self.stem(img)))))
        if not observe:
            for m, was in saved:
                m.frozen = was
        return x

    def forward(self, img):
        s4 = self._front(img)
        bright = self._front((img * self.exposure).clamp(max=1.0),
                             observe=False)
        z = s4.mean((2, 3))                                # state vector
        h = F.silu(self.mask_hidden(s4)
                   + self.mask_state(z).unsqueeze(-1).unsqueeze(-1))
        m = torch.sigmoid(self.mask_out(h))                # (B,1,135,240)
        s4 = s4 + torch.tanh(self.blend_gain) * m * (bright - s4)
        y = self.blocks(self.down20(s4))
        out = self.head(y)
        if self.training:
            return out, self.bg_head(y)                    # + aux bg logits
        return out


class LearnedAct(nn.Module):
    """v19 (D69-B): owner's learned per-layer activation — a learned-slope
    SiLU plus a learned Gaussian bump, weights shared across the layer:

        f(x) = x * sigmoid(beta * x) + gamma * exp(-(x - mu)^2 / (2 sigma^2))

    Parametric identity at init (beta=1, gamma=0 -> exactly SiLU), so no
    valve is needed. 4 scalars per site. Deploy: any scalar function of x
    is ONE Hailo LUT — the cheapest mechanism in the family."""

    def __init__(self):
        super().__init__()
        # ALL bounded by construction (first v19 gate run NaN'd: at the
        # identity point these params see gradients in the 1e2-1e3 range,
        # and unbounded beta/gamma fly before any scheduler can help):
        #   beta  = 1 + 0.5*tanh(b)   in (0.5, 1.5), init exactly 1
        #   gamma = 0.5*tanh(g)       in (-0.5, 0.5), init exactly 0
        #   mu    = tanh(m)           in (-1, 1)
        #   sigma = softplus(rho)+0.25 >= 0.25
        self.b = nn.Parameter(torch.zeros(()))
        self.g = nn.Parameter(torch.zeros(()))
        self.m = nn.Parameter(torch.zeros(()))
        self.rho = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        beta = 1.0 + 0.5 * torch.tanh(self.b)
        gamma = 0.5 * torch.tanh(self.g)
        mu = torch.tanh(self.m)
        sigma = F.softplus(self.rho) + 0.25
        return x * torch.sigmoid(beta * x) \
            + gamma * torch.exp(-(x - mu) ** 2 / (2 * sigma * sigma))


class ExposureBumps(nn.Module):
    """v19 (D69-C): the owner's simplified exposure mechanism — steal a
    latent from a MICRO look at the image (8x-pooled, two tiny convs, GAP),
    and a small head emits 4 normalized (x,y) coords + a per-bump exposure
    amount (sigmoid-capped at +1.5 stops). Gaussian bumps are rendered at
    those coords and applied to the INPUT (exposure lives in image space):

        img' = clamp(img * (1 + tanh(valve) * E), 0, 1),
        E    = sum_k amt_k * exp(-((X-x_k)^2 + (Y-y_k)^2) / (2 s^2))

    Identity at init (valve=0). Deploy: the micro-head is 4 scalar triples
    per frame — a D17-style CPU stage (microseconds); the bump multiply is
    elementwise. ~1.6k params."""

    MAX_GAIN = 2.0 ** 1.5 - 1.0  # +1.5 stops, the owner's number

    def __init__(self, k=4):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.SiLU(),
        )
        self.head = nn.Linear(16, k * 3)
        self.log_s = nn.Parameter(torch.tensor(-2.5))  # bump radius ~0.08
        self.valve = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.head.bias)

    def forward(self, img):  # (B,3,H,W) in [0,1] -> adjusted image
        z = self.enc(F.avg_pool2d(img, 8)).mean((2, 3))
        p = self.head(z).reshape(-1, self.k, 3)
        xy = torch.sigmoid(p[..., :2])                       # (B,k,2) in [0,1]
        amt = torch.sigmoid(p[..., 2]) * self.MAX_GAIN       # (B,k)
        B, _, H, W = img.shape
        ys = torch.linspace(0, 1, H, device=img.device).reshape(1, 1, H, 1)
        xs = torch.linspace(0, 1, W, device=img.device).reshape(1, 1, 1, W)
        s2 = (self.log_s.exp() ** 2) * 2.0
        d2 = (xs - xy[..., 0].reshape(B, self.k, 1, 1)) ** 2 \
            + (ys - xy[..., 1].reshape(B, self.k, 1, 1)) ** 2
        E = (amt.reshape(B, self.k, 1, 1) * torch.exp(-d2 / s2)).sum(
            1, keepdim=True)
        return (img * (1.0 + torch.tanh(self.valve) * E)).clamp(0.0, 1.0)


class V19Backbone(nn.Module):
    """v19 (D69): the ATTRIBUTION build — every mechanism behind its own
    valve so one training run + per-mechanism ablation (the v17 autopsy
    method, built in) splits credit. On the untouched v13 trunk:

      A  bias injectors at 4 stage boundaries (176 params) — the mechanism
         the v17 autopsy identified as the actual worker; doubles as the
         pre-registered 16.5 bias-only control (ablate everything else).
      B  LearnedAct at every _DWSep pw-activation + post-stem (owner ask).
      C  ExposureBumps on the input (owner ask; replaces v18's mask).
      D  one identity-init QuatShift post-stem (owner invitation; reused
         from D60) + the v18 bg-aux training head (the family's fp-record
         earner; train-only, free at deploy).

    Identity-at-init contract (D63) holds for the whole assembly."""

    def __init__(self, head_width=24, prior_fg=None, channels=(16, 32, 64),
                 n_blocks=3):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels
        self.channels = tuple(channels)
        self.bumps = ExposureBumps()                       # C
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = LearnedAct()                            # B (post-stem)
        self.qshift = QuatShift(ch_stem)                   # D
        self.down4 = _DWSep(ch_stem, ch_mid, k=3, stride=2)
        self.block4 = _DWSep(ch_mid, ch_mid, k=3, stride=1)
        self.down20 = _DWSep(ch_mid, ch_top, k=5, stride=5)
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1) for _ in range(n_blocks)])
        for blk in (self.down4, self.block4, self.down20, *self.blocks):
            blk.act = LearnedAct()                         # B (shared per layer)
        self.bias1 = nn.Parameter(torch.zeros(1, ch_stem, 1, 1))   # A
        self.bias2 = nn.Parameter(torch.zeros(1, ch_mid, 1, 1))
        self.bias3 = nn.Parameter(torch.zeros(1, ch_top, 1, 1))
        self.bias4 = nn.Parameter(torch.zeros(1, ch_top, 1, 1))
        self.bg_head = nn.Conv2d(ch_top, 1, 1)             # D (train-only)
        self.head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        for mod in (self.stem, self.down4, self.block4, self.down20,
                    self.blocks, self.head):
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if prior_fg:
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.head[-1].bias.zero_()
                self.head[-1].bias[0:2] = b

    def forward(self, img):
        x = self.bumps(img)                                # C
        x = self.qshift(self.act(self.stem_norm(self.stem(x)))) + self.bias1
        x = self.block4(self.down4(x)) + self.bias2
        x = self.down20(x) + self.bias3
        y = self.blocks(x) + self.bias4
        out = self.head(y)
        if self.training:
            return out, self.bg_head(y)
        return out
