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


class AnisotropyContrast(nn.Module):
    """v23 (D76): two-scale structure-tensor coherence CONTRAST — the feature
    the family never had, and the direct answer to "the model scores local
    contrast, not person" (the viz_web_scenes diagnosis).

    Every prior shape idea in this project was 2-WAY (anisotropic vs
    isotropic), which structurally cannot separate a person from a painted
    runway stripe — both are elongated. Measured consequence: v22 fires
    0.50-0.58 on runway numbers while a clear spread-eagle person on clean
    dirt scores 0.10 (background corners score 0.36 — the margin is
    INVERTED). This module is 3-WAY, by comparing coherence at two scales:

        coherent at FINE but not COARSE  -> person (short limb/torso
                                            segments at several angles)
        coherent at EVERY scale          -> paint stripe / fence / shadow
                                            edge (straight for 100s of px)
        coherent at NO scale             -> canopy / brush (isotropic
                                            fractal texture)

    Structure tensor from FIXED (non-trainable) luminance + Sobel at s2:
    J = [[Ix^2, IxIy], [IxIy, Iy^2]], box-averaged at a fine window (~limb
    width) and a coarse window (~whole-body extent). Per scale we keep
    trace = Jxx+Jyy (how much gradient energy) and the eigenvalue-gap
    magnitude sqrt(((Jxx-Jyy)/2)^2 + Jxy^2) (how ORIENTED that energy is).
    The gap uses the alpha-max-beta-min approximation (max + 0.5*min, ~4%
    worst case) ON PURPOSE: it needs only abs/max/min/mul/add, so no sqrt
    and NO DIVIDE enters the deploy graph (division is structurally close
    to the data-dependent-normalization class the Hailo rules forbid).

    A DeployNorm(4) sits before the tiny MLP: the four J-statistics have
    wildly different natural scales (squared gradients over two window
    sizes), and without it the sigmoid saturates at init — the same
    cold-start reasoning as D39/D58. 4->8->1 with sigmoid gives a bounded
    A(x,y) in (0,1), 49 params + 8 norm affines.

    Deploy: fixed convs, elementwise mul/abs/max/min (the D61 'energy =
    elementwise square' precedent), two stride-1 avg_pools (the D6/D7
    blurred-gate-map precedent), two 1x1 convs, one sigmoid. No new op
    class, nothing data-dependent, nothing dynamic."""

    LUM = (0.299, 0.587, 0.114)

    def __init__(self, k_fine=5, k_coarse=21, hidden=8):
        super().__init__()
        self.k_fine, self.k_coarse = k_fine, k_coarse
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]])
        # register as buffers: fixed by design (a LEARNED pre-filter ahead of
        # the evidence is the falsified v21.4 attenuator; these never move)
        self.register_buffer("sobel", torch.stack([sobel_x, sobel_x.t()])
                             .unsqueeze(1))                 # (2,1,3,3)
        self.register_buffer("lum_w", torch.tensor(self.LUM).reshape(1, 3, 1, 1))
        self.norm = DeployNorm(4)
        self.fc1 = nn.Conv2d(4, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, 1, 1)

    @staticmethod
    def _amb(a, b):
        """alpha-max-beta-min approximation of sqrt(a^2+b^2) — no sqrt, no
        divide; monotone in both arguments, which is all the head needs."""
        a, b = a.abs(), b.abs()
        return torch.maximum(a, b) + 0.5 * torch.minimum(a, b)

    def _stats(self, jxx, jyy, jxy, k):
        pad = k // 2
        jxx = F.avg_pool2d(jxx, k, 1, pad)
        jyy = F.avg_pool2d(jyy, k, 1, pad)
        jxy = F.avg_pool2d(jxy, k, 1, pad)
        trace = jxx + jyy                      # total gradient energy
        gap = self._amb((jxx - jyy) * 0.5, jxy)  # how ORIENTED it is
        return trace, gap

    def forward(self, img):  # (B,3,540,960) -> (B,1,270,480) in (0,1)
        lum = F.avg_pool2d((img * self.lum_w.to(img.dtype)).sum(1, keepdim=True), 2)
        g = F.conv2d(lum, self.sobel.to(lum.dtype), padding=1)   # (B,2,270,480)
        ix, iy = g[:, 0:1], g[:, 1:2]
        jxx, jyy, jxy = ix * ix, iy * iy, ix * iy
        t_f, g_f = self._stats(jxx, jyy, jxy, self.k_fine)
        t_c, g_c = self._stats(jxx, jyy, jxy, self.k_coarse)
        z = self.norm(torch.cat([t_f, g_f, t_c, g_c], 1))
        return torch.sigmoid(self.fc2(F.silu(self.fc1(z))))


class V23Backbone(nn.Module):
    """v23 (D76-D79): frozen-trunk DUAL-GRID anisotropy head — the ground-up
    answer to the mannequin margin failure, at <=40k params.

    The viz_web_scenes cases proved the disease is not capacity (v22 added a
    full-rank funnel; its runs eroded, and the class that improved was the
    one already working). It is that (a) at stride-20 a 49x13px person is a
    1-2 cell POINT with no spatial support, so a lone bright person-cell and
    a lone bright bush-cell are the same object to the head, and (b) the
    only evidence the trunk offers is brightness/edge MAGNITUDE, which paint
    and canopy win as often as people do. Measured: easy spread-eagle person
    0.10 vs empty-corner background 0.36 — an INVERTED margin.

    Two structural changes, mannequin-only (tents already work: 0.75-0.93):

      1. PER-CLASS ANISOTROPIC GRID. Mannequin is read at stride-10
         (54x96), tapped off the stem at s2 BEFORE the s4->s20 funnel that
         D62/D64 measured as diluting small-object evidence. A 49x13px
         person goes from ~1 cell to ~5x1.3 cells: elongation becomes
         representable. Tent keeps v13's proven stride-20 path, untouched.
      2. ANISOTROPY CONTRAST (above) concatenated onto the stem tap, giving
         the head a qualitatively new, 3-way degree of freedom that no
         amount of stacking the existing primitive can express.

    TENT SAFETY BY CONSTRUCTION, not by hope: the shared trunk and the tent
    head are loaded from v13_best and FROZEN — weights AND DeployNorm stats
    (freeze_donor(); the D39/16.1 law that frozen weights require frozen
    stats whenever anything trainable sits upstream). The mannequin branch
    only READS a tap off the stem and writes a separate output tensor, so
    the tent forward is bit-for-bit v13_best's, unconditionally. This is
    strictly stronger than v14's zero-init valve idiom (D63): a valve can
    drift under gradient pressure; a frozen parameter cannot.

    33,119 params (25,187 frozen + 7,932 trainable) — 17% under the <=40k
    cap the owner chose. All ops Hailo-legal; the dual-grid output is
    structurally YOLO's own multi-scale P3/P4 head pattern. The mannequin
    grid's targets/metrics ride the already-landed plumbing
    (boxes_to_heatmap(grid_hw=, classes=), SUASCells(center_grid=),
    shape-derived CenterObjectMetrics)."""

    def __init__(self, head_width=24, prior_fg=None, n_blocks=3,
                 channels=(16, 32, 64), man_ch=16):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels
        self.channels = tuple(channels)
        # ---- shared trunk: v13 verbatim (module names preserved so a
        # v13_best state_dict lands by name) ----
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = nn.SiLU()
        self.down4 = _DWSep(ch_stem, ch_mid, k=3, stride=2)
        self.block4 = _DWSep(ch_mid, ch_mid, k=3, stride=1)
        self.down20 = _DWSep(ch_mid, ch_top, k=5, stride=5)
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1) for _ in range(n_blocks)])
        # ---- tent head: v13's head with the mannequin output row dropped
        # (3 outputs: tent_heat, dx, dy). Sliced from the donor at load. ----
        self.tent_head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 3, 1),
        )
        # ---- mannequin branch (trainable) ----
        self.aniso = AnisotropyContrast()
        # full-rank strided projection (D64: full-rank, not depthwise, at a
        # funnel) from the s2 tap + anisotropy channel straight to s10.
        # 270/5=54, 480/5=96 — exact, no padding, no pixel_unshuffle (so v23
        # is NOT in the v15/v20 ROCm inductor-miscompile family).
        self.man_proj = nn.Conv2d(ch_stem + 1, man_ch, 5, stride=5, bias=False)
        self.man_norm = DeployNorm(man_ch)
        self.man_block = _DWSep(man_ch, man_ch, k=5, stride=1)
        self.man_head = nn.Sequential(
            nn.Conv2d(man_ch, man_ch, 1),
            nn.SiLU(),
            nn.Conv2d(man_ch, 3, 1),
        )
        for mod in self.modules():  # D58: Kaiming is load-bearing under DN
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        with torch.no_grad():  # fixed Sobel must survive the Kaiming sweep
            pass  # (sobel/lum_w are buffers, not Conv2d — untouched)
        if prior_fg:
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.tent_head[-1].bias.zero_()
                self.tent_head[-1].bias[0] = b
                self.man_head[-1].bias.zero_()
                self.man_head[-1].bias[0] = b

    # ------------------------------------------------------------- freezing
    def donor_modules(self):
        """The v13-derived subgraph: frozen in run 1 (see class docstring)."""
        return (self.stem, self.stem_norm, self.down4, self.block4,
                self.down20, self.blocks, self.tent_head)

    def freeze_donor(self):
        """Freeze donor WEIGHTS and DeployNorm STATS together — the D39/16.1
        hard law (the v14 adapter collapsed because stats chased an upstream
        adapter while frozen weights could not follow). Returns (n_params,
        n_norms) for the caller to log."""
        n_p = n_n = 0
        for mod in self.donor_modules():
            for p in mod.parameters():
                p.requires_grad_(False)
                n_p += 1
            for m in mod.modules():
                if isinstance(m, DeployNorm):
                    m.frozen = True
                    n_n += 1
        return n_p, n_n

    def forward(self, img):
        """(B,3,540,960) -> dict of two grids:
        mann_* on 54x96 (stride 10), tent_* on 27x48 (stride 20)."""
        x2 = self.act(self.stem_norm(self.stem(img)))          # (B,16,270,480)
        # tent path — bit-for-bit v13_best while the donor is frozen
        t = self.blocks(self.down20(self.block4(self.down4(x2))))
        tent = self.tent_head(t)                                # (B,3,27,48)
        # mannequin path — fine grid off the undiluted s2 tap + anisotropy
        a = self.aniso(img)                                     # (B,1,270,480)
        m = self.act(self.man_norm(self.man_proj(torch.cat([x2, a], 1))))
        m = self.man_head(self.man_block(m))                    # (B,3,54,96)
        return {"mann_heat": m[:, 0:1], "mann_offset": m[:, 1:3],
                "tent_heat": tent[:, 0:1], "tent_offset": tent[:, 1:3]}


class V22Backbone(nn.Module):
    """v22 (D72-D75): "grown, not retrained" — peak-augmented full-rank funnel
    growth of v13_best. Three measured facts pin the design:

      1. §16.2: v13 UNDERFITS its own training data (train ≈ test at
         0.83/0.59-decile) — capacity, correctly placed, is the only open
         lever (data/distill/training are measured-closed).
      2. §16.3: the indicted site is down20 — every fine feature funnels
         through a depthwise-5x5 average + one 2,048-param 1x1, a rank
         constraint YOLO26n never commits. The D64 full-rank fix was
         pre-registered but never successfully trained: from-scratch-at-
         scale is 0-for-6 in this project.
      3. D62: worst-decile misses peak at heat 0.2-0.3 — evidence DILUTED
         by strided averaging, not absent. A full-rank LINEAR projection
         restores rank but provably cannot reconstruct a MAX statistic;
         peak evidence needs its own path.

    So v22 GROWS the trained v13 instead of retraining at scale: the
    full-rank funnel is a zero-gamma-valved PARALLEL branch beside the
    donor's intact down20 —

        x20 = down20(x_s4)                                   [donor path]
            + 2*tanh(spd_gain) * SiLU(DN(  spd_proj(x_s4)    [new, valved]
                                 + peak_proj(max_pool2d(x_s4, 5, 5))))

    where spd_proj is a full-rank Conv2d(ch_mid, ch_top, 5, stride=5) —
    mathematically IDENTICAL to pixel_unshuffle(5) + 1x1 over 25*ch_mid
    channels (the D64 honesty note, used as an optimization this time:
    the fused conv skips materializing the 800-channel unshuffle tensor
    and the 832-channel concat, measured -0.5ms/frame batch-1 MPS, and
    removes pixel_unshuffle from the graph entirely, so v22 is NOT in
    the ROCm inductor miscompile family and compile stays ON). peak_proj
    (1x1 on the max-pooled map) is the same linear map as the 32 extra
    concat columns — split out so the peak mechanism has its own named
    weight tensor for the post-training column autopsy.

    plus v18's train-only bg-aux head (the family fp-record earner; run-1
    trains with ANET_BG_W=0 so the capacity/peak verdict is unconfounded —
    the D69 interference law — and run-2 turns it on against v17_best/
    v18_best as the pre-existing ablation baselines). Two red-team-removed
    pieces, recorded so they are not re-proposed: a down4-level peak blend
    (unlicensed — D62/D64 indicted down20, not down4 — and measured as the
    LARGEST latency cost: elementwise passes over the 2M-element s2 map beat
    the whole 69M-MAC funnel on eager wall-clock) and four standalone bias-
    recalibration tensors (v17's bias win is an ADAPTER-regime result; in a
    full fine-tune every DeployNorm bias is already trainable, so separate
    bias params are redundant dof at real dispatch cost — the D74 protocol
    is a post-training bias-only adapter phase, not architecture).

    At step 0 a warm-started v22 IS v13_best bit-exactly — ALL donor
    tensors land, including every DeployNorm running-stat buffer (legal
    because no donor module's input distribution changes until a valve
    opens). This is the family's first capacity increase under the full
    D63 identity contract: it answers the 0-for-6 from-scratch record and
    the capacity ceiling with the same move. spd_norm is fresh and
    observes real branch activations from the first forward (the valve
    sits AFTER the norm — the D63 zero-gamma idiom, not a zeroed conv).
    spd_gain is tanh-BOUNDED (x2, range (-2,2)): the third-time law says
    every scalar that reshapes the trunk is bounded by construction, and
    unlike the historical zero-gamma valves this one gates 68% of the
    model's new capacity — an 80-epoch drift has to be unrepresentable,
    not merely unlikely (red-team blocker, resolved here).

    Pre-registered controls (ARCHITECTURE.md §17): a plain-SPD sibling
    (no maxpool concat) isolates peak-vs-rank; per-channel gain/column
    autopsies attribute each mechanism; peak-thresh sweep for fp claims.
    Deploy: conv/pool/space-to-depth/DN/SiLU only — single-shot NPU
    graph, ~0.05% of Hailo-8 int8 peak at 216.7M MACs (1.47x v13).
    78,844 deploy params (+65 train-only bg head) at spec width — on the
    pre-registered D65 curve near tier-S. Full fine-tune only: freezing
    the trunk would strand the fresh funnel between frozen stages (the
    measured 16.1 adapter failure mode). spd_proj keeps that NAME so the
    trainer's 0.2x slow-LR group (the v15 funnel-stability fix) matches.
    ROCm: pixel_unshuffle shape family -> compile defaults OFF (v15/v20
    precedent)."""

    def __init__(self, head_width=24, prior_fg=None, n_blocks=3,
                 channels=(16, 32, 64), zero_gain_blocks=0):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels
        self.channels = tuple(channels)
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = nn.SiLU()
        self.down4 = _DWSep(ch_stem, ch_mid, k=3, stride=2)
        self.block4 = _DWSep(ch_mid, ch_mid, k=3, stride=1)
        self.down20 = _DWSep(ch_mid, ch_top, k=5, stride=5)
        # D88: the last `zero_gain_blocks` s20 blocks are zero-gamma-valved,
        # so growing depth over a trained v22 is an EXACT identity at step 0
        # (D63 contract, the same move that made v22's funnel growth work).
        # Depth is the measured lever: §21.5 fit linear probes at every stage
        # of v22_d85_best and the s20 blocks carry the discrimination —
        # post-funnel tail 0.00444 -> post-blocks 0.00004, a 100x drop — while
        # the funnel itself adds nothing over its own input (s4 max-pool
        # 0.00441 -> post-funnel 0.00444) yet holds 65% of the parameters.
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1,
                     zero_gain=(i >= n_blocks - zero_gain_blocks))
              for i in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        # D72: full-rank peak-augmented funnel branch. spd_proj IS
        # pixel_unshuffle(5)+1x1 fused into one conv (identical linear map,
        # D64 honesty note); peak_proj is the maxpool side-channel's own
        # matrix. spd_proj keeps its NAME for the trainer's 0.2x slow-LR
        # group (fan-in 800-equivalent); peak_proj (fan-in ch_mid) trains
        # at full LR — the tiny new mechanism should not be slowed.
        self.spd_proj = nn.Conv2d(ch_mid, ch_top, 5, stride=5, bias=False)
        self.peak_proj = nn.Conv2d(ch_mid, ch_top, 1, bias=False)
        self.spd_norm = DeployNorm(ch_top)
        # tanh-bounded valve (x2): zero-init identity, range (-2, 2) — the
        # bounded-by-construction law applied to the one gain that gates
        # ~68% of all new capacity.
        self.spd_gain = nn.Parameter(torch.zeros(1, ch_top, 1, 1))
        # D74: v18's bg-mask aux head (train-only, dropped at eval/export).
        self.bg_head = nn.Conv2d(ch_top, 1, 1)
        for mod in self.modules():  # Kaiming everywhere (D58, load-bearing)
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
        x_s4 = self.block4(self.down4(x))
        z = self.spd_proj(x_s4) + self.peak_proj(F.max_pool2d(x_s4, 5, 5))
        branch = self.act(self.spd_norm(z))
        x = self.down20(x_s4) + 2.0 * torch.tanh(self.spd_gain) * branch
        y = self.blocks(x)
        out = self.head(y)
        if self.training:
            return out, self.bg_head(y)
        return out


class V20Backbone(nn.Module):
    """v20 (D70): the RE-RENDER CYCLE trunk (owner direction) — both of
    v13's strided transitions become an explicit embed -> unembed pair:

        conv stage -> EMBED (lossless space-to-depth + 1x1 funnel into a
        narrow latent, LearnedAct) -> UNEMBED (cheap 1x1 expansion back to
        a full-width feature image, identity-init QuatShift remix)
        -> conv stage -> repeat.

    The bottleneck (E=16 << 4*ch_stem=64 and 25*ch_mid=800) forces the next
    stage's input to be a RE-RENDERED visual: information survives the
    transition only by being re-encoded, and the unembed stays cheap
    (owner: "unembed is just expanding it back up cheaply"). Assembled from
    measured-good parts only: pixel_unshuffle descent (D64's lossless
    projection), bounded LearnedAct (D69-B), identity-init QuatShift
    (D60/D63), Kaiming init (D58, load-bearing), DeployNorm throughout.

    Everything OUTSIDE the two pairs keeps v13's exact shapes — stem,
    block4, blocks, head warm-start from a v13 checkpoint via strict=False
    (~20.5k of 25.2k donor params land). The s4->s20 funnel is named
    spd_proj ON PURPOSE: the trainer's slow-LR group (0.2x, the measured
    v15 stability fix for exactly this fan-in-800 layer) matches by name.
    ROCm: pixel_unshuffle shapes are the v15 inductor-miscompile family ->
    presets default compile OFF for v20 too.
    Hailo: space-to-depth, 1x1 convs, one-LUT activations, foldable
    quaternions — all legal. ~36.5k params at spec.
    """

    def __init__(self, head_width=24, prior_fg=None, n_blocks=3,
                 channels=(16, 32, 64), e1=16, e2=16):
        super().__init__()
        ch_stem, ch_mid, ch_top = channels
        self.channels = tuple(channels)
        self.stem = nn.Conv2d(3, ch_stem, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(ch_stem)
        self.act = nn.SiLU()
        # cycle 1: s2 -> s4 (480x270 -> 240x135)
        self.embed1 = nn.Conv2d(4 * ch_stem, e1, 1, bias=False)
        self.embed1_norm = DeployNorm(e1)
        self.embed1_act = LearnedAct()
        self.unembed1 = nn.Conv2d(e1, ch_mid, 1, bias=False)
        self.unembed1_norm = DeployNorm(ch_mid)
        self.qshift1 = QuatShift(ch_mid)
        self.block4 = _DWSep(ch_mid, ch_mid, k=3, stride=1)
        # cycle 2: s4 -> s20 (240x135 -> 48x27)
        self.spd_proj = nn.Conv2d(25 * ch_mid, e2, 1, bias=False)
        self.spd_norm = DeployNorm(e2)
        self.spd_act = LearnedAct()
        self.unembed2 = nn.Conv2d(e2, ch_top, 1, bias=False)
        self.unembed2_norm = DeployNorm(ch_top)
        self.qshift2 = QuatShift(ch_top)
        self.blocks = nn.Sequential(
            *[_DWSep(ch_top, ch_top, k=5, stride=1) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(ch_top, head_width, 1),
            nn.SiLU(),
            nn.Conv2d(head_width, 4, 1),
        )
        for mod in self.modules():
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
        z = self.embed1_act(self.embed1_norm(
            self.embed1(F.pixel_unshuffle(x, 2))))
        x = self.qshift1(self.act(self.unembed1_norm(self.unembed1(z))))
        x = self.block4(x)
        z = self.spd_act(self.spd_norm(
            self.spd_proj(F.pixel_unshuffle(x, 5))))
        x = self.qshift2(self.act(self.unembed2_norm(self.unembed2(z))))
        return self.head(self.blocks(x))
