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

    CH_STEM, CH_MID, CH_TOP = 16, 32, 64

    def __init__(self, head_width=24, prior_fg=None, n_blocks=3):
        super().__init__()
        self.stem = nn.Conv2d(3, self.CH_STEM, 3, 2, 1, bias=False)
        self.stem_norm = DeployNorm(self.CH_STEM)
        self.act = nn.SiLU()
        self.down4 = _DWSep(self.CH_STEM, self.CH_MID, k=3, stride=2)
        self.block4 = _DWSep(self.CH_MID, self.CH_MID, k=3, stride=1)
        self.down20 = _DWSep(self.CH_MID, self.CH_TOP, k=5, stride=5)
        self.blocks = nn.Sequential(
            *[_DWSep(self.CH_TOP, self.CH_TOP, k=5, stride=1)
              for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(self.CH_TOP, head_width, 1),
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
    weighted sum

        y' = y * (w_pass + w_gate * g),   w_pass init 1, w_gate init 0

    so the layer is exact identity at init (D63) and training decides, per
    channel, how much texture-conditioned suppression/boost to apply. All ops
    fold-legal: depthwise conv, mul (square), DeployNorm affine, avg-pool,
    1x1 convs, sigmoid, mul."""

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
        self.w_pass = nn.Parameter(torch.ones(1, ch_out, 1, 1))
        self.w_gate = nn.Parameter(torch.zeros(1, ch_out, 1, 1))

    def forward(self, x_s4, y):  # texture stats from s4 modulate the s20 trunk
        e = self.hp(x_s4)
        e = self.norm(e * e)                              # local texture energy
        e = F.avg_pool2d(e, 5, 5)                         # 135x240 -> 27x48
        g = torch.sigmoid(self.fc2(self.act(self.fc1(e))))
        return y * (self.w_pass + self.w_gate * g)


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
