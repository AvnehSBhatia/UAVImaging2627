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

from .norm import DeployNorm


class _DWSep(nn.Module):
    """Depthwise-separable unit: dw kxk -> DN -> SiLU -> pw 1x1 -> DN -> SiLU.
    Residual only when it changes nothing about shape (stride 1, ch_in==ch_out).
    Padding: k//2 keeps H/W (stride 1) or halves it exactly on even sizes
    (stride 2: 270->135, 480->240); the stride==k case (dw5x5 s5) tiles the
    grid exactly with padding 0 (135/5=27, 240/5=48 — 540 = 2*2*5 * 27, no
    half-pixel alignment anywhere)."""

    def __init__(self, ch_in, ch_out, k, stride):
        super().__init__()
        self.residual = stride == 1 and ch_in == ch_out
        pad = 0 if stride == k else k // 2
        self.dw = nn.Conv2d(ch_in, ch_in, k, stride, pad, groups=ch_in, bias=False)
        self.dw_norm = DeployNorm(ch_in)
        self.pw = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        self.pw_norm = DeployNorm(ch_out)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.dw_norm(self.dw(x)))
        y = self.act(self.pw_norm(self.pw(y)))
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
