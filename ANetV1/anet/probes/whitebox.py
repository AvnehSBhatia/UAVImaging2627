"""P1 whitebox segmenter — deep large-kernel conv stack (owner redirect).

Original P1 was strictly pointwise DualQuaternionRGB + LearnedAct (~50
params) to bound colour-only separability. Owner redirect: go deep with
large convolutions so the probe can actually paint boxes using spatial
context. Same task (per-pixel logit, GT boxes white / bg black) and the
same crop trainer; architecture is now a residual large-kernel tower.

Default: DQ colour front → 11×11 stem → 16 residual blocks cycling
7 / 11 / 15 kernels → 1×1 logit. Env: ANET_STAGES, ANET_CH, ANET_K
(comma list). ~0.5–2M params depending on width/depth.
"""

import torch
import torch.nn as nn

from ..model.backbone import LearnedAct
from ..model.blocks import DualQuaternionRGB


class _LargeResBlock(nn.Module):
    """Full k×k conv residual (not depthwise) + bounded LearnedAct."""

    def __init__(self, ch, k):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, k, padding=k // 2, bias=True)
        self.act = LearnedAct()
        # residual gate starts at 0 so the block is identity at init —
        # deep stacks of large kernels otherwise blow the first step.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x + torch.tanh(self.gate) * self.act(self.conv(x))


class WhiteboxDQ(nn.Module):
    def __init__(self, stages=16, width=32, kernels=(7, 11, 15)):
        super().__init__()
        if isinstance(kernels, str):
            kernels = tuple(int(x) for x in kernels.split(",") if x.strip())
        self.kernels = tuple(kernels)
        self.dq = DualQuaternionRGB()
        k0 = self.kernels[1] if len(self.kernels) > 1 else self.kernels[0]
        self.stem = nn.Conv2d(3, width, k0, padding=k0 // 2)
        self.stem_act = LearnedAct()
        self.blocks = nn.ModuleList(
            _LargeResBlock(width, self.kernels[i % len(self.kernels)])
            for i in range(stages))
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x):  # (B,3,S,S) in [0,1] -> per-pixel logits (B,1,S,S)
        h = self.stem_act(self.stem(self.dq(x)))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)

    @torch.no_grad()
    def intermediates(self, x):
        """RGB-ish maps for viz: stem + every 4th block (channel-mean)."""
        maps = []
        h = self.stem_act(self.stem(self.dq(x)))
        maps.append(h.mean(1, keepdim=True).expand(-1, 3, -1, -1))
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if (i + 1) % max(len(self.blocks) // 4, 1) == 0 or i + 1 == len(self.blocks):
                maps.append(h.mean(1, keepdim=True).expand(-1, 3, -1, -1))
        return maps, self.out(h)
