"""P1: pointwise colour algebra as a segmenter (OBSERVATIONS.md P1).

Owner spec: take each 40x40 (mannequin) / 100x100 (tent) region and learn a
series of dual-quaternion transforms + non-linear thresholding/activations
until the GT boxes are white and the background pure black.

The probe is deliberately pointwise everywhere — NO spatial context at all.
It measures how far colour-space algebra alone goes at separating rendered
assets from OAM backgrounds; whatever it cannot separate is, by
elimination, texture/shape work the conv blocks must be doing.

Reuses the family's proven pieces verbatim: DualQuaternionRGB (D5, folds to
a 1x1 conv at export) and the bounded LearnedAct (D69-B, one Hailo LUT).
~120 params total.
"""

import torch.nn as nn

from ..model.backbone import LearnedAct
from ..model.blocks import DualQuaternionRGB


class WhiteboxDQ(nn.Module):
    def __init__(self, stages=4):
        super().__init__()
        self.dq = nn.ModuleList(DualQuaternionRGB() for _ in range(stages))
        self.act = nn.ModuleList(LearnedAct() for _ in range(stages))
        self.out = nn.Conv2d(3, 1, 1)

    def forward(self, x):  # (B,3,S,S) in [0,1] -> per-pixel logits (B,1,S,S)
        for dq, act in zip(self.dq, self.act):
            x = act(dq(x))
        return self.out(x)
