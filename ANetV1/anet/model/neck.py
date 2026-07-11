"""ConvNeck (v9, D43) — cross-window context on the 53x95 embedding grid.

The v8 head could only see other windows through Path-A box averages and one
per-frame context vector; the 000008 viz showed features firing on objects the
head could not resolve at the right scale. Two residual depthwise-separable
rounds give every window a learned 50-110 px receptive field of *trainable*
spatial structure (vs Path A's fixed-shape averages) for ~4k params. Depthwise
+ 1x1 convs are the Hailo DFC's favourite ops; the residual starts near zero
(pointwise init scaled down) so the neck is a no-op at init and cannot disturb
the cold start.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import DeployNorm


class ConvNeck(nn.Module):
    def __init__(self, dim, rounds=2, k=5):
        super().__init__()
        self.rounds = rounds
        self.norms = nn.ModuleList([DeployNorm(dim) for _ in range(rounds)])
        self.dw = nn.ModuleList(
            [nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
             for _ in range(rounds)]
        )
        self.pw = nn.ModuleList(
            [nn.Conv2d(dim, dim, 1) for _ in range(rounds)]
        )
        with torch.no_grad():
            for conv in self.pw:  # near-identity start: residual ~ 0 at init
                conv.weight.mul_(0.1)
                conv.bias.zero_()

    def forward(self, m):  # (B, d, 53, 95)
        for norm, dw, pw in zip(self.norms, self.dw, self.pw):
            m = m + pw(F.silu(dw(norm(m))))
        return m
