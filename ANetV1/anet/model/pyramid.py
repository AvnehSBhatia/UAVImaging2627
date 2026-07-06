import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ManualBatchNorm


class ScalarKernelPool(nn.Module):
    """k x k neighborhood sum with one learned scalar per position, shared
    across channels (D13). Same-resolution output feeds Path A."""

    def __init__(self, channels, k):
        super().__init__()
        self.channels = channels
        self.k = k
        self.weight = nn.Parameter(torch.full((1, 1, k, k), 1.0 / (k * k)))

    def forward(self, m):  # (B, C, H, W)
        w = self.weight.expand(self.channels, 1, self.k, self.k)
        return F.conv2d(m, w, padding=self.k // 2, groups=self.channels)

    def reg_l1(self):
        return self.weight.abs().sum()


class GatedGlobalPool(nn.Module):
    """Path B per level: 1x1-conv scorer -> sigmoid gate -> global mean ->
    Linear 18->256. Pool-then-expand (linear ops commute, D16)."""

    def __init__(self, dim=18, out=256):
        super().__init__()
        self.bn = ManualBatchNorm(dim)
        self.scorer = nn.Sequential(
            nn.Conv2d(dim, 8, 1), nn.SiLU(),
            nn.Conv2d(8, 4, 1), nn.SiLU(),
            nn.Conv2d(4, 1, 1),
        )
        self.expand = nn.Linear(dim, out)

    def forward(self, m):  # (B, 18, H, W) -> (B, 256)
        gate = torch.sigmoid(self.scorer(self.bn(m)))
        return self.expand((gate * m).mean((2, 3)))
