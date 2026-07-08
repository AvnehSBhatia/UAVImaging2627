import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ManualBatchNorm


class ScalarKernelPool(nn.Module):
    """k x k neighborhood map feeding Path A.

    per_channel=False (D13 spec): ONE learned scalar per position, shared across
    all channels — 179 params for the three scales.
    per_channel=True  (D37, viz upgrade): a full depthwise k x k kernel PER
    channel, box-filter-initialised so it starts bit-identical to the shared
    form and only specialises from there. The three context scales can now
    weight each embedding channel differently — the pre-registered response to
    mushy/under-filled tent blobs and mannequin/car scale confusion (ARCH §8
    risk 2 second step). ~4.5k params at d=26; still tiny next to Path B."""

    def __init__(self, channels, k, per_channel=False):
        super().__init__()
        self.channels = channels
        self.k = k
        self.per_channel = per_channel
        c = channels if per_channel else 1
        self.weight = nn.Parameter(torch.full((c, 1, k, k), 1.0 / (k * k)))

    def forward(self, m):  # (B, C, H, W)
        w = self.weight if self.per_channel else \
            self.weight.expand(self.channels, 1, self.k, self.k)
        return F.conv2d(m, w, padding=self.k // 2, groups=self.channels)

    def reg_l1(self):
        # keep the per-kernel L1 pressure identical to the shared form (D24) so
        # per_channel doesn't silently multiply the penalty by `channels`
        s = self.weight.abs().sum()
        return s / self.channels if self.per_channel else s


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
