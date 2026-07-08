import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CosineGate, ManualBatchNorm, bounded_cos_score


class MixRound(nn.Module):
    """One context round: gaussian-blurred cosine gate -> gated window mean
    added to RGB only; (u,v) coordinate channels pass through frozen (D7, D8).

    Two equivalent forward paths: `forward` on unfolded window tokens, and
    `forward_dense` on dense phase images (4-phase formulation, D25) — the
    dense path is ~an order of magnitude faster in training and matches the
    Hailo deployment graph."""

    GRID = 20
    KSIZE = 9

    def __init__(self, dim=5):
        super().__init__()
        self.bn = ManualBatchNorm(dim)
        self.V = nn.Parameter(torch.randn(3, dim) * 0.2)
        self.phi = nn.Parameter(torch.tensor(math.pi / 2))
        self.raw_sigma = nn.Parameter(torch.tensor(2.5))  # softplus+0.5 ~ 3.1 px

    def _kernel1d(self, device, dtype):
        sigma = F.softplus(self.raw_sigma) + 0.5
        r = torch.arange(-(self.KSIZE // 2), self.KSIZE // 2 + 1, device=device, dtype=dtype)
        g = torch.exp(-(r * r) / (2 * sigma * sigma))
        return g / g.sum()

    def _blur_tiles(self, t):  # (M, 1, 20, 20) separable gaussian, zero-pad
        g = self._kernel1d(t.device, t.dtype)
        t = F.conv2d(t, g.reshape(1, 1, self.KSIZE, 1), padding=(self.KSIZE // 2, 0))
        return F.conv2d(t, g.reshape(1, 1, 1, self.KSIZE), padding=(0, self.KSIZE // 2))

    def _blur(self, s):  # (N, 400)
        return self._blur_tiles(s.reshape(-1, 1, self.GRID, self.GRID)).reshape(s.shape)

    def _blur_matrix(self, device, dtype):
        """The 1-d tile blur as a banded GRID x GRID matrix: K[j,i] = g[j-i+4]
        (zero beyond the 9-tap band == the zero padding at tile borders)."""
        g = self._kernel1d(device, dtype)
        eye = torch.eye(self.GRID, device=device, dtype=dtype)
        return F.conv1d(eye.unsqueeze(1), g.reshape(1, 1, self.KSIZE),
                        padding=self.KSIZE // 2).squeeze(1)

    def _blur_dense(self, s):  # (B, C, H, W), H/W multiples of 20: per-tile blur
        # Separable blur within aligned 20-blocks == a block-diagonal matrix per
        # axis. Each contiguous 20-slice is a GEMM row: reshape(-1,20) @ K does
        # the whole W axis in one matmul (no per-tile convs, no permute storm);
        # the H axis is the same trick after one transpose. ~10x less dispatch
        # and pure GEMM work vs blurring 5,184 separate (1,20,20) tiles.
        K = self._blur_matrix(s.device, s.dtype)
        t = (s.reshape(-1, self.GRID) @ K).reshape(s.shape)
        t = t.transpose(-1, -2)
        t = (t.reshape(-1, self.GRID) @ K).reshape(t.shape)
        return t.transpose(-1, -2)

    def forward(self, x):  # (N, 400, dim)
        n, t, c = x.shape
        xn = self.bn(x.reshape(-1, c)).reshape(n, t, c)
        s = torch.einsum("ntc,kc->nkt", xn, self.V)
        s1, s2, s3 = s.unbind(1)
        score = bounded_cos_score(self._blur(s1), self._blur(s2), s3, self.phi)
        gate = torch.sigmoid(score)
        pooled = (gate.unsqueeze(-1) * x).mean(1)  # (N,5)
        rgb = F.silu(x[..., :3] + pooled[:, None, :3])
        return torch.cat([rgb, x[..., 3:]], -1)

    def _fold(self):
        """Eval-BN as a per-channel affine (running stats): scale, shift."""
        scale = self.bn.weight * torch.rsqrt(self.bn.running_var + self.bn.eps)
        shift = self.bn.bias - self.bn.running_mean * scale
        return scale, shift

    def forward_dense(self, x):  # (B, dim, H, W), H/W multiples of 20 — same math
        s = F.conv2d(self.bn(x), self.V.reshape(3, -1, 1, 1))
        s12 = self._blur_dense(s[:, :2])  # s1, s2 blurred in one pass
        gate = torch.sigmoid(bounded_cos_score(s12[:, 0:1], s12[:, 1:2], s[:, 2:3], self.phi))
        pooled = F.avg_pool2d(gate * x, self.GRID)  # gated window mean (B,5,nh,nw)
        up = F.interpolate(pooled[:, :3], scale_factor=self.GRID, mode="nearest")
        rgb = F.silu(x[:, :3] + up)
        return torch.cat([rgb, x[:, 3:]], 1)

    def dense_round_rgb(self, rgb, frozen_score):
        """Eval fast path: same round math on the 3 RGB channels only.
        `frozen_score` is this round's precomputed V·BN(frozen channels) map
        (they never change across rounds), the BN affine is folded into the
        RGB conv, and the gated pool runs on RGB alone — the round never
        touches the 8 frozen channels (original pooled[:, :3] slice == pooling
        only RGB)."""
        scale, _ = self._fold()
        w = (self.V * scale)[:, :3]
        s = F.conv2d(rgb, w.reshape(3, 3, 1, 1)) + frozen_score
        s12 = self._blur_dense(s[:, :2])
        gate = torch.sigmoid(bounded_cos_score(s12[:, 0:1], s12[:, 1:2], s[:, 2:3], self.phi))
        pooled = F.avg_pool2d(gate * rgb, self.GRID)
        up = F.interpolate(pooled, scale_factor=self.GRID, mode="nearest")
        return F.silu(rgb + up)

    def reg_l2(self):
        return (self.V[1] ** 2).sum() + (self.V[2] ** 2).sum()


class WindowEncoder(nn.Module):
    """Shared 20x20 window encoder: 400 (r,g,b,hp1..hp3,u,v) tokens -> embedding.
    in_dim=8 at spec: 3 quat-RGB + 3 high-pass texture channels (D32) + (u,v).
    The non-RGB channels pass through the mixing rounds frozen."""

    def __init__(self, hidden=16, in_dim=8):
        super().__init__()
        self.hidden = hidden
        self.in_dim = in_dim
        self.rounds = nn.ModuleList([MixRound(dim=in_dim) for _ in range(3)])
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.bn = ManualBatchNorm(hidden)
        self.gate = CosineGate(hidden)

    def forward(self, x):  # (N, 400, in_dim) -> (N, hidden)
        for r in self.rounds:
            x = r(x)
        h = self.mlp(x)
        n, t, c = h.shape
        hn = self.bn(h.reshape(-1, c)).reshape(n, t, c)
        return (self.gate(hn).unsqueeze(-1) * h).mean(1)

    def forward_dense(self, x):  # (B, in_dim, H, W) -> (B, hidden, H/20, W/20)
        if self.training:
            for r in self.rounds:
                x = r.forward_dense(x)
        else:
            # eval fast path: the frozen channels' score contributions for all
            # 3 rounds in ONE conv over the 8 frozen channels (constant across
            # rounds), then each round runs on the 3 RGB channels only
            rgb, frozen = x[:, :3], x[:, 3:]
            ws, bs = [], []
            for r in self.rounds:
                scale, shift = r._fold()
                ws.append((r.V * scale)[:, 3:])
                bs.append(r.V @ shift)
            fs = F.conv2d(frozen, torch.cat(ws).reshape(-1, self.in_dim - 3, 1, 1),
                          torch.cat(bs))
            for i, r in enumerate(self.rounds):
                rgb = r.dense_round_rgb(rgb, fs[:, 3 * i : 3 * i + 3])
            x = torch.cat([rgb, frozen], 1)
        # per-token MLP == 1x1 convs with the shared Linear weights
        fc1, fc2 = self.mlp[0], self.mlp[2]
        h = F.silu(F.conv2d(x, fc1.weight.reshape(self.hidden, self.in_dim, 1, 1), fc1.bias))
        h = F.silu(F.conv2d(h, fc2.weight.reshape(self.hidden, self.hidden, 1, 1), fc2.bias))
        if self.training:
            s = F.conv2d(self.bn(h), self.gate.V.reshape(3, self.hidden, 1, 1))
        else:  # fold eval-BN affine into the gate conv (as in MixRound._score_conv)
            scale = self.bn.weight * torch.rsqrt(self.bn.running_var + self.bn.eps)
            shift = self.bn.bias - self.bn.running_mean * scale
            s = F.conv2d(h, (self.gate.V * scale).reshape(3, self.hidden, 1, 1),
                         self.gate.V @ shift)
        gate = torch.sigmoid(bounded_cos_score(s[:, 0:1], s[:, 1:2], s[:, 2:3], self.gate.phi))
        return F.avg_pool2d(gate * h, MixRound.GRID)

    def reg_l2(self):
        return sum(r.reg_l2() for r in self.rounds) + self.gate.reg_l2()
