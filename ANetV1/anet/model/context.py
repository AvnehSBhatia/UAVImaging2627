"""SlimContext (v9, D44) — global context without the Path-B parameter sink.

v8 spent 14.6k params (60% of the model) on three 18->256 expansions feeding
one per-frame vector that D31 showed was actively diluting per-window
evidence. v9 keeps the signature pieces — per-scale gated global pooling and
the multi-cosine state weave (still CPU-friendly: a handful of d-dim vectors)
— but the states stay at embedding width d and the mixed vector feeds the
head classifier directly. ~1.3k params for the same "what frame is this"
signal, freeing the budget for per-region discrimination.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import DeployNorm


class SlimGatedPool(nn.Module):
    """Per level: norm -> 1x1 scorer -> sigmoid gate -> global mean -> d-state."""

    def __init__(self, dim):
        super().__init__()
        self.norm = DeployNorm(dim)
        self.scorer = nn.Sequential(
            nn.Conv2d(dim, 8, 1), nn.SiLU(),
            nn.Conv2d(8, 4, 1), nn.SiLU(),
            nn.Conv2d(4, 1, 1),
        )

    def forward(self, m):  # (B, d, H, W) -> (B, d)
        gate = torch.sigmoid(self.scorer(self.norm(m)))
        return (gate * m).mean((2, 3))


class SlimCosineMix(nn.Module):
    """Multi-cosine weave over the 3 level states at width d (same math as
    GlobalCosineMix, no 256-d expansion / token split): each state contributes
    an (amplitude, frequency) pair, every state's probe is read under all
    three lenses, softmax over the weave -> mixed d-vector."""

    def __init__(self, dim):
        super().__init__()
        self.U = nn.Parameter(torch.randn(3, dim) * 0.05)
        self.phi = nn.Parameter(torch.tensor(math.pi / 2))

    def forward(self, states):  # (B, 3, d) -> (B, d)
        s = torch.matmul(states, self.U.t())  # (B, 3, 3): s[b,i,k] = U_k . v_i
        s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2]
        arg = torch.tanh(s2.unsqueeze(-1) * s3.unsqueeze(-2))  # (B, j, i)
        w = (s1.unsqueeze(-1) * torch.cos(math.pi * arg + self.phi)).sum(1)
        g = torch.softmax(w, -1)
        return (g.unsqueeze(-1) * states).sum(1)

    def reg_l2(self):
        return (self.U[1] ** 2).sum() + (self.U[2] ** 2).sum()


class SlimContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pools = nn.ModuleList([SlimGatedPool(dim) for _ in range(3)])
        self.mix = SlimCosineMix(dim)

    def forward(self, maps):  # list of 3 (B, d, H, W) -> (B, d)
        states = torch.stack([p(m) for p, m in zip(self.pools, maps)], 1)
        return self.mix(states)

    def reg_l2(self):
        return self.mix.reg_l2()
