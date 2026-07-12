"""v9 window encoder (D39/D40/D42) — same signature mechanisms as v8
(cosine-gated mixing rounds, gaussian-lens gates, cosine-gated pooling), made
tile-local so the whole thing fuses into one kernel:

  - DeployNorm everywhere (running-stat affine; no batch coupling, D39);
  - the frozen channels' round scores are precomputed ONCE for all 3 rounds
    even in training (legal now that the norm is an affine — v8 could only do
    this at eval);
  - fc2 moves AFTER the gated pool (D42): the per-token stage is
    fc1 (in_dim -> h1) + gate + pool, and the h1 -> hidden layer runs on the
    5,035 pooled windows instead of 2M full-res positions. The data-dependent
    gate is still a second nonlinearity applied at token level, and h1 > hidden
    keeps the pre-pool width; this removes ~45% of full-res FLOPs and the
    single biggest activation tensor.

Three equivalent paths:
  forward_tokens  — (N, 400, in_dim) reference (parity tests, windowed path)
  pool_features_dense / forward_dense — dense PyTorch path (any device)
  the fused Triton op (anet/train/fused.py) substitutes pool_features_dense
  exactly, so fc2 and everything downstream is shared.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .blocks import CosineGate, bounded_cos_score
from .norm import DeployNorm


class MixRoundV9(nn.Module):
    """One context round on the RGB stream (frozen channels enter through
    their precomputed score contribution). Math identical to v8's MixRound
    with the BN replaced by the DeployNorm affine."""

    GRID = 20
    KSIZE = 9

    def __init__(self, dim=17):
        super().__init__()
        self.dim = dim
        self.norm = DeployNorm(dim)
        self.V = nn.Parameter(torch.randn(3, dim) * 0.2)
        self.phi = nn.Parameter(torch.tensor(math.pi / 2))
        self.raw_sigma = nn.Parameter(torch.tensor(2.5))  # softplus+0.5 ~ 3.1 px

    def kernel1d(self, device, dtype):
        sigma = F.softplus(self.raw_sigma) + 0.5
        r = torch.arange(-(self.KSIZE // 2), self.KSIZE // 2 + 1,
                         device=device, dtype=dtype)
        g = torch.exp(-(r * r) / (2 * sigma * sigma))
        return g / g.sum()

    def blur_matrix(self, device, dtype):
        """Tile blur as a banded GRID x GRID matrix (zero-pad at tile borders)."""
        g = self.kernel1d(device, dtype)
        eye = torch.eye(self.GRID, device=device, dtype=dtype)
        return F.conv1d(eye.unsqueeze(1), g.reshape(1, 1, self.KSIZE),
                        padding=self.KSIZE // 2).squeeze(1)

    def _blur_dense(self, s):  # (B, C, H, W), H/W multiples of 20
        K = self.blur_matrix(s.device, s.dtype)
        t = (s.reshape(-1, self.GRID) @ K).reshape(s.shape)
        t = t.transpose(-1, -2)
        t = (t.reshape(-1, self.GRID) @ K).reshape(t.shape)
        return t.transpose(-1, -2)

    def score_convs(self):
        """Fold the norm affine into the score dots: s = Wr*rgb + Wf*frozen + b.
        Returns (Wr (3,3), Wf (3,dim-3), b (3,)). Snapshot these BEFORE any
        observe() so every consumer of this step sees the same affine."""
        scale, shift = self.norm.fold()
        Vs = self.V * scale
        return Vs[:, :3], Vs[:, 3:], self.V @ shift

    def forward_dense_rgb(self, rgb, frozen_score, Wr=None):
        """(B,3,H,W) + this round's precomputed frozen score (B,3,H,W).
        Wr: pre-snapshotted folded RGB score conv (pre-observation affine)."""
        if Wr is None:
            Wr = self.score_convs()[0]
        s = F.conv2d(rgb, Wr.reshape(3, 3, 1, 1).to(rgb.dtype)) + frozen_score
        s12 = self._blur_dense(s[:, :2])
        gate = torch.sigmoid(
            bounded_cos_score(s12[:, 0:1], s12[:, 1:2], s[:, 2:3], self.phi))
        pooled = F.avg_pool2d(gate * rgb, self.GRID)
        up = F.interpolate(pooled, scale_factor=self.GRID, mode="nearest")
        return F.silu(rgb + up)

    def forward_tokens(self, x):  # (N, 400, dim) reference
        scale, shift = self.norm.fold()
        xn = x * scale.to(x.dtype) + shift.to(x.dtype)
        s = torch.einsum("ntc,kc->nkt", xn, self.V)
        K = self.blur_matrix(x.device, x.dtype)

        def blur(v):  # (N, 400)
            t = v.reshape(-1, self.GRID, self.GRID)
            t = torch.einsum("nij,jk->nik", t, K)
            t = torch.einsum("nij,ik->nkj", t, K)
            return t.reshape(v.shape)

        s1, s2, s3 = s.unbind(1)
        score = bounded_cos_score(blur(s1), blur(s2), s3, self.phi)
        gate = torch.sigmoid(score)
        pooled = (gate.unsqueeze(-1) * x[..., :3]).mean(1)
        rgb = F.silu(x[..., :3] + pooled[:, None, :])
        return torch.cat([rgb, x[..., 3:]], -1)

    def reg_l2(self):
        return (self.V[1] ** 2).sum() + (self.V[2] ** 2).sum()


class TileEncoder(nn.Module):
    """Shared 20x20 window encoder v9: 400 (rgb, edge..., u, v) tokens ->
    hidden-d embedding. in_dim = stem channels + 2 (uv)."""

    # Dynamic-box pooling (v11, D55): the cosine-gate readout is normalized by its
    # own gate mass, Σ(gate·h)/(Σgate + BOX_EPS), instead of the fixed 400-px
    # window area (avg_pool). The sigmoid cosine-gate already computes a soft,
    # data-dependent object mask per 20x20 window; dividing by its mass reads
    # out the MEAN INSIDE THE SOFT BOX, so a mannequin covering 1-2% of the
    # window is recovered at full strength instead of being averaged ~50-90x
    # into the background (the measured small-object collapse). Large/uniform
    # objects (tents) are unchanged — for a near-uniform gate the mass-mean
    # equals the area-mean. This PRESERVES the cosine-gated-pooling mechanism
    # (D42) and only replaces the normalizer; it stays two avg-pools + a divide
    # (Hailo-legal) and is ~a no-op at init (gate ~0.5 -> mass-mean ~ area-mean).
    # BOX_EPS is in the SUM domain (a fraction of one virtual gate-pixel), kept
    # identical across the token / dense / fused paths so their parity holds.
    BOX_EPS = 0.5

    def __init__(self, hidden=32, h1=48, in_dim=17):
        super().__init__()
        self.hidden = hidden
        self.h1 = h1
        self.in_dim = in_dim
        self.rounds = nn.ModuleList([MixRoundV9(dim=in_dim) for _ in range(3)])
        self.fc1 = nn.Linear(in_dim, h1)
        self.pool_norm = DeployNorm(h1)
        self.gate = CosineGate(h1)
        self.fc2 = nn.Linear(h1, hidden)

    # -------------------------------------------------------------- helpers
    def _frozen_scores(self, frozen):
        """All 3 rounds' frozen-channel score contributions in one conv:
        (B, dim-3, H, W) -> (B, 9, H, W), rows [round0 s1..s3, round1 ..., ...].
        The full score bias (incl. the RGB shift term) rides here."""
        ws, bs = [], []
        for r in self.rounds:
            _, Wf, b = r.score_convs()
            ws.append(Wf)
            bs.append(b)
        w = torch.cat(ws).reshape(-1, self.in_dim - 3, 1, 1).to(frozen.dtype)
        return F.conv2d(frozen, w, torch.cat(bs).to(frozen.dtype))

    # ---------------------------------------------------------- dense path
    def pool_features_dense(self, x, ckpt=False):
        """(B, in_dim, H, W) -> (B, h1, H/20, W/20) gated-pooled window
        features. This is exactly the region the fused Triton op replaces."""
        rgb, frozen = x[:, :3], x[:, 3:]
        # snapshot every round's folded affine BEFORE any stat observation so
        # the whole step normalizes with the pre-batch stats (fused-kernel
        # semantics; keeps dense fallback and Triton path bit-comparable)
        rgb_convs = [r.score_convs()[0] for r in self.rounds]
        fs = self._frozen_scores(frozen)
        for i, r in enumerate(self.rounds):
            if self.training:  # stats: each round's own RGB input; frozen is
                with torch.no_grad():  # identical every round (one EMA step)
                    r.norm.observe_parts(rgb.detach(), frozen.detach())
            f = fs[:, 3 * i: 3 * i + 3]
            if ckpt:
                rgb = torch.utils.checkpoint.checkpoint(
                    r.forward_dense_rgb, rgb, f, rgb_convs[i], use_reentrant=False)
            else:
                rgb = r.forward_dense_rgb(rgb, f, rgb_convs[i])
        x = torch.cat([rgb, frozen], 1)
        if ckpt:
            # non-reentrant checkpoint re-executes the wrapped fn during
            # backward; the stateful observe() must fire exactly once, so a
            # one-shot closure disables it on the recompute pass
            fired = []

            def tail(t):
                observe = not fired
                fired.append(True)
                return self._pool_tail(t, observe=observe)

            return torch.utils.checkpoint.checkpoint(
                tail, x, use_reentrant=False)
        return self._pool_tail(x)

    def _pool_tail(self, x, observe=True):  # (B,in_dim,H,W) -> (B,h1,H/20,W/20)
        fc1 = self.fc1
        h = F.silu(F.conv2d(x, fc1.weight.reshape(self.h1, self.in_dim, 1, 1)
                            .to(x.dtype), fc1.bias.to(x.dtype)))
        scale, shift = self.pool_norm.fold()  # pre-batch affine (fused semantics)
        if self.training and observe:
            self.pool_norm.observe(h)
        s = F.conv2d(h, (self.gate.V * scale).reshape(3, self.h1, 1, 1).to(h.dtype),
                     (self.gate.V @ shift).to(h.dtype))
        gate = torch.sigmoid(
            bounded_cos_score(s[:, 0:1], s[:, 1:2], s[:, 2:3], self.gate.phi))
        # dynamic-box readout: mean of h INSIDE the soft gate box (mass-
        # normalized), not the 400-px area mean. avg_pool of both numerator and
        # denominator keeps it in the avg domain, so the sum-domain BOX_EPS is
        # scaled by the window area to stay bit-consistent with the token path.
        g = MixRoundV9.GRID
        num = F.avg_pool2d(gate * h, g)
        den = F.avg_pool2d(gate, g)
        return num / (den + self.BOX_EPS / (g * g))

    def embed(self, pooled):  # (B, h1, nh, nw) -> (B, hidden, nh, nw)
        fc2 = self.fc2
        return F.silu(F.conv2d(pooled,
                               fc2.weight.reshape(self.hidden, self.h1, 1, 1)
                               .to(pooled.dtype), fc2.bias.to(pooled.dtype)))

    def forward_dense(self, x, ckpt=False):
        return self.embed(self.pool_features_dense(x, ckpt=ckpt))

    # ------------------------------------------------------ token reference
    def forward_tokens(self, x):  # (N, 400, in_dim) -> (N, hidden)
        for r in self.rounds:
            x = r.forward_tokens(x)
        h = F.silu(self.fc1(x))
        scale, shift = self.pool_norm.fold()
        hn = h * scale.to(h.dtype) + shift.to(h.dtype)
        gate = self.gate(hn)  # (N, 400)
        # dynamic-box readout (see BOX_EPS): mass-normalized gated mean, the
        # sum-domain twin of _pool_tail's avg-domain form (parity-checked).
        num = (gate.unsqueeze(-1) * h).sum(1)
        den = gate.sum(1, keepdim=True)
        pooled = num / (den + self.BOX_EPS)
        return F.silu(self.fc2(pooled))

    def reg_l2(self):
        return sum(r.reg_l2() for r in self.rounds) + self.gate.reg_l2()
