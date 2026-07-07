import math

import torch
import torch.nn as nn


class GlobalCosineMix(nn.Module):
    """Multi-cosine mixing of the 3 level states (D17). Softmax kept — this
    block runs on the Pi CPU in exact fp32 at deploy (771 params, KB tensors).
    256 splits into 16 tokens x 16-d (D18)."""

    def __init__(self, dim=256, n_tokens=16, token_dim=16, pad_to=18):
        super().__init__()
        assert n_tokens * token_dim == dim and pad_to > token_dim
        self.U = nn.Parameter(torch.randn(3, dim) * 0.05)
        self.phi = nn.Parameter(torch.tensor(math.pi / 2))
        self.pad = nn.Parameter(torch.zeros(pad_to - token_dim))  # tokens -> head dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim

    def forward(self, states):  # (B, 3, 256) -> (B, 16, 18)
        s = torch.matmul(states, self.U.t())  # s[b,i,j] = U_j . v_i (einsum-free, D34)
        s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2]
        # w_i = sum_j s1_j * cos(pi*tanh(s2_j * s3_i) + phi)  — cross-vector weave
        arg = torch.tanh(s2.unsqueeze(-1) * s3.unsqueeze(-2))  # (B, j, i)
        w = (s1.unsqueeze(-1) * torch.cos(math.pi * arg + self.phi)).sum(1)  # (B, 3)
        g = torch.softmax(w, -1)
        mixed = (g.unsqueeze(-1) * states).sum(1)  # (B, 256)
        toks = mixed.reshape(-1, self.n_tokens, self.token_dim)
        pad = self.pad.expand(toks.shape[0], self.n_tokens, -1)
        return torch.cat([toks, pad], -1)

    def reg_l2(self):
        return (self.U[1] ** 2).sum() + (self.U[2] ** 2).sum()
