import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CosineGate, ManualBatchNorm


class RegionHead(nn.Module):
    """Per-window classifier over 20 tokens: 16 global + own embedding +
    own 3 Path-A vectors (D14, D19). Cosine-gated pooling, no QK matmul."""

    def __init__(self, dim=18, n_classes=3):
        super().__init__()
        self.gate = CosineGate(dim)
        self.bn = ManualBatchNorm(dim)
        self.fc1 = nn.Linear(dim, 8)
        self.fc2 = nn.Linear(8, n_classes)

    def forward(self, toks):  # (B, W, T, 18) -> (B, W, 3)
        pooled = (self.gate(toks).unsqueeze(-1) * toks).mean(2)  # (B, W, 18)
        b, w, c = pooled.shape
        pooled = self.bn(pooled.reshape(-1, c)).reshape(b, w, c)
        h = torch.tanh(F.silu(self.fc1(pooled)))  # 18 -> SiLU -> Tanh (D20)
        return self.fc2(h)

    def reg_l2(self):
        return self.gate.reg_l2()
