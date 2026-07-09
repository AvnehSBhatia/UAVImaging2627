import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CosineGate, ManualBatchNorm


class RegionHead(nn.Module):
    """Per-window classifier (D14, D19, D31). The window's own evidence (its
    embedding + 3 Path-A vectors) and the 16 image-global tokens are pooled in
    SEPARATE gated streams and concatenated: the 16 global tokens are identical
    for every window of a frame, so pooling them together with the 4 per-window
    tokens caps window-specific signal at 4/20 of the vector and the head
    degenerates into an image classifier (measured: constant logits, 0.000
    mannequin recall). Cosine-gated pooling, no QK matmul."""

    def __init__(self, dim=18, n_classes=3, prior_fg=None):
        super().__init__()
        self.local_bn = ManualBatchNorm(dim)
        self.local_gate = CosineGate(dim)
        self.ctx_bn = ManualBatchNorm(dim)
        self.ctx_gate = CosineGate(dim)
        self.fc1 = nn.Linear(2 * dim, 8)
        self.fc2 = nn.Linear(8, n_classes)
        # prior-bias init (RetinaNet §4.1): start each foreground class at a
        # small NONZERO probability so the head isn't at the saturated all-
        # background point where the softmax Jacobian p·(1−p)→0 kills the
        # foreground-logit gradient. Without it, a 99%-background grid drives
        # the foreground logits to −∞ in the first steps and they never recover
        # (measured: soft p(fg)=0.000, model stuck all-bg). prior_fg=0.1 ->
        # p_bg~0.8, p_mann~p_tent~0.1 at init. None = default zero-bias.
        if prior_fg:
            import math
            b = math.log(prior_fg / max(1.0 - (n_classes - 1) * prior_fg, 1e-6))
            with torch.no_grad():
                self.fc2.bias.zero_()
                self.fc2.bias[1:] = b   # class 0 = background stays at 0

    def forward(self, ltoks, gtoks):  # (B,W,4,d), (B,16,d) -> (B,W,3)
        b, w, t, c = ltoks.shape
        ln = self.local_bn(ltoks.reshape(-1, c)).reshape(b, w, t, c)
        loc = (self.local_gate(ln).unsqueeze(-1) * ln).mean(2)  # (B, W, d)
        gn = self.ctx_bn(gtoks.reshape(-1, c)).reshape(gtoks.shape)
        ctx = (self.ctx_gate(gn).unsqueeze(-1) * gn).mean(1)  # (B, d)
        # fc1(cat[loc, ctx]) as split matmuls + broadcast add: same math, but
        # Expand(ctx to W windows)+Concat forced a CPU partition in the CoreML EP
        h = loc @ self.fc1.weight[:, :c].t() + \
            (ctx @ self.fc1.weight[:, c:].t() + self.fc1.bias).unsqueeze(1)
        h = torch.tanh(F.silu(h))  # SiLU -> Tanh (D20)
        return self.fc2(h)

    def reg_l2(self):
        return self.local_gate.reg_l2() + self.ctx_gate.reg_l2()
