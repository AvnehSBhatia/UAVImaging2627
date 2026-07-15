import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CosineGate, ManualBatchNorm
from .norm import DeployNorm


def prior_bias_(fc, prior_fg, n_classes=3):
    """RetinaNet-style prior init: start each foreground class at probability
    ~prior_fg so the head is off the saturated all-background point."""
    b = math.log(prior_fg / max(1.0 - (n_classes - 1) * prior_fg, 1e-6))
    with torch.no_grad():
        fc.bias.zero_()
        fc.bias[1:] = b  # class 0 = background stays at 0


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


class RegionHeadV9(nn.Module):
    """v9 head (D45). Keeps D31's split streams — per-window evidence is
    pooled separately from the frame context and owns half the classifier
    input unconditionally — but widens the classifier from the 8-d choke to
    `width` (default 24): the per-cell decision is where the discrimination
    the encoder built must survive, and 8 dims with a Tanh was the narrowest
    point in the entire network. The context stream is now the SlimContext
    vector (one d-dim vector per frame), so no context pooling is needed.
    Tanh kept before the final layer (int8 calibration, D20).

    v11 metric-prototype classifier (D56, proto=True). The final Linear(width,
    3) is replaced by a distance-to-prototype readout in the bounded Tanh
    metric space z:

        logit_c = scale * (2 z·p_c - ‖p_c‖²) + prior_c
                = -scale·‖z - p_c‖² + scale·‖z‖² + prior_c

    The +scale·‖z‖² term is IDENTICAL across classes at a cell, so it cancels
    under softmax/argmax and is dropped — leaving an exactly deployable linear
    map (weight_c = 2·scale·p_c, bias_c = prior_c − scale·‖p_c‖²), so the head
    still folds to one conv on Hailo (no runtime L2-norm, affine-foldable). The
    point is NOT extra capacity (a linear layer has the same freedom) but that
    the classifier weights ARE the class prototypes: they are shaped by the
    supervised-contrastive metric objective (train/losses.proto_metric_loss)
    that clusters mannequin / tent / background in z BEFORE detection precision
    tuning, then keep receiving that metric gradient jointly. That is what lets
    the tiny under-represented mannequin separate from the easy tent instead of
    being swallowed by it (the measured p(mann)→0 collapse). `logits_z` exposes
    z + prototypes so the trainer can compute the metric loss on the same space
    the decision uses."""

    def __init__(self, dim, width=24, n_classes=3, prior_fg=None, proto=True):
        super().__init__()
        self.n_classes = n_classes
        self.width = width
        self.proto = proto
        self.local_norm = DeployNorm(dim)
        self.local_gate = CosineGate(dim)
        # ctx is ONE d-vector per image (SlimContext already pooled space away),
        # so ctx_norm sees only B samples/channel — ~20,000x fewer than every
        # other DeployNorm (which see B*W*... ~ 1e5). At momentum 0.05 its
        # running stats random-walk ~4% every step from pure sampling noise, and
        # that noise folds into a scale/shift added IDENTICALLY to all ~5035
        # windows of the image (head.py forward broadcast) — a globally-coherent
        # logit wobble, exactly the shape of the 0<->185k argmax swing. Slower
        # momentum averages the small sample harder (v10 stability fix).
        self.ctx_norm = DeployNorm(dim, momentum=0.01)
        self.fc1 = nn.Linear(2 * dim, width)
        if proto:
            # class prototypes in the width-d Tanh metric space (‖·‖<√width).
            # small init so distances start comparable and softmax isn't saturated
            self.prototypes = nn.Parameter(torch.randn(n_classes, width)
                                           * (width ** -0.5))
            self.proto_log_scale = nn.Parameter(torch.zeros(()))  # softplus→~0.69
            self.proto_bias = nn.Parameter(torch.zeros(n_classes))  # class prior
            if prior_fg:
                b = math.log(prior_fg / max(1.0 - (n_classes - 1) * prior_fg, 1e-6))
                with torch.no_grad():
                    self.proto_bias[1:] = b
        else:
            self.fc2 = nn.Linear(width, n_classes)
            if prior_fg:
                prior_bias_(self.fc2, prior_fg, n_classes)

    def _z(self, ltoks, ctx):  # (B,W,4,d),(B,d) -> (B,W,width) metric embedding
        b, w, t, c = ltoks.shape
        ln = self.local_norm.forward_tokens(ltoks)
        loc = (self.local_gate(ln).unsqueeze(-1) * ln).mean(2)  # (B, W, d)
        cn = self.ctx_norm.forward_tokens(ctx)  # (B, d)
        # fc1(cat[loc, ctx]) as split matmuls + broadcast add (CoreML-safe, D31)
        h = loc @ self.fc1.weight[:, :c].t() + \
            (cn @ self.fc1.weight[:, c:].t() + self.fc1.bias).unsqueeze(1)
        return torch.tanh(F.silu(h))

    def _logits(self, z):  # (B,W,width) -> (B,W,3)
        if not self.proto:
            return self.fc2(z)
        scale = F.softplus(self.proto_log_scale)
        # deployable linear form of -scale·‖z-p‖² (the ‖z‖² term cancels in
        # softmax/argmax and is dropped so this is exactly a conv at export)
        weight = 2.0 * scale * self.prototypes                       # (3, width)
        bias = self.proto_bias - scale * (self.prototypes ** 2).sum(-1)  # (3,)
        return z @ weight.t() + bias

    def forward(self, ltoks, ctx):  # (B,W,4,d),(B,d) -> (B,W,3)
        return self._logits(self._z(ltoks, ctx))

    def logits_z(self, ltoks, ctx):  # -> (logits (B,W,3), z (B,W,width))
        """Training entry: returns decisions AND the metric embedding so the
        proto_metric_loss can shape z / prototypes on the decision space."""
        z = self._z(ltoks, ctx)
        return self._logits(z), z

    def metric_scale(self):
        return F.softplus(self.proto_log_scale)

    def reg_l2(self):
        return self.local_gate.reg_l2()


class CenterHead(nn.Module):
    """v12 head: object center-heatmap detector (see workflow spec / planned
    ARCHITECTURE.md delta). Reuses RegionHeadV9's split-stream `_z` verbatim —
    local DeployNorm + CosineGate gated-mean pool over the 4 local tokens
    concatenated (via split matmuls, D31) with the ctx DeployNorm(momentum=
    0.01) stream, SiLU->Tanh — because that metric embedding is exactly the
    per-window evidence a center/offset readout needs too; only the final
    projection changes; there is no prototype path here; scale/proto losses
    are not this head's job.

    fc2 projects the width-d embedding to 4 raw logits per window:
      [center_mannequin, center_tent, dx, dy]
    center_* are independent-sigmoid heatmap logits (no softmax competition,
    per v12 spec); dx/dy are class-agnostic sub-cell offset logits (squashed
    through sigmoid downstream by offset_l1, not here) — kept as raw logits
    out of this module so the head still folds to one plain Linear/1x1-conv
    at export (D20/D31)."""

    def __init__(self, dim, width=24, prior_fg=None):
        super().__init__()
        self.width = width
        self.local_norm = DeployNorm(dim)
        self.local_gate = CosineGate(dim)
        # see RegionHeadV9.ctx_norm docstring for why this momentum is slow:
        # ctx is one d-vector/image, so a faster EMA random-walks noise into
        # every window's logit identically (measured argmax instability).
        self.ctx_norm = DeployNorm(dim, momentum=0.01)
        self.fc1 = nn.Linear(2 * dim, width)
        self.fc2 = nn.Linear(width, 4)
        if prior_fg:
            # RetinaNet-style prior (see prior_bias_ above), but here BOTH
            # output channels 0/1 are independent foreground sigmoids (no
            # background class to leave at 0), so bias = logit(prior_fg)
            # directly rather than the softmax-normalized form in
            # prior_bias_/RegionHead. dx/dy offset channels start at 0.
            b = math.log(prior_fg / max(1.0 - prior_fg, 1e-6))
            with torch.no_grad():
                self.fc2.bias.zero_()
                self.fc2.bias[0:2] = b
                self.fc2.bias[2:4] = 0.0

    def _z(self, ltoks, ctx):  # (B,W,4,d),(B,d) -> (B,W,width) — identical to
        # RegionHeadV9._z (kept as a literal copy, not a shared call, so this
        # head has no import/coupling dependency on RegionHeadV9's proto path)
        b, w, t, c = ltoks.shape
        ln = self.local_norm.forward_tokens(ltoks)
        loc = (self.local_gate(ln).unsqueeze(-1) * ln).mean(2)  # (B, W, d)
        cn = self.ctx_norm.forward_tokens(ctx)  # (B, d)
        # fc1(cat[loc, ctx]) as split matmuls + broadcast add (CoreML-safe, D31)
        h = loc @ self.fc1.weight[:, :c].t() + \
            (cn @ self.fc1.weight[:, c:].t() + self.fc1.bias).unsqueeze(1)
        return torch.tanh(F.silu(h))

    def forward(self, ltoks, ctx):  # (B,W,4,d),(B,d) -> (B,W,4) raw logits
        return self.fc2(self._z(ltoks, ctx))

    def reg_l2(self):
        return self.local_gate.reg_l2()
