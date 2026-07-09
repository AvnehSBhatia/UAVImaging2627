import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


class DualQuaternionRGB(nn.Module):
    """Rigid transform of RGB space. Deploys as a constant 3x3 conv + bias:
    call .matrix() at export and bake into a 1x1 conv (ARCHITECTURE.md D5)."""

    def __init__(self):
        super().__init__()
        self.qr = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.qd = nn.Parameter(torch.zeros(4))

    def matrix(self):
        q = self.qr / self.qr.norm().clamp_min(1e-8)
        w, x, y, z = q.unbind(-1)
        r = torch.stack(
            [
                torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
                torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
                torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
            ]
        )
        conj = q * q.new_tensor([1.0, -1.0, -1.0, -1.0])
        t = 2.0 * quat_mul(self.qd, conj)[1:]
        return r, t

    def forward(self, img):  # (B,3,H,W)
        # einsum("ij,bjhw->bihw") as a 1x1 conv: identical math, and the ORT
        # CoreML EP has no Einsum builder (einsum forced a CPU partition)
        r, t = self.matrix()
        return F.conv2d(img, r.reshape(3, 3, 1, 1), t)

    @torch.no_grad()
    def to_conv(self):
        """Bake to a constant 1x1 conv for export (D5). The quaternion algebra
        in matrix() (nested stack/unbind on parameter scalars) is constant at
        inference and chokes the coremltools trace frontend."""
        r, t = self.matrix()
        conv = nn.Conv2d(3, 3, 1)
        conv.weight.copy_(r.reshape(3, 3, 1, 1))
        conv.bias.copy_(t)
        return conv


def fuse_dq(module):
    """Recursively replace every DualQuaternionRGB with its baked 1x1 conv.
    Call on a deepcopy before tracing/converting for deployment."""
    for name, child in module.named_children():
        if isinstance(child, DualQuaternionRGB):
            setattr(module, name, child.to_conv())
        else:
            fuse_dq(child)
    return module


class ManualBatchNorm(nn.Module):
    """BatchNorm built from primitive ops. The fused MPS NativeBatchNormBackward
    kernel rejects some stride patterns in this graph; primitives are stride-safe.
    Same math and running-stat tracking as nn.BatchNorm*, folds at export.
    Accepts (N, C) or (B, C, H, W)."""

    def __init__(self, num_features, momentum=0.05, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        # fused native BN (cuDNN/MIOpen/CPU): identical math and running-stat
        # convention (lerp(momentum) == (1-m)*run + m*batch; unbiased var in
        # the running buffer). The win is autograd: the fused kernel saves the
        # input + two per-channel vectors, where the primitive-op form below
        # saves TWO full-res fp32 intermediates per call — measured 1.35 GiB/img
        # of the 2.87 GiB/img training footprint. Stats still accumulate in fp32
        # internally for half inputs, so the bf16 two-pass-variance concern that
        # motivated the manual form's .float() does not apply.
        # Two cases MUST take the primitive path instead:
        #   - MPS: the fused kernel rejects some stride patterns in this graph.
        #   - numel >= 2^31: MIOpen's BatchNorm kernel indexes N*C*H*W with a
        #     32-bit int; past INT32_MAX it wraps negative and fails its own
        #     static_assert(NCHW>=0) at HIP compile time (miopenStatusUnknownError).
        #     Hits the full-res embed BN (4B·24·540·960) at batch >= ~44.
        if x.device.type != "mps" and x.numel() < 2**31:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, self.training,
                                self.momentum, self.eps)
        if x.dim() == 4:
            dims, shape = (0, 2, 3), (1, -1, 1, 1)
        else:
            dims, shape = (0,), (1, -1)
        # stats and normalization in fp32 regardless of autocast (matches the
        # autocast policy for native batch_norm; half-precision reductions over
        # ~2M spatial elements are not trustworthy, and lerp_ rejects Half)
        xf = x.float()
        if self.training:
            mean = xf.mean(dims)
            # two-pass variance: non-negative by construction. E[x^2]-E[x]^2
            # cancels catastrophically once inputs are bf16-rounded (~0.4% rel
            # err) — near-constant channels went var<-1e-5 -> rsqrt=NaN on
            # MI300X. Also still avoids the Welford kernel inductor-MPS
            # miscompiles (torch 2.12).
            var = (xf - mean.reshape(shape)).square().mean(dims)
            with torch.no_grad():
                n = x.numel() / x.shape[1 if x.dim() == 4 else -1]
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / max(n - 1.0, 1.0), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        xhat = (xf - mean.reshape(shape)) * torch.rsqrt(var.reshape(shape) + self.eps)
        return (xhat * self.weight.reshape(shape) + self.bias.reshape(shape)).to(x.dtype)


def bounded_cos_score(s1, s2, s3, phi):
    # tanh keeps the cos argument inside one period -> int8-LUT-safe on Hailo (D6)
    return s1 * torch.cos(math.pi * torch.tanh(s2 * s3) + phi)


def sobel7(orient):
    """7x7 separable Sobel/Scharr edge kernel. 'v' -> d/dx (fires on vertical
    edges), 'h' -> d/dy (horizontal edges). Used to init the oriented edge convs
    (learnable from there)."""
    smooth = torch.tensor([1.0, 4.0, 8.0, 10.0, 8.0, 4.0, 1.0])
    smooth = smooth / smooth.sum()
    deriv = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) / 6.0
    return torch.outer(smooth, deriv) if orient == "v" else torch.outer(deriv, smooth)


class EdgeDQStem(nn.Module):
    """Oriented-edge dual-quaternion front-end (D33). Injects the texture/edge
    evidence a colour-only encoder lacks — the probe showed mannequin signal is
    in the 540p pixels but not in the embeddings.

    Triplicate the frame; leave one copy raw; send the other two through a
    learned dual-quaternion colour rotation then a 7x7 FREELY-LEARNED conv
    (D35: two learned filters, not fixed Sobel — the data picks the texture
    operators, initialised small so they grow from ~0 rather than injecting
    noise); stack -> 9ch; apply 'one more learned DQ' per stacked image
    (block-diagonal, keeps the colour/edge grouping clean so the encoder still
    updates only the 3 colour channels and reads the learned-texture channels as
    frozen evidence). Every op is a dense conv that bakes to a constant at export
    -> Hailo-legal (D5-style), the NPU's favourite op class."""

    out_channels = 9

    def __init__(self):
        super().__init__()
        self.dq_v = DualQuaternionRGB()  # learned colour frame before each filter
        self.dq_h = DualQuaternionRGB()
        # freely-learned 7x7 depthwise texture filters (D35), small init so the
        # channels start near zero and the encoder ignores them until they train
        self.edge_v = nn.Conv2d(3, 3, 7, padding=3, groups=3, bias=False)
        self.edge_h = nn.Conv2d(3, 3, 7, padding=3, groups=3, bias=False)
        with torch.no_grad():
            self.edge_v.weight.mul_(0.2)
            self.edge_h.weight.mul_(0.2)
        # "one more learned DQ" after stacking, per stacked image (block-diagonal)
        self.dq_out = nn.ModuleList([DualQuaternionRGB() for _ in range(3)])

    def forward(self, img):  # (B,3,H,W) -> (B,9,H,W): [colour, tex1, tex2]
        groups = (img, self.edge_v(self.dq_v(img)), self.edge_h(self.dq_h(img)))
        return torch.cat([dq(g) for dq, g in zip(self.dq_out, groups)], 1)


class CosineGate(nn.Module):
    """3 shared dots -> s1*cos(pi*tanh(s2*s3)+phi) -> sigmoid token weights (D10)."""

    def __init__(self, dim, init_std=0.2):
        super().__init__()
        self.V = nn.Parameter(torch.randn(3, dim) * init_std)
        self.phi = nn.Parameter(torch.tensor(math.pi / 2))  # alive-at-init (D6)

    def forward(self, x):  # (..., T, dim) -> (..., T)
        # matmul instead of einsum (no CoreML Einsum builder); same math
        s = torch.matmul(x, self.V.t()).transpose(-1, -2)  # (..., 3, T)
        s1, s2, s3 = s.unbind(-2)
        return torch.sigmoid(bounded_cos_score(s1, s2, s3, self.phi))

    def reg_l2(self):
        # bounding the s2/s3 vectors bounds the cosine frequency (D24)
        return (self.V[1] ** 2).sum() + (self.V[2] ** 2).sum()
