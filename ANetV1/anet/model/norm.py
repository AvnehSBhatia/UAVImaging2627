"""DeployNorm — deploy-form normalization (v9, D39).

BatchNorm's training mode normalizes with *batch* statistics, which couples
every tile to every other tile in the batch. That coupling is what made the
encoder unfusable (each mixing round forced a full-resolution round trip to
HBM so the next round could see batch stats), triggered the MIOpen int32
overflow at batch >= 44, and made checkpointing double-update stats.

DeployNorm normalizes with the RUNNING statistics — exactly the affine the
deploy graph uses after BN folding — and updates those statistics as a
detached EMA of the batch stats it observes. Consequences:

  - the training forward is the deployment forward, bit-for-bit ("train what
    you deploy", taken to its conclusion: there is no train/eval BN gap left);
  - within a step the normalization is a constant per-channel affine, so the
    whole encoder becomes tile-local and fuses into one kernel (D40);
  - no gradient flows through the statistics (they are running buffers). The
    learnable affine (weight, bias) trains normally. At batch ~96 x 4 phases
    x 1296 tiles x 400 tokens the batch stats are averages over ~10^8 values,
    so the EMA is glassy smooth and the one-step lag is negligible.

Cold start: the first `forward` seeds the buffers from the first batch
(cumulative-average momentum ramp), and the trainer additionally runs a few
no-grad seeding passes before step 0 (`Trainer._seed_norm_stats`) so training
never normalizes against init-garbage stats.

Buffer updates are DEFERRED: observe() only stashes the batch stats; the
trainer calls apply_norm_updates(model) after each backward(). Mutating the
buffers inside the forward->backward window breaks torch.compile — AOT
autograd's partitioner saves the buffer tensors themselves and rematerializes
the fold in backward, so an in-step mutation trips the version counter
(reproduced; eager hides it because fold() clones). Anyone training a model
outside the Trainer must call apply_norm_updates(model) after backward, or
the stats silently never move.

Folds into adjacent convs at export exactly like BatchNorm (same buffers).
"""

import torch
import torch.nn as nn


def apply_norm_updates(model):
    """Apply every DeployNorm's stashed batch stats to its running buffers.
    Call after backward() (the trainer does) — never inside the step."""
    for m in model.modules():
        if isinstance(m, DeployNorm):
            m.apply_pending()


class DeployNorm(nn.Module):
    def __init__(self, num_features, momentum=0.05, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.momentum = momentum
        self.eps = eps
        self._pending = None  # (mean, var) stashed by observe(), applied post-step

    # ------------------------------------------------------------- statistics
    @torch.no_grad()
    def _update(self, mean, var):
        # cumulative average for the first ~1/momentum steps (seeds fast from
        # the true distribution), then a plain EMA
        m = max(self.momentum, 1.0 / float(self.steps.item() + 1))
        if self.steps.item() == 0:
            self.running_mean.copy_(mean)
            self.running_var.copy_(var)
        else:
            self.running_mean.lerp_(mean.to(self.running_mean.dtype), m)
            self.running_var.lerp_(var.to(self.running_var.dtype), m)
        self.steps += 1

    @staticmethod
    def _stats(x, channels_last):
        xf = x.float()
        if x.dim() == 4 and not channels_last:  # channels-first
            return xf.mean((0, 2, 3)), xf.var((0, 2, 3), unbiased=False)
        xf = xf.reshape(-1, xf.shape[-1])  # channels-last (tokens)
        return xf.mean(0), xf.var(0, unbiased=False)

    @torch.no_grad()
    def observe(self, x, channels_last=False):
        """Stash this batch's stats for the post-backward update.
        channels_last=False reads (B, C, H, W) / (N, C); True reads (..., C)."""
        self._pending = self._stats(x, channels_last)

    @torch.no_grad()
    def observe_parts(self, *parts):
        """Stash stats from channel-partitioned views of the same token set
        (e.g. the RGB stream + the frozen channels, in channel order): per-
        part stats, ONE EMA step — identical semantics to the fused kernel's
        concatenated stat update."""
        means, vs = [], []
        for x in parts:
            m, v = self._stats(x, channels_last=False)
            means.append(m)
            vs.append(v)
        self._pending = (torch.cat(means), torch.cat(vs))

    def set_pending(self, mean, var):
        """Fused-kernel stat hook: stash externally computed batch stats."""
        self._pending = (mean, var)

    @torch.no_grad()
    def apply_pending(self):
        if self._pending is not None:
            self._update(*self._pending)
            self._pending = None

    @torch.no_grad()
    def update_from_sums(self, s, ss, n):
        """EMA update from kernel-accumulated per-channel sums: s = sum(x),
        ss = sum(x^2) over n values per channel (the fused path's stat hook)."""
        mean = s / n
        var = (ss / n - mean * mean).clamp_min_(0.0)
        self._update(mean, var)

    # ---------------------------------------------------------- normalization
    def fold(self):
        """(scale, shift) of the running-stat affine: y = x*scale + shift.
        Buffers are cloned so the later in-place EMA update can't invalidate
        tensors autograd saved for the weight/bias backward."""
        scale = self.weight * torch.rsqrt(self.running_var.clone() + self.eps)
        shift = self.bias - self.running_mean.clone() * scale
        return scale, shift

    def forward(self, x):  # (B, C, H, W) channels-first
        # fold-then-observe: normalization always uses the stats from BEFORE
        # this batch, exactly like the fused kernel (which reads the buffers,
        # accumulates sums, and updates after the step). The trainer seeds the
        # buffers with a few no-grad passes before step 0.
        scale, shift = self.fold()
        y = x * scale.reshape(1, -1, 1, 1).to(x.dtype) + \
            shift.reshape(1, -1, 1, 1).to(x.dtype)
        if self.training:
            self.observe(x)
        return y

    def forward_tokens(self, x):  # (..., C) channels-last
        scale, shift = self.fold()
        y = x * scale.to(x.dtype) + shift.to(x.dtype)
        if self.training:
            self.observe(x, channels_last=True)
        return y
