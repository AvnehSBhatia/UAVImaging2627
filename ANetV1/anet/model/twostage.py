"""v21 (D71, owner spec): the two-stage filter-front-end detector.

Pipeline, exactly as directed (ARCHITECTURE.md 16.9):

  1. sample 20 random rows + 20 random cols -> mean (R,G,B) in [0,1]
     (train: random per step; eval: fixed stride — deterministic)
  2. three learned 11x11 matrices, image-conditioned by elementwise power:
     K_k = thresh(A_k ^ chan_k) with A_k = exp(W_k) (positive by
     construction, so the power is exp(chan*W) — clamped to +-4, the D24
     one-period discipline applied to exponents), thresh = relu(. - tau)
     ("if value below n set to 0", v17's form), then L1-normalized for
     scale stability. Per-image kernels -> per-sample grouped conv.
  3. triplicate the input, depthwise-conv each copy with its kernel
  4. mean RGB -> Linear(3,8) -> SiLU -> Linear(8,3) -> weights ->
     weighted sum of the 3 filtered images = 1 composite (weights init
     to exactly 1/3 each: composite starts as the plain average)
  5. quaternion #1 (DualQuaternionRGB, D5) — trained with a DEDICATED
     background-smoothness term (owner: "separate loss"; it sits in the
     main path so main-task gradients also reach it — a strictly
     separate loss for an in-path module would require cutting the main
     gradient, documented in 16.9)
  6. quaternion #2 — the learned-Sobel slot. NOTE: a pointwise
     quaternion cannot BE a spatial Sobel; it feeds a Sobel-init 7x7
     depthwise kernel (the D5/D33 EdgeDQStem pattern — quaternion
     rotation choosing WHICH colour axis the edge operator sees)
  7. saliency = channel L2 of the edge image, max-pooled 20x20 to the
     family 27x48 grid, affine-calibrated -> center focal loss (D57)
  8. peaks (3x3 local max, top-K) -> 100x100 crops from the filtered
     image -> tiny CNN -> {BG, mannequin, tent}. NO dense classifier
     (owner call): crops are gathered per detected center.

Deploy note (recorded, not enforced): per-image kernels are dynamic conv
weights — not Hailo-compilable as-is (basis-expansion fix in 16.9); the
crop gather is CPU-side. This is a research-track architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DualQuaternionRGB, sobel7

CROP = 100
GRID_H, GRID_W = 27, 48
STRIDE = 20
IMG_H, IMG_W = 540, 960


class CropCNN(nn.Module):
    """100x100x9 crop + 4 context scalars -> 3 logits (BG, mannequin,
    tent). ~5.7k params. The crop stacks EVERYTHING the pipeline knows
    about the window (v21.1, owner: "use all of our info"): raw RGB
    (color is the family's strongest class signal), the smoothed
    composite, and the edge image. The context vector rides past the
    conv stack into the head: the peak's saliency prob (stage-1
    confidence) + the frame's mean RGB (the same scene stats that
    condition the kernels). GroupNorm: the crop batch is small and
    variable (K peaks + GT + bg crops per step) — batch statistics
    would be noise."""

    def __init__(self, in_ch=9, n_ctx=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, 2, 1), nn.GroupNorm(4, 8), nn.SiLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(24 + n_ctx, 3)

    def forward(self, x, ctx):
        return self.fc(torch.cat([self.net(x), ctx], 1))


class V21TwoStage(nn.Module):
    def __init__(self, k=11, n_lines=20, max_peaks=12):
        super().__init__()
        self.k = k
        self.n_lines = n_lines
        self.max_peaks = max_peaks
        # small random init breaks the symmetry between the three kernels;
        # exp(chan*W) stays ~1 so the thresholded, normalized kernels start
        # near a box blur — a sane, finite cold start for the saliency path
        self.W = nn.Parameter(torch.randn(3, k, k) * 0.1)
        self.tau = nn.Parameter(torch.full((3,), 0.25))
        self.mix = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 3))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.constant_(self.mix[-1].bias, 1.0 / 3.0)
        self.quat_smooth = DualQuaternionRGB()   # 5: the smoothing quat
        self.quat_edge = DualQuaternionRGB()     # 6: the edge quat...
        # ...feeding the actual spatial operator: one Sobel-init 7x7
        # depthwise kernel per channel (sobel7 from the family stem)
        e = torch.stack([sobel7("v"), sobel7("h"), sobel7("d1")])
        self.edge = nn.Parameter(e.reshape(3, 1, 7, 7).clone())
        # saliency logit calibration (pooled edge energy -> focal logits)
        self.sal_a = nn.Parameter(torch.tensor(4.0))
        self.sal_b = nn.Parameter(torch.tensor(-2.0))
        self.crop_cnn = CropCNN()

    # ------------------------------------------------------------ stage 1
    def line_means(self, img):
        """20 random rows + 20 random cols -> per-channel mean (B,3).
        Eval: fixed strided lines so inference is deterministic."""
        B, _, H, W = img.shape
        if self.training:
            rows = torch.randint(0, H, (self.n_lines,), device=img.device)
            cols = torch.randint(0, W, (self.n_lines,), device=img.device)
        else:
            rows = torch.linspace(0, H - 1, self.n_lines,
                                  device=img.device).long()
            cols = torch.linspace(0, W - 1, self.n_lines,
                                  device=img.device).long()
        r = img[:, :, rows, :].mean((2, 3))   # (B,3)
        c = img[:, :, :, cols].mean((2, 3))
        return (r + c) / 2.0

    def kernels(self, means):
        """(B,3) channel means -> (B,3,k,k) thresholded L1-normed kernels."""
        expo = (means.reshape(-1, 3, 1, 1) * self.W.unsqueeze(0)).clamp(-4, 4)
        kern = F.relu(torch.exp(expo) - self.tau.reshape(1, 3, 1, 1))
        return kern / kern.sum((2, 3), keepdim=True).clamp_min(1e-6)

    def filter_bank(self, img, kern):
        """Triplicate the image, depthwise-conv copy k with kernel k
        (per-sample weights via one grouped conv). -> (B,3,3,H,W)."""
        B, C, H, W = img.shape
        outs = []
        flat = img.reshape(1, B * C, H, W)
        for i in range(3):
            w = kern[:, i].unsqueeze(1)                    # (B,1,k,k)
            w = w.repeat_interleave(C, dim=0)              # (B*3,1,k,k)
            y = F.conv2d(flat, w, padding=self.k // 2, groups=B * C)
            outs.append(y.reshape(B, C, H, W))
        return torch.stack(outs, 1)

    def forward(self, img):
        """Returns the training dict; peak/crop assembly lives in the
        trainer (data-dependent gather, deliberately outside the graph)."""
        means = self.line_means(img)                       # 1
        bank = self.filter_bank(img, self.kernels(means))  # 2-3
        w = self.mix(means)                                # 4
        composite = (bank * w.reshape(-1, 3, 1, 1, 1)).sum(1)
        smooth = self.quat_smooth(composite)               # 5
        edge = F.conv2d(self.quat_edge(smooth), self.edge,
                        padding=3, groups=3)               # 6
        sal = edge.norm(dim=1, keepdim=True)               # 7
        pooled = F.max_pool2d(sal, STRIDE).squeeze(1)      # (B,27,48)
        sal_logits = self.sal_a * pooled + self.sal_b
        return {"sal_logits": sal_logits, "smooth": smooth, "edge": edge,
                "means": means}

    @staticmethod
    def stack_maps(img, out):
        """The 9-channel crop source (v21.1): raw RGB + smoothed
        composite + edge image, concatenated once per batch."""
        return torch.cat([img, out["smooth"], out["edge"]], 1)

    # ------------------------------------------------------------ stage 2
    @torch.no_grad()
    def find_peaks(self, sal_prob, thresh=0.3):
        """(B,27,48) prob -> per-sample list of (row, col, p): 3x3 local
        maxima above thresh, strongest-first, capped at max_peaks."""
        m = F.max_pool2d(sal_prob.unsqueeze(1), 3, 1, 1).squeeze(1)
        mask = (sal_prob >= m) & (sal_prob > thresh)
        out = []
        for b in range(sal_prob.shape[0]):
            rs, cs = torch.nonzero(mask[b], as_tuple=True)
            vals = sal_prob[b, rs, cs]
            order = vals.argsort(descending=True)[: self.max_peaks]
            out.append([(int(rs[i]), int(cs[i]), float(vals[i]))
                        for i in order])
        return out

    @staticmethod
    def crop_at(edge_img, cx_px, cy_px):
        """100x100 crop around a full-res center, clamped inside the frame
        (same clamp geometry as the probes' PatchCrops)."""
        ox = int(min(max(round(cx_px - CROP / 2), 0), IMG_W - CROP))
        oy = int(min(max(round(cy_px - CROP / 2), 0), IMG_H - CROP))
        return edge_img[:, oy:oy + CROP, ox:ox + CROP]

    @torch.no_grad()
    def detect(self, img, peak_thresh=0.3):
        """Full inference: saliency peaks -> crops -> CropCNN. Emits the
        family (heat, offset) tensors — each detection writes its class
        prob at its peak cell — so CenterObjectMetrics and the ladder
        numbers stay apples-to-apples with v13..v20."""
        out = self.forward(img)
        prob = torch.sigmoid(out["sal_logits"])
        peaks = self.find_peaks(prob, peak_thresh)
        stacked = self.stack_maps(img, out)
        B = img.shape[0]
        heat = torch.zeros(B, 2, GRID_H, GRID_W)
        offset = torch.full((B, 2, GRID_H, GRID_W), 0.5)
        for b in range(B):
            if not peaks[b]:
                continue
            crops = torch.stack([
                self.crop_at(stacked[b], (c + 0.5) * STRIDE,
                             (r + 0.5) * STRIDE) for r, c, _ in peaks[b]])
            ctx = torch.stack([
                torch.cat([torch.tensor([v], device=img.device),
                           out["means"][b]]) for _, _, v in peaks[b]])
            cls_p = torch.softmax(self.crop_cnn(crops, ctx), 1)  # (P,3)
            for (r, c, _), p in zip(peaks[b], cls_p):
                if int(p.argmax()) > 0:
                    heat[b, int(p.argmax()) - 1, r, c] = float(p.max())
        return heat, offset
