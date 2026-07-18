"""v21.2 (D71 ablation, owner-directed): the minimal quat-Sobel detector.

The v21/v21.1 front end (line-sampled means -> A^chan 11x11 kernels ->
blend MLP) and the ENTIRE crop stage are removed (owner: ablate 00-03,
"pass the raw image into 04"; cropping "ruins everything — remove").
What remains is the smallest detector in the family, ~170 params:

  raw image -> quaternion #1 (D5, dedicated background-smoothness loss)
            -> quaternion #2 + Sobel-init 7x7 depthwise (the D5/D33
               EdgeDQStem pattern: the quat picks WHICH colour axis the
               edge operator sees)
            -> per-channel |edge| energy, max-pooled 20x20 to the family
               27x48 grid
            -> 1x1 conv (3 -> 2): the minimal class readout. With crops
               gone the classes must come from SOMEWHERE — this is 8
               params of per-class mixing over three oriented edge-energy
               channels, not a conv trunk.

Trains with the standard 2-class center focal (D57) directly on the heat
logits — no more class-agnostic proxy — plus the smoothing quat's
dedicated term. detect() emits the family (heat, offset) contract, so
CenterObjectMetrics and the v13..v20 ladder stay apples-to-apples.

What this ablation measures: whether v21.1's epoch-0 signal (sel 0.098)
came from the conditioned filter bank + crop classifier at all, or
whether two colour quaternions and a Sobel were carrying everything.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import quat_mul, sobel7

GRID_H, GRID_W = 27, 48
STRIDE = 20
IMG_H, IMG_W = 540, 960
_LUMA = (0.299, 0.587, 0.114)


def _skew(w):
    z = torch.zeros((), device=w.device, dtype=w.dtype)
    return torch.stack([
        torch.stack([z, -w[2], w[1]]),
        torch.stack([w[2], z, -w[0]]),
        torch.stack([-w[1], w[0], z]),
    ])


class HyperDualQuaternionRGB(nn.Module):
    """v21.3 (owner): hyper-dual quaternion colour transform — the D5
    dual quat extended over the hyper-dual algebra q = q0 + e1*q1 +
    e2*q2 + e1e2*q3 (e1^2 = e2^2 = 0). Sandwiching a colour 3-vector
    gives an AFFINE map built from four generators:

        rotation R from q0; translation t1 from q1; an infinitesimal
        rotation W (skew) from q2 composed as (I + W)R — a controlled
        shear/scale departure from rigidity; and the cross term from q3
        as a second translation t2 (linearly redundant with t1 at init;
        its training curvature is not — documented, not hidden).

    Strictly contains DualQuaternionRGB (identity init: q0=[1,0,0,0],
    rest 0) and still folds to ONE 1x1 conv + bias at export, so D5
    deploy legality is unchanged. 16 params."""

    def __init__(self):
        super().__init__()
        self.q0 = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.q1 = nn.Parameter(torch.zeros(4))
        self.q2 = nn.Parameter(torch.zeros(4))
        self.q3 = nn.Parameter(torch.zeros(4))

    def matrix(self):
        q = self.q0 / self.q0.norm().clamp_min(1e-8)
        w_, x, y, z = q.unbind(-1)
        r = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w_ * z), 2 * (x * z + w_ * y)]),
            torch.stack([2 * (x * y + w_ * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w_ * x)]),
            torch.stack([2 * (x * z - w_ * y), 2 * (y * z + w_ * x), 1 - 2 * (x * x + y * y)]),
        ])
        conj = q * q.new_tensor([1.0, -1.0, -1.0, -1.0])
        t1 = 2.0 * quat_mul(self.q1, conj)[1:]
        w = 2.0 * quat_mul(self.q2, conj)[1:]
        t2 = 2.0 * quat_mul(self.q3, conj)[1:]
        wx = _skew(w)
        m = r + wx @ r
        t = t1 + wx @ t1 + t2
        return m, t

    def forward(self, img):  # (B,3,H,W)
        m, t = self.matrix()
        return F.conv2d(img, m.reshape(3, 3, 1, 1), t)


class V21TwoStage(nn.Module):
    """Name kept from v21/v21.1 so the trainer/viz plumbing is stable;
    since v21.2 there is no second stage — checkpoints tag arch v21.2."""

    def __init__(self, max_peaks=12):
        super().__init__()
        self.max_peaks = max_peaks
        self.quat_smooth = HyperDualQuaternionRGB()   # v21.3 swap
        self.quat_edge = HyperDualQuaternionRGB()
        e = torch.stack([sobel7("v"), sobel7("h"), sobel7("d1")])
        self.edge = nn.Parameter(e.reshape(3, 1, 7, 7).clone())
        # minimal class readout on pooled per-channel edge energy; init
        # mirrors the old saliency calibration (a=4 spread over 3 ch,
        # b=-2) so cold-start probs sit in the same range v21.1 trained
        # through
        self.head = nn.Conv2d(3, 2, 1)
        nn.init.constant_(self.head.weight, 4.0 / 3.0)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, img):  # (B,3,540,960) in [0,1]
        smooth = self.quat_smooth(img)
        edge = F.conv2d(self.quat_edge(smooth), self.edge,
                        padding=3, groups=3)
        energy = F.max_pool2d(edge.abs(), STRIDE)     # (B,3,27,48)
        return {"heat_logits": self.head(energy), "smooth": smooth,
                "edge": edge, "energy": energy}

    @torch.no_grad()
    def find_peaks(self, prob, thresh=0.3):
        """(B,H,W) prob map -> per-sample list of (row, col, p): 3x3
        local maxima above thresh, strongest-first, capped at max_peaks.
        Single host sync (the vectorized form — the loop version cost
        183 ms/step on MPS)."""
        m = F.max_pool2d(prob.unsqueeze(1), 3, 1, 1).squeeze(1)
        mask = (prob >= m) & (prob > thresh)
        idx = mask.nonzero()
        packed = torch.cat([idx.float(), prob[mask].unsqueeze(1)],
                           1).cpu().numpy()
        out = [[] for _ in range(prob.shape[0])]
        for j in (-packed[:, 3]).argsort(kind="stable"):
            b, r, c, v = packed[j]
            if len(out[int(b)]) < self.max_peaks:
                out[int(b)].append((int(r), int(c), float(v)))
        return out

    @torch.no_grad()
    def detect(self, img, peak_thresh=0.3):
        """Family contract: per-class sigmoid heat + offset (constant
        cell-center 0.5 — v21.2 has no offset head). CenterObjectMetrics
        does its own peak finding on these maps, same as v13..v20."""
        out = self.forward(img)
        heat = torch.sigmoid(out["heat_logits"]).cpu()
        offset = torch.full_like(heat, 0.5)
        return heat, offset


class EntropyProbe(nn.Module):
    """v21.3 side probe (owner: "just want to see if entropy is
    useful"). NOT part of the main model: the trainer feeds it a
    DETACHED image, its loss touches only its own 60 params, and the
    main heat never sees it.

    Pipeline: its own hyper-dual quat -> luminance -> vertical slices
    50 px wide (full height, owner's 50xN) -> differentiable soft-
    histogram entropy per slice (16 bins; rows subsampled 4x, cols 2x
    for memory) -> per-slice feature [H, dH_left, dH_right] ("sudden
    changes in entropy") -> linear -> slice logit. Target: does the
    slice contain a GT object centre. The per-epoch val AUC of that
    logit IS the answer to the owner's question — AUC ~0.5 means
    entropy carries nothing, AUC >> 0.5 means it's worth wiring in."""

    SLICE = 50
    BINS = 16

    def __init__(self):
        super().__init__()
        self.hdq = HyperDualQuaternionRGB()
        self.register_buffer("luma", torch.tensor(_LUMA).reshape(1, 3, 1, 1))
        self.register_buffer("centers", torch.linspace(0.0, 1.0, self.BINS))
        self.log_sigma = nn.Parameter(torch.tensor(-3.0))  # bin softness
        self.score = nn.Linear(3, 1)

    @property
    def n_slices(self):
        return IMG_W // self.SLICE  # 19 (last 10 px unused)

    def forward(self, img):  # img: (B,3,540,960), DETACHED by the caller
        lum = (self.hdq(img) * self.luma).sum(1)            # (B,H,W)
        lum = lum[:, ::4, : self.n_slices * self.SLICE : 2]  # subsample
        B, H, W = lum.shape
        s = lum.reshape(B, H, self.n_slices, self.SLICE // 2)
        s = s.permute(0, 2, 1, 3).reshape(B, self.n_slices, -1)
        sig = self.log_sigma.exp().clamp_min(1e-3)
        d = s.unsqueeze(-1) - self.centers                   # (B,S,N,BINS)
        p = torch.exp(-(d / sig) ** 2).sum(2)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
        ent = -(p * p.clamp_min(1e-8).log()).sum(-1)         # (B,S)
        dl = torch.zeros_like(ent)
        dr = torch.zeros_like(ent)
        dl[:, 1:] = (ent[:, 1:] - ent[:, :-1]).abs()
        dr[:, :-1] = (ent[:, :-1] - ent[:, 1:]).abs()
        feats = torch.stack([ent, dl, dr], -1)               # (B,S,3)
        return self.score(feats).squeeze(-1)                 # (B,S) logits

    def targets(self, boxes):
        """(B,256,5) padded normalized boxes -> (B,S) 1.0 where a GT
        object centre falls in the slice."""
        B = boxes.shape[0]
        t = torch.zeros(B, self.n_slices)
        for b in range(B):
            for cls, cx, cy, w, h in boxes[b]:
                if cls < 0:
                    continue
                s = int(cx * IMG_W / self.SLICE)
                if 0 <= s < self.n_slices:
                    t[b, s] = 1.0
        return t
