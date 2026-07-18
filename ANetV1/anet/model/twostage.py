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

from .blocks import DualQuaternionRGB, sobel7

GRID_H, GRID_W = 27, 48
STRIDE = 20
IMG_H, IMG_W = 540, 960


class V21TwoStage(nn.Module):
    """Name kept from v21/v21.1 so the trainer/viz plumbing is stable;
    since v21.2 there is no second stage — checkpoints tag arch v21.2."""

    def __init__(self, max_peaks=12):
        super().__init__()
        self.max_peaks = max_peaks
        self.quat_smooth = DualQuaternionRGB()
        self.quat_edge = DualQuaternionRGB()
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
