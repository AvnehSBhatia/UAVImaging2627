"""v21.5 (D71 line, owner-directed): chunk detector — no crops at all.

The 5.6k dual-scale CropCNN is gone (owner: too slow — full-res crops
were the measured cost). Stage 2 is now entirely cell-resolution:

  learned threshold (mean-RGB -> 3->8-SiLU->4->1 -> sigmoid, cold 0.3)
  -> cells above threshold, 8-connected into CHUNKS
  -> per-chunk features: soft-membership-pooled cell features (avg RGB,
     avg smooth, max |edge|), saliency max/mean, chunk SIZE + spans
     (the tent feature the family lacked: tents span ~5x5 cells,
     mannequins 1-2), the threshold itself
  -> ~340-param CLS head -> {BG, mannequin, tent} at the chunk's
     saliency-weighted centroid.

The hard mask only SELECTS; gradients flow through the soft membership
weights sigmoid((p - thresh)/tau) into the threshold MLP and the trunk.

--- prior rev (v21.4) ---

History of this file, one experiment per rev (ARCHITECTURE.md 16.9):
v21/v21.1 = A^chan 11x11 filter bank -> quats -> saliency -> 100x100
crops (9ch + ctx). v21.2 = ablate everything but the quats+Sobel.
v21.3 = hyper-dual quats + the EntropyProbe. v21.4 = the owner's
synthesis of what the measurements said:

  - the 11x11 bank is REMOVED for good: the eval-time bypass on the
    trained v21.1 measured it as an ATTENUATOR (blur-init kernels ahead
    of the Sobel; bypassing them took mann 0.072 -> 0.185)
  - the crop stage is BACK: the same bypass showed the classifier
    turning sharper saliency into 2.6x recall, and v21.3's cropless
    energy readout was structurally tent-blind
  - crops are now DUAL-SCALE (50 tight + 100 wide, shared trunk): the
    tent fix — a tent's edges ring its perimeter (wide view), a
    mannequin fills the tight view
  - quats stay hyper-dual (strict superset of D5, folds to 1x1 conv)
  - EntropyProbe continues on the side (isolated; ent_auc is its
    entire output)

Deploy notes unchanged: crop gather is a CPU stage; everything else
folds. Research track.
"""

import math

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


def label_chunks(mask):
    """8-connected component labelling of a (27,48) bool numpy mask ->
    list of cell lists [(r,c), ...]. Grid is tiny; plain DFS."""
    import numpy as np
    lbl = -1 * np.ones(mask.shape, dtype=int)
    chunks = []
    for r, c in zip(*np.nonzero(mask)):
        if lbl[r, c] >= 0:
            continue
        k = len(chunks)
        stack, cells = [(int(r), int(c))], []
        lbl[r, c] = k
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                            and mask[yy, xx] and lbl[yy, xx] < 0):
                        lbl[yy, xx] = k
                        stack.append((yy, xx))
        chunks.append(cells)
    return chunks


class V21TwoStage(nn.Module):
    """v21.4: the prev-best two-stage (v21.1) with the 11x11 filter
    bank REMOVED (owner: "we borked the model... remove the 11x11 conv"
    — and the eval-time bypass measured why: the blur-init bank
    attenuated small-object edges before the Sobel; bypassing it took
    the trained v21.1 from mann 0.072 to 0.185). Raw image feeds the
    smoothing quat directly; quats stay hyper-dual (v21.3, strict
    superset of D5). Stage 2 is the crop pipeline back from v21.1, now
    dual-scale (50 + 100) per the owner's tent fix. line_means survives
    only as the crop context's scene statistic."""

    MAX_CHUNKS = 24  # per image, strongest-first
    N_FEAT = 14      # per-chunk feature vector width
    THRESH = 0.8     # fixed chunk threshold (owner call: the learned
    #                  MLP kept the cold mask saturated — at 0.8 only
    #                  genuinely hot cells chunk, from step 0)

    def __init__(self, max_peaks=12, n_lines=20):
        super().__init__()
        self.max_peaks = max_peaks
        self.n_lines = n_lines
        self.quat_smooth = HyperDualQuaternionRGB()
        self.quat_edge = HyperDualQuaternionRGB()
        e = torch.stack([sobel7("v"), sobel7("h"), sobel7("d1")])
        self.edge = nn.Parameter(e.reshape(3, 1, 7, 7).clone())
        self.sal_a = nn.Parameter(torch.tensor(4.0))
        self.sal_b = nn.Parameter(torch.tensor(-2.0))
        # v21.5 chunk CLS head (~330 params, replaces the 5.6k CropCNN):
        # per-chunk features are cell-resolution ONLY — no full-res crops
        self.cls_head = nn.Sequential(
            nn.Linear(self.N_FEAT, 16), nn.SiLU(), nn.Linear(16, 3))

    def line_means(self, img):
        """20 random rows + 20 random cols -> per-channel mean (B,3);
        strided (deterministic) at eval. Crop-context only in v21.4."""
        B, _, H, W = img.shape
        if self.training:
            rows = torch.randint(0, H, (self.n_lines,), device=img.device)
            cols = torch.randint(0, W, (self.n_lines,), device=img.device)
        else:
            rows = torch.linspace(0, H - 1, self.n_lines,
                                  device=img.device).long()
            cols = torch.linspace(0, W - 1, self.n_lines,
                                  device=img.device).long()
        r = img[:, :, rows, :].mean((2, 3))
        c = img[:, :, :, cols].mean((2, 3))
        return (r + c) / 2.0

    def forward(self, img):  # (B,3,540,960) in [0,1]
        smooth = self.quat_smooth(img)
        edge = F.conv2d(self.quat_edge(smooth), self.edge,
                        padding=3, groups=3)
        sal = edge.norm(dim=1, keepdim=True)
        pooled = F.max_pool2d(sal, STRIDE).squeeze(1)   # (B,27,48)
        means = self.line_means(img)
        # cell-resolution feature grid for chunk pooling: avg raw RGB,
        # avg smoothed, max |edge| — everything the classifier sees
        cell_feats = torch.cat([
            F.avg_pool2d(img, STRIDE),
            F.avg_pool2d(smooth, STRIDE),
            F.max_pool2d(edge.abs(), STRIDE)], 1)       # (B,9,27,48)
        return {"sal_logits": self.sal_a * pooled + self.sal_b,
                "smooth": smooth, "edge": edge, "means": means,
                "cell_feats": cell_feats,
                "thresh": torch.full((img.shape[0],), self.THRESH,
                                     device=img.device)}

    def image_chunks(self, prob_b, thresh_b):
        """Hard selection (no grad needed): cells above the learned
        threshold, 8-connected into chunks, strongest-first, capped."""
        mask = (prob_b > thresh_b).detach().cpu().numpy()
        chunks = label_chunks(mask)
        if len(chunks) > self.MAX_CHUNKS:
            peak = [max(float(prob_b[r, c]) for r, c in cells)
                    for cells in chunks]
            order = sorted(range(len(chunks)), key=lambda i: -peak[i])
            chunks = [chunks[i] for i in order[: self.MAX_CHUNKS]]
        return chunks

    def chunk_logits(self, b, chunks, prob, out, tau=0.1):
        """Per-chunk feature vector -> CLS logits (K,3). Soft membership
        w = sigmoid((p - thresh)/tau) carries gradient into the
        threshold MLP and (through p) the whole trunk."""
        feats = []
        thresh = out["thresh"][b]
        cf = out["cell_feats"][b]                        # (9,27,48)
        for cells in chunks:
            idx = torch.tensor(cells, device=prob.device)
            rs, cs_ = idx[:, 0], idx[:, 1]
            p = prob[b, rs, cs_]
            w = torch.sigmoid((p - thresh) / tau)
            pooled = (cf[:, rs, cs_] * w).sum(1) / w.sum().clamp_min(1e-6)
            span_w = float(cs_.max() - cs_.min() + 1) / GRID_W
            span_h = float(rs.max() - rs.min() + 1) / GRID_H
            f = torch.cat([
                pooled, p.max().reshape(1), p.mean().reshape(1),
                p.new_tensor([min(len(cells), 200) / 200.0,
                              span_w, span_h])])
            feats.append(f)
        return self.cls_head(torch.stack(feats))

    @staticmethod
    def chunk_centroid(cells, prob_b):
        """Saliency-weighted centroid cell of a chunk."""
        idx = torch.tensor(cells, device=prob_b.device)
        p = prob_b[idx[:, 0], idx[:, 1]].clamp_min(1e-6)
        r = int(round(float((idx[:, 0].float() * p).sum() / p.sum())))
        c = int(round(float((idx[:, 1].float() * p).sum() / p.sum())))
        return min(max(r, 0), GRID_H - 1), min(max(c, 0), GRID_W - 1)

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
        """Full inference: learned threshold -> 8-connected chunks ->
        chunk CLS head. Each non-BG chunk writes its class prob at its
        saliency-weighted centroid cell, so CenterObjectMetrics and the
        ladder stay apples-to-apples. peak_thresh is accepted for
        interface compatibility; v21.5's threshold is LEARNED."""
        out = self.forward(img)
        prob = torch.sigmoid(out["sal_logits"])
        B = img.shape[0]
        heat = torch.zeros(B, 2, GRID_H, GRID_W)
        offset = torch.full((B, 2, GRID_H, GRID_W), 0.5)
        for b in range(B):
            chunks = self.image_chunks(prob[b], out["thresh"][b])
            if not chunks:
                continue
            cls_p = torch.softmax(
                self.chunk_logits(b, chunks, prob, out), 1)
            for cells, p in zip(chunks, cls_p):
                if int(p.argmax()) > 0:
                    r, c = self.chunk_centroid(cells, prob[b])
                    k = int(p.argmax()) - 1
                    heat[b, k, r, c] = max(float(heat[b, k, r, c]),
                                           float(p.max()))
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
