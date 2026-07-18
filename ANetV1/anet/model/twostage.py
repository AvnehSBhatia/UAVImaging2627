"""v21.4 (D71 line, owner-directed): raw-front two-stage, dual-scale crops.

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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import quat_mul, sobel7

GRID_H, GRID_W = 27, 48
STRIDE = 20
IMG_H, IMG_W = 540, 960
CROP_S, CROP_L = 50, 100  # v21.4 dual-scale crops (owner: "50x50 and
#                           then a 100x100" — the tent fix)
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


class CropCNN(nn.Module):
    """v21.4 dual-scale crop classifier: ONE shared conv trunk (GAP
    makes it size-agnostic) applied to the 50x50 tight view AND the
    100x100 wide view of each peak, embeddings concatenated with the
    4-scalar context into the class head.

    Why two scales fixes tents (measured, v21.3 viz): a ~100 px tent's
    CENTER cell is smooth fabric — its edge evidence rings the
    perimeter, which only the wide view contains; a ~25 px mannequin
    drowns in a 100x100 crop but fills the tight view. GroupNorm: crop
    batches are small and variable."""

    def __init__(self, in_ch=9, n_ctx=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, 2, 1), nn.GroupNorm(4, 8), nn.SiLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(24 * 2 + n_ctx, 3)

    def forward(self, crop_s, crop_l, ctx):
        return self.fc(torch.cat([self.net(crop_s), self.net(crop_l),
                                  ctx], 1))


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
        self.crop_cnn = CropCNN()

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
        return {"sal_logits": self.sal_a * pooled + self.sal_b,
                "smooth": smooth, "edge": edge,
                "means": self.line_means(img)}

    @staticmethod
    def stack_maps(img, out):
        """The 9-channel crop source: raw RGB + smooth + edge."""
        return torch.cat([img, out["smooth"], out["edge"]], 1)

    @staticmethod
    def crop_at(maps, cx_px, cy_px, size):
        """size x size crop around a full-res center, clamped inside
        the frame."""
        ox = int(min(max(round(cx_px - size / 2), 0), IMG_W - size))
        oy = int(min(max(round(cy_px - size / 2), 0), IMG_H - size))
        return maps[:, oy:oy + size, ox:ox + size]

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
        """Full inference: saliency peaks -> dual-scale crops ->
        CropCNN. Emits the family (heat, offset) tensors — each
        detection writes its class prob at its peak cell — so
        CenterObjectMetrics and the ladder stay apples-to-apples."""
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
            cs, cl, ctx = [], [], []
            for r, c, v in peaks[b]:
                cx, cy = (c + 0.5) * STRIDE, (r + 0.5) * STRIDE
                cs.append(self.crop_at(stacked[b], cx, cy, CROP_S))
                cl.append(self.crop_at(stacked[b], cx, cy, CROP_L))
                ctx.append(torch.cat([
                    torch.tensor([v], device=img.device), out["means"][b]]))
            cls_p = torch.softmax(self.crop_cnn(
                torch.stack(cs), torch.stack(cl), torch.stack(ctx)), 1)
            for (r, c, _), p in zip(peaks[b], cls_p):
                if int(p.argmax()) > 0:
                    heat[b, int(p.argmax()) - 1, r, c] = float(p.max())
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
