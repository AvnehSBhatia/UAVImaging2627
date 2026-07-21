"""Cell-level P/R/F1 + object-level recall/FP.

Worst-decile proxy (no per-image GSD meta in the dataset export): object
recall is additionally sliced by source (synthetic vs VisDrone) and by GT
box-area decile computed from the labels themselves.
"""

from collections import deque

import numpy as np
import torch

from ..data.rasterize import GRID_H, GRID_W, box_footprint_cells

CLASS_NAMES = ("background", "mannequin", "tent")


def _decile_keys(recs):
    """Worst-GT-area-decile mannequin recall, POOLED and SYNTHETIC-ONLY (D82).

    The pooled key is retained so historical numbers stay comparable, but it
    must not be read as a mission metric. Measured on the test split:

        mannequin boxes            27,562   of which 98.5% are VisDrone
        pooled smallest decile      2,756   of which  100% are VisDrone
        decile area cutoff           13.1 px^2   (a ~3.6x3.6 px blob)
        median GT area   synthetic 1365.0 px^2 | VisDrone 59.1 px^2

    VisDrone frames are oblique urban street scenes whose person boxes are
    ~23x smaller than the SUAS mission geometry (mannequins/tents at 150 ft
    AGL nadir), and both raw VisDrone person classes remap to mannequin, so
    they outnumber synthetic mannequins ~64:1. The pooled decile therefore
    selects 3-4 px blobs from the wrong task, and every mechanism from D59 to
    D81 was tuned against it while being DESIGNED from synthetic failures.

    `..._synthetic` slices the decile within the synthetic distribution
    (cutoff ~574 px^2, i.e. genuinely small mission objects) and is the key
    to read for flight decisions.

    D85/§20.5 adds a POWER caveat on top of the task caveat. The synthetic
    decile holds ~42 objects on the test split, so it cannot resolve the
    differences the family compares: D85 vs v13 differed by 3 objects
    (-0.071) with a bootstrap 95% CI of [-0.286, +0.143], straddling zero.
    That also dissolves §16.6's "three consecutive checkpoints at exactly
    0.571", which was read as ~8/14 structurally immovable objects and is
    actually small-n quantization.

    So two companions ship alongside: `..._n` (never quote a decile without
    it) and `..._smallest_quartile_synthetic`, which has ~2.5x the sample and
    is the key to compare revisions on. The decile stays as the headline
    because it is what the historical record uses.
    """
    out = {}
    for suffix, keep in (("", lambda r: True), ("_synthetic", lambda r: not r[3])):
        m = sorted((r for r in recs if r[0] == 1 and keep(r)), key=lambda r: r[1])
        d = m[: max(len(m) // 10, 1)]
        out[f"mannequin_recall_smallest_decile{suffix}"] = (
            sum(r[2] for r in d) / len(d) if d else float("nan")
        )
        out[f"mannequin_recall_smallest_decile{suffix}_n"] = len(d)
    syn = sorted((r for r in recs if r[0] == 1 and not r[3]), key=lambda r: r[1])
    q = syn[: max(len(syn) // 4, 1)]
    out["mannequin_recall_smallest_quartile_synthetic"] = (
        sum(r[2] for r in q) / len(q) if q else float("nan")
    )
    out["mannequin_recall_smallest_quartile_synthetic_n"] = len(q)
    return out


def confident_pred(logits, thresh=0.0):
    """argmax, but demote a foreground win to background if its softmax prob
    doesn't clear `thresh` — the eval/deploy false-positive gate. logits (B,3,H,W).
    thresh<=0 is plain argmax."""
    probs = torch.softmax(logits, 1)
    pred = probs.argmax(1)
    if thresh > 0:
        top = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
        pred = torch.where((pred > 0) & (top < thresh), torch.zeros_like(pred), pred)
    return pred


class CellConfusion:
    def __init__(self):
        self.mat = np.zeros((3, 3), np.int64)

    def update(self, pred, target):  # (B,54,96) int arrays
        idx = target.reshape(-1) * 3 + pred.reshape(-1)
        self.mat += np.bincount(idx, minlength=9).reshape(3, 3)

    def summary(self):
        out = {}
        for k in (1, 2):
            tp = self.mat[k, k]
            fp = self.mat[:, k].sum() - tp
            fn = self.mat[k, :].sum() - tp
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            out[CLASS_NAMES[k]] = {
                "precision": p, "recall": r,
                "f1": 2 * p * r / max(p + r, 1e-9),
                # raw counts: distinguishes "class never predicted anywhere"
                # (pred_cells==0, true collapse) from "predicted in wrong places"
                "pred_cells": int(self.mat[:, k].sum()),
                "gt_cells": int(self.mat[k, :].sum()),
            }
        return out


class ObjectMetrics:
    def __init__(self):
        self.records = []  # (cls, area, found, is_vd)
        self.fp_components = 0
        self.images = 0

    def update(self, pred_grid, boxes, is_vd):
        """pred_grid (54,96) int; boxes (N,5) canvas-normalized, -1 padded."""
        self.images += 1
        gt_cells = set()
        for box in boxes:
            if box[0] < 0:
                continue
            cls = int(box[0]) + 1
            cells = box_footprint_cells(box)
            gt_cells.update(cells)
            found = any(pred_grid[r, c] == cls for r, c in cells)
            self.records.append((cls, float(box[3] * box[4]), found, bool(is_vd)))
        self.fp_components += self._count_fp(pred_grid, gt_cells)

    @staticmethod
    def _count_fp(pred_grid, gt_cells):
        seen = np.zeros_like(pred_grid, bool)
        n_fp = 0
        for r in range(GRID_H):
            for c in range(GRID_W):
                if pred_grid[r, c] == 0 or seen[r, c]:
                    continue
                comp, hits_gt = [], False
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    if (cr, cc) in gt_cells:
                        hits_gt = True
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < GRID_H and 0 <= nc < GRID_W
                                and not seen[nr, nc] and pred_grid[nr, nc] != 0):
                            seen[nr, nc] = True
                            q.append((nr, nc))
                n_fp += not hits_gt
        return n_fp

    def summary(self):
        out = {"fp_per_image": self.fp_components / max(self.images, 1)}
        recs = self.records
        for k, name in ((1, "mannequin"), (2, "tent")):
            sub = [r for r in recs if r[0] == k]
            out[f"{name}_recall"] = (
                sum(r[2] for r in sub) / len(sub) if sub else float("nan")
            )
        # slices: source and smallest-decile GT boxes (worst-decile proxy)
        for name, cond in (("synthetic", lambda r: not r[3]), ("visdrone", lambda r: r[3])):
            sub = [r for r in recs if r[0] == 1 and cond(r)]
            out[f"mannequin_recall_{name}"] = (
                sum(r[2] for r in sub) / len(sub) if sub else float("nan")
            )
        out.update(_decile_keys(recs))
        return out


# v12 center-heatmap grid — stride-20 single-phase, distinct from the v8/v9
# 54x96 overlap grid above (GRID_H/GRID_W). Kept local to avoid conflating
# the two conventions.
V12_H, V12_W = 27, 48


class CenterObjectMetrics:
    """Object-level recall/FP for the v12 center-heatmap head. Peaks are
    3x3 local maxima of the per-class sigmoid heatmap above `peak_thresh`;
    each surviving peak's (row,col) is de-quantized by the class-agnostic
    offset head to a canvas-normalized (cx,cy) and matched against GT boxes
    of the same class by point-in-box containment. summary() mirrors
    ObjectMetrics.summary()'s key names so trainer best.pt selection
    (`mannequin_synth + 0.5*tent`) is unchanged whichever head is active."""

    def __init__(self, peak_thresh=0.3):
        self.peak_thresh = peak_thresh
        self.records = []  # (cls, area, found, is_vd) — cls: 1=mannequin, 2=tent
        self.fp_components = 0
        self.images = 0
        # MARGIN (v23/D76): the diagnostic the viz_web_scenes cases exposed
        # and that recall/fp structurally hide. Per image and class:
        #   margin = p(heat at the GT object's own center cell)
        #          - max p(heat) over cells far from every GT of that class
        # Measured on the trained family: a clear spread-eagle person scored
        # 0.10 while empty-corner background scored 0.36 — i.e. margin was
        # NEGATIVE (~-0.23) on the EASIEST case. Read at the GT centre rather
        # than at a matched peak on purpose: it stays defined when the object
        # is missed entirely, which is exactly the interesting case.
        self._margins = {1: [], 2: []}
        # D87: fp sliced by source. `fp_per_image` pools all frames, and the
        # test split is 1,267 VisDrone vs 449 synthetic — so the number the
        # family has always quoted is ~74% VisDrone-driven, the same defect
        # D82 found in recall. Mission fp is the synthetic-only figure.
        self.fp_synth = 0
        self.images_synth = 0

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _find_peaks(prob, thresh):
        """(H,W) prob -> (rows, cols) of cells that are >thresh and >= every
        neighbor in their 3x3 window (self included, so plateaus/ties all
        fire — matches CenterNet-style NMS-free peak picking)."""
        h, w = prob.shape
        padded = np.full((h + 2, w + 2), -np.inf, dtype=prob.dtype)
        padded[1:-1, 1:-1] = prob
        local_max = np.full_like(prob, -np.inf)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                local_max = np.maximum(local_max, padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w])
        mask = (prob > thresh) & (prob >= local_max)
        return np.where(mask)

    def update(self, heat_prob, offset_prob, boxes, is_vd, cls_ids=(0, 1),
               count_image=True):
        """heat_prob/offset_prob (C,H,W) sigmoid probs; boxes (N,5)
        canvas-normalized [cls,cx,cy,w,h], -1 padded. Grid dims are derived
        from the tensors (27x48 for the v12-v21 stride-20 family; 54x96 for a
        stride-10 readout) — peak matching is point-in-box containment on the
        canvas, so recall/fp stay apples-to-apples across grid resolutions.

        cls_ids maps heat channel i -> box class cls_ids[i]. The default
        (0,1) is the single-tensor two-class contract every arch up to v22
        uses. v23 (D76) reads its classes on DIFFERENT grids, so it calls
        this once per class — update(mann_heat(1,54,96), ..., cls_ids=(0,))
        then update(tent_heat(1,27,48), ..., cls_ids=(1,), count_image=False)
        — both appending into the same records/fp counters, so summary()'s
        keys stay identical and the whole v13-v22 ladder stays comparable.
        count_image=False on the second call keeps fp/img per-frame, not
        per-(frame,class)."""
        heat_prob = self._to_numpy(heat_prob)
        offset_prob = self._to_numpy(offset_prob)
        grid_h, grid_w = heat_prob.shape[-2:]
        if count_image:
            self.images += 1
            if not is_vd:
                self.images_synth += 1
        for i, c in enumerate(cls_ids):
            rows, cols = self._find_peaks(heat_prob[i], self.peak_thresh)
            dx = offset_prob[0, rows, cols]
            dy = offset_prob[1, rows, cols]
            cx = (cols + dx) / grid_w
            cy = (rows + dy) / grid_h
            matched = np.zeros(len(rows), dtype=bool)
            gt_mask = np.zeros((grid_h, grid_w), bool)  # for the margin below
            true_scores = []
            for box in boxes:
                if box[0] < 0 or int(box[0]) != c:
                    continue
                bx, by, bw, bh = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                inside = ((cx >= bx - bw / 2) & (cx <= bx + bw / 2)
                          & (cy >= by - bh / 2) & (cy <= by + bh / 2))
                found = bool(inside.any())
                matched |= inside
                self.records.append((c + 1, bw * bh, found, bool(is_vd)))
                # margin bookkeeping: this object's own centre cell, and a
                # generous exclusion zone so "background" never includes the
                # object's own splat
                gr = min(max(int(by * grid_h), 0), grid_h - 1)
                gc = min(max(int(bx * grid_w), 0), grid_w - 1)
                true_scores.append(float(heat_prob[i, gr, gc]))
                r0, r1 = max(gr - 3, 0), min(gr + 4, grid_h)
                c0, c1 = max(gc - 3, 0), min(gc + 4, grid_w)
                gt_mask[r0:r1, c0:c1] = True
            n_fp = int((~matched).sum())
            self.fp_components += n_fp
            if not is_vd:
                self.fp_synth += n_fp
            if true_scores and not gt_mask.all():
                bg_max = float(heat_prob[i][~gt_mask].max())
                self._margins[c + 1].append(
                    float(np.mean(true_scores)) - bg_max)

    def summary(self):
        out = {"fp_per_image": self.fp_components / max(self.images, 1),
               "fp_per_image_synthetic": self.fp_synth / max(self.images_synth, 1)}
        recs = self.records
        for k, name in ((1, "mannequin"), (2, "tent")):
            sub = [r for r in recs if r[0] == k]
            out[f"{name}_recall"] = (
                sum(r[2] for r in sub) / len(sub) if sub else float("nan")
            )
        for name, cond in (("synthetic", lambda r: not r[3]), ("visdrone", lambda r: r[3])):
            sub = [r for r in recs if r[0] == 1 and cond(r)]
            out[f"mannequin_recall_{name}"] = (
                sum(r[2] for r in sub) / len(sub) if sub else float("nan")
            )
        out.update(_decile_keys(recs))
        # v23/D76: the margin the cases exposed (see __init__). Negative =
        # background outscores the objects themselves — the measured v13/v22
        # disease, invisible to every other key in this dict.
        for k, name in ((1, "mannequin"), (2, "tent")):
            v = self._margins[k]
            out[f"{name}_margin"] = float(np.mean(v)) if v else float("nan")
        return out
