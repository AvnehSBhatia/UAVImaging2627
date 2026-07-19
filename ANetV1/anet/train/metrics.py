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
        m = sorted((r for r in recs if r[0] == 1), key=lambda r: r[1])
        decile = m[: max(len(m) // 10, 1)]
        out["mannequin_recall_smallest_decile"] = (
            sum(r[2] for r in decile) / len(decile) if decile else float("nan")
        )
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

    def update(self, heat_prob, offset_prob, boxes, is_vd):
        """heat_prob/offset_prob (2,H,W) sigmoid probs; boxes (N,5)
        canvas-normalized [cls,cx,cy,w,h], -1 padded. Grid dims are derived
        from the tensors (27x48 for the v12-v21 stride-20 family; 54x96 for a
        stride-10 readout) — peak matching is point-in-box containment on the
        canvas, so recall/fp stay apples-to-apples across grid resolutions."""
        heat_prob = self._to_numpy(heat_prob)
        offset_prob = self._to_numpy(offset_prob)
        grid_h, grid_w = heat_prob.shape[-2:]
        self.images += 1
        for c in range(2):  # 0=mannequin, 1=tent
            rows, cols = self._find_peaks(heat_prob[c], self.peak_thresh)
            dx = offset_prob[0, rows, cols]
            dy = offset_prob[1, rows, cols]
            cx = (cols + dx) / grid_w
            cy = (rows + dy) / grid_h
            matched = np.zeros(len(rows), dtype=bool)
            for box in boxes:
                if box[0] < 0 or int(box[0]) != c:
                    continue
                bx, by, bw, bh = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                inside = ((cx >= bx - bw / 2) & (cx <= bx + bw / 2)
                          & (cy >= by - bh / 2) & (cy <= by + bh / 2))
                found = bool(inside.any())
                matched |= inside
                self.records.append((c + 1, bw * bh, found, bool(is_vd)))
            self.fp_components += int((~matched).sum())

    def summary(self):
        out = {"fp_per_image": self.fp_components / max(self.images, 1)}
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
        m = sorted((r for r in recs if r[0] == 1), key=lambda r: r[1])
        decile = m[: max(len(m) // 10, 1)]
        out["mannequin_recall_smallest_decile"] = (
            sum(r[2] for r in decile) / len(decile) if decile else float("nan")
        )
        return out
