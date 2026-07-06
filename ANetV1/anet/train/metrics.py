"""Cell-level P/R/F1 + object-level recall/FP.

Worst-decile proxy (no per-image GSD meta in the dataset export): object
recall is additionally sliced by source (synthetic vs VisDrone) and by GT
box-area decile computed from the labels themselves.
"""

from collections import deque

import numpy as np

from ..data.rasterize import GRID_H, GRID_W, box_footprint_cells

CLASS_NAMES = ("background", "mannequin", "tent")


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
