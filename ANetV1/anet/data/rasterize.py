"""YOLO boxes -> 54x96 cell-label grids on the 960x540 letterboxed canvas.

Coverage uses overlap / min(box_area, cell_area) so sub-cell boxes (VisDrone
persons at 540p are often < one 10x10 cell) still reach full coverage; every
box additionally marks its center cell unconditionally.
"""

import numpy as np

CANVAS_W, CANVAS_H = 960, 540
GRID_W, GRID_H = 96, 54
N_CLASSES = 2  # mannequin, tent (background handled as class 0 in grids)


def letterbox_params(w0, h0, tw=CANVAS_W, th=CANVAS_H):
    """Scale-to-fit + center pad. Synthetic 1920x1080 -> scale 0.5, no pad."""
    s = min(tw / w0, th / h0)
    nw, nh = round(w0 * s), round(h0 * s)
    px, py = (tw - nw) // 2, (th - nh) // 2
    return s, nw, nh, px, py


def transform_boxes(boxes, w0, h0):
    """(N,5) [cls, cx, cy, w, h] normalized in the ORIGINAL image ->
    normalized on the letterboxed canvas."""
    if len(boxes) == 0:
        return np.zeros((0, 5), np.float32)
    b = np.asarray(boxes, np.float32).copy()
    _, nw, nh, px, py = letterbox_params(w0, h0)
    b[:, 1] = (b[:, 1] * nw + px) / CANVAS_W
    b[:, 3] = b[:, 3] * nw / CANVAS_W
    b[:, 2] = (b[:, 2] * nh + py) / CANVAS_H
    b[:, 4] = b[:, 4] * nh / CANVAS_H
    return b


def _box_cell_coverage(box):
    """Yield (row, col, coverage) for one canvas-normalized box."""
    _, cx, cy, w, h = box
    if w <= 0 or h <= 0:
        return
    x0, x1 = (cx - w / 2) * GRID_W, (cx + w / 2) * GRID_W
    y0, y1 = (cy - h / 2) * GRID_H, (cy + h / 2) * GRID_H
    box_area = (x1 - x0) * (y1 - y0)  # in cell units; cell area == 1
    c0, c1 = max(int(np.floor(x0)), 0), min(int(np.ceil(x1)), GRID_W)
    r0, r1 = max(int(np.floor(y0)), 0), min(int(np.ceil(y1)), GRID_H)
    denom = max(min(box_area, 1.0), 1e-9)
    for r in range(r0, r1):
        oy = min(y1, r + 1) - max(y0, r)
        if oy <= 0:
            continue
        for c in range(c0, c1):
            ox = min(x1, c + 1) - max(x0, c)
            if ox > 0:
                yield r, c, min(ox * oy / denom, 1.0)


def center_cell(box):
    _, cx, cy, _, _ = box
    r = min(max(int(cy * GRID_H), 0), GRID_H - 1)
    c = min(max(int(cx * GRID_W), 0), GRID_W - 1)
    return r, c


def boxes_to_grid(boxes, coverage_thresh=0.3):
    """Hard labels: (54, 96) int64; 0=background, 1=mannequin, 2=tent."""
    cov = np.zeros((N_CLASSES, GRID_H, GRID_W), np.float32)
    for box in boxes:
        k = int(box[0])
        for r, c, f in _box_cell_coverage(box):
            cov[k, r, c] = max(cov[k, r, c], f)
        r, c = center_cell(box)
        cov[k, r, c] = 1.0  # a box always marks at least its center cell
    grid = np.zeros((GRID_H, GRID_W), np.int64)
    best = cov.max(0)
    cls = cov.argmax(0)
    hit = best >= coverage_thresh
    grid[hit] = cls[hit] + 1
    return grid


def boxes_to_soft_grid(boxes_conf, coverage_thresh=0.0):
    """Teacher soft labels: boxes_conf rows [cls, cx, cy, w, h, conf] ->
    (3, 54, 96) probabilities."""
    p = np.zeros((1 + N_CLASSES, GRID_H, GRID_W), np.float32)
    for row in boxes_conf:
        k, conf = int(row[0]), float(row[5])
        for r, c, f in _box_cell_coverage(row[:5]):
            if f >= coverage_thresh:
                p[k + 1, r, c] = max(p[k + 1, r, c], conf * f)
        r, c = center_cell(row[:5])
        p[k + 1, r, c] = max(p[k + 1, r, c], conf)
    fg = p[1:].sum(0)
    scale = np.where(fg > 1.0, 1.0 / np.maximum(fg, 1e-9), 1.0)
    p[1:] *= scale
    p[0] = 1.0 - p[1:].sum(0)
    return p


def box_footprint_cells(box, coverage_thresh=0.05):
    """Cells a GT box occupies (for object-level metrics)."""
    cells = [(r, c) for r, c, f in _box_cell_coverage(box) if f >= coverage_thresh]
    if not cells:
        cells = [center_cell(box)]
    return cells
