"""Loader for datasets/suas-synth-50k (YOLO format, mixed sources).

Realities handled here (see dataset README):
  - synthetic frames are 1920x1080; VisDrone (`vd_*`) frames vary in size
    -> everything is letterboxed to 960x540 with boxes remapped
  - VisDrone person boxes dominate mannequin counts -> per-source sampler weight
  - teacher soft grids (distillation) load from a cache dir keyed by stem
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .rasterize import (
    CANVAS_H,
    CANVAS_W,
    boxes_to_grid,
    letterbox_params,
    transform_boxes,
)

PAD_VALUE = 114  # YOLO-convention gray
MAX_BOXES = 256


def _read_label(path):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                boxes.append([float(v) for v in parts[:5]])
    return np.asarray(boxes, np.float32).reshape(-1, 5)


class SUASCells(Dataset):
    def __init__(self, root, split, coverage_thresh=0.3, teacher_dir=None,
                 vd_weight=0.4, mannequin_weight=4.0, tent_weight=2.0):
        self.root = Path(root)
        self.split = split
        self.coverage_thresh = coverage_thresh
        self.teacher_dir = Path(teacher_dir) if teacher_dir else None
        img_dir = self.root / "images" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(img_dir)
        self.items = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.label_dir = self.root / "labels" / split

        # one pass over label files for sampler weights + eval metadata
        self._weights = np.empty(len(self.items), np.float32)
        self._has = np.zeros((len(self.items), 2), bool)
        for i, p in enumerate(self.items):
            boxes = _read_label(self.label_dir / f"{p.stem}.txt")
            has_m = bool((boxes[:, 0] == 0).any()) if len(boxes) else False
            has_t = bool((boxes[:, 0] == 1).any()) if len(boxes) else False
            self._has[i] = (has_m, has_t)
            w = 1.0 + mannequin_weight * has_m + tent_weight * has_t
            if p.stem.startswith("vd_"):
                w *= vd_weight
            self._weights[i] = w

    def __len__(self):
        return len(self.items)

    def sample_weights(self):
        return torch.from_numpy(self._weights)

    def is_visdrone(self, i):
        return self.items[i].stem.startswith("vd_")

    def __getitem__(self, i):
        path = self.items[i]
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        s, nw, nh, px, py = letterbox_params(w0, h0)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (PAD_VALUE,) * 3)
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        tensor = torch.from_numpy(
            np.asarray(canvas, np.float32).transpose(2, 0, 1) / 255.0
        )

        boxes = transform_boxes(_read_label(self.label_dir / f"{path.stem}.txt"), w0, h0)
        grid = torch.from_numpy(boxes_to_grid(boxes, self.coverage_thresh))

        padded = torch.full((MAX_BOXES, 5), -1.0)
        n = min(len(boxes), MAX_BOXES)
        if n:
            padded[:n] = torch.from_numpy(boxes[:n])

        out = {"image": tensor, "grid": grid, "boxes": padded,
               "vd": self.is_visdrone(i), "stem": path.stem}
        if self.teacher_dir is not None:
            soft = np.load(self.teacher_dir / f"{path.stem}.npz")["grid"]
            out["teacher"] = torch.from_numpy(soft)
        return out
