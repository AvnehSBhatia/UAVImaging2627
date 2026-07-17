"""Object-patch crops for the P1/P2 probes (results: OBSERVATIONS.md).

Each item is ONE image decoded once, returning every object-centred crop
(mannequin -> 40x40, tent -> 100x100, owner spec) plus deterministic
background crops. Two targets ride along:
  - mask: union of GT box rectangles inside the crop window (white = box,
    black = background) — P1's target
  - label: the centre object's class (0 bg, 1 mannequin, 2 tent) — P2's

Synthetic-only by default: the probes ask whether colour algebra / a fixed
filter bank separates the RENDERED assets from OAM backgrounds; VisDrone's
native-resolution people are a different question (include_vd opts in).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.dataset import SUASCells

CANVAS_W, CANVAS_H = 960, 540
CROP = {0: 40, 1: 100}  # class id -> crop size in px (owner spec)


def _rects(boxes):
    """(N,5) normalized [cls,cx,cy,w,h] -> pixel rects (cls,x0,y0,x1,y1)."""
    out = []
    for c, cx, cy, w, h in boxes:
        if c < 0:
            continue
        out.append((int(c), (cx - w / 2) * CANVAS_W, (cy - h / 2) * CANVAS_H,
                    (cx + w / 2) * CANVAS_W, (cy + h / 2) * CANVAS_H))
    return out


class PatchCrops(Dataset):
    def __init__(self, root, split, include_vd=False, n_bg=2, limit=0,
                 cache=False):
        self.base = SUASCells(root, split, cache=cache)
        self.n_bg = n_bg
        self.idx = [i for i in range(len(self.base))
                    if include_vd or not self.base.is_visdrone(i)]
        if limit:
            self.idx = self.idx[:limit]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        s = self.base[i]
        img = s["image"]  # (3,540,960) float [0,1]
        rects = _rects(s["boxes"].numpy())
        out = {40: [], 100: []}
        for c, x0, y0, x1, y1 in rects:
            size = CROP.get(c)
            if size is None:
                continue
            crop, mask, org = self._window(img, rects, (x0 + x1) / 2,
                                           (y0 + y1) / 2, size)
            out[size].append((crop, mask, c + 1, org))
        # background crops: deterministic per image index (resume-safe, same
        # crops every epoch/eval), rejection-sampled off every GT rect
        rng = np.random.default_rng(0xB6 + 9973 * i)
        for size in (40, 100):
            made = 0
            for _ in range(20 * self.n_bg):
                if made >= self.n_bg:
                    break
                ox = int(rng.integers(0, CANVAS_W - size + 1))
                oy = int(rng.integers(0, CANVAS_H - size + 1))
                if any(x0 < ox + size and x1 > ox and
                       y0 < oy + size and y1 > oy
                       for _, x0, y0, x1, y1 in rects):
                    continue
                out[size].append((img[:, oy:oy + size, ox:ox + size].clone(),
                                  torch.zeros(size, size), 0,
                                  torch.tensor([ox, oy])))
                made += 1
        return out

    @staticmethod
    def _window(img, rects, cx, cy, size):
        ox = int(np.clip(round(cx - size / 2), 0, CANVAS_W - size))
        oy = int(np.clip(round(cy - size / 2), 0, CANVAS_H - size))
        crop = img[:, oy:oy + size, ox:ox + size].clone()
        return crop, rect_mask(rects, ox, oy, size), torch.tensor([ox, oy])

    def sample_weights(self):
        w = self.base.sample_weights().numpy()
        return torch.from_numpy(w[np.asarray(self.idx)])


def rect_mask(rects, ox, oy, size):
    """Union of pixel rects rendered white inside a size x size window at
    (ox, oy). Shared by GT mask building and the YOLO benchmark's
    prediction rasterizer, so both sides use identical geometry."""
    mask = torch.zeros(size, size)
    for _, x0, y0, x1, y1 in rects:
        xa, ya = max(int(x0) - ox, 0), max(int(y0) - oy, 0)
        xb = min(int(np.ceil(x1)) - ox, size)
        yb = min(int(np.ceil(y1)) - oy, size)
        if xa < xb and ya < yb:
            mask[ya:yb, xa:xb] = 1.0
    return mask


def collate_patches(items):
    """Per-image dicts -> {"40": (imgs, masks, labels, origins), "100": ...}.
    A size group is absent when the batch has no crops at that size —
    consumers must iterate over whatever keys exist."""
    out = {}
    for size in (40, 100):
        quad = [t for d in items for t in d[size]]
        if quad:
            out[str(size)] = (torch.stack([c for c, _, _, _ in quad]),
                              torch.stack([m for _, m, _, _ in quad]),
                              torch.tensor([y for _, _, y, _ in quad]),
                              torch.stack([o for _, _, _, o in quad]))
    return out
