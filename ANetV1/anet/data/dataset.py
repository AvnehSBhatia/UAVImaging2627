"""Loader for datasets/suas-synth-50k (YOLO format, mixed sources).

Realities handled here (see dataset README):
  - synthetic frames are 1920x1080; VisDrone (`vd_*`) frames vary in size
    -> everything is letterboxed to 960x540 with boxes remapped
  - VisDrone person boxes dominate mannequin counts -> per-source sampler weight
  - teacher soft grids (distillation) load from a cache dir keyed by stem
  - cache=True: one-time preprocessing pass into memmapped .npy files under
    <root>/.anet_cache/ (letterboxed uint8 frames + grids + bands + boxes).
    After that an item is a ~1.5 MB memcpy instead of a PIL decode + 1080p
    bilinear resize (~10 ms) — at MI300X training speeds the in-process decode
    IS the epoch time. Train split costs ~70 GB of disk, built once.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .rasterize import (
    CANVAS_H,
    CANVAS_W,
    boxes_to_grid,
    boxes_to_grid_band,
    boxes_to_heatmap,
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
                 vd_weight=0.4, mannequin_weight=4.0, tent_weight=2.0, uint8=False,
                 band_lo=None, cache=False, center=False, center_sigma=1.5,
                 center_grid=None, center_dual=False):
        self.root = Path(root)
        self.split = split
        self.coverage_thresh = coverage_thresh
        # band_lo: also emit out["band"] (2,54,96 bool) — per-class partial-
        # coverage cells in [band_lo, coverage_thresh), used by the loss as an
        # ignore band (boundary label noise). None = off (eval scripts).
        self.band_lo = band_lo
        # center=True: also emit v12's object-center targets ("heat",
        # "offset", "reg_mask") on the 27x48 stride-20 grid, rasterized from
        # the same transformed+padded boxes already produced for the v9 grid.
        # center_grid=(H, W) overrides the target grid (v22's stride-10
        # mannequin readout uses (54, 96)); None keeps the family default.
        # Rasterization happens per item at load time (never memmap-cached),
        # so changing the grid needs no cache rebuild; sigma is in cells of
        # the chosen grid.
        self.center = center
        self.center_sigma = center_sigma
        self.center_grid = center_grid
        # center_dual=True (v23/D76): emit PER-CLASS targets on per-class
        # grids — mannequin at stride-10 (54x96) where a 49x13px person is
        # ~5x1.3 cells (elongation representable) and tent at the family
        # stride-20 (27x48), unchanged. sigma is in CELLS, so the mannequin
        # sigma is doubled to 3.0 to keep the splat the same ~30px on the
        # ground as v13's 1.5 cells at stride-20.
        self.center_dual = center_dual
        # uint8=True ships images as (3,H,W) uint8 and leaves the /255 float
        # conversion to the consumer (the Trainer does it on-GPU): 4x less
        # pinned-memory + H2D traffic per image and no fp32 convert in the
        # loader workers. Default False so eval/viz scripts see [0,1] floats.
        self.uint8 = uint8
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

        # per-process memmap handles (opened lazily; never pickled to workers)
        self._cache_files = None
        self._mm = None
        self._mm_pid = None
        if cache:
            self._cache_files = self._build_or_reuse_cache()

    # ------------------------------------------------------------------ cache
    def _cache_dir(self):
        tag = f"{self.split}_c{self.coverage_thresh:g}_b" \
              f"{'off' if self.band_lo is None else format(self.band_lo, 'g')}"
        return self.root / ".anet_cache" / tag

    def _build_or_reuse_cache(self):
        d = self._cache_dir()
        meta_path = d / "meta.json"
        stems = [p.stem for p in self.items]
        names = ["images", "grids", "boxes"] + (["bands"] if self.band_lo is not None else [])
        files = {n: d / f"{n}.npy" for n in names}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("stems") == stems and all(f.exists() for f in files.values()):
                return files
            print(f"[SUASCells] cache at {d} is stale (item list changed) — rebuilding")
        d.mkdir(parents=True, exist_ok=True)
        n = len(self.items)
        gb = n * 3 * CANVAS_H * CANVAS_W / 2**30
        print(f"[SUASCells] building {self.split} cache: {n} frames -> {d} "
              f"(~{gb:.1f} GB, one-time)", flush=True)
        mm = {
            "images": np.lib.format.open_memmap(
                files["images"], mode="w+", dtype=np.uint8, shape=(n, 3, CANVAS_H, CANVAS_W)),
            "grids": np.lib.format.open_memmap(
                files["grids"], mode="w+", dtype=np.int8, shape=(n, 54, 96)),
            "boxes": np.lib.format.open_memmap(
                files["boxes"], mode="w+", dtype=np.float32, shape=(n, MAX_BOXES, 5)),
        }
        if self.band_lo is not None:
            mm["bands"] = np.lib.format.open_memmap(
                files["bands"], mode="w+", dtype=bool, shape=(n, 2, 54, 96))
        t0 = time.time()
        for i in range(n):
            s = self._load_raw(i)
            mm["images"][i] = s["image_u8"]
            mm["grids"][i] = s["grid"].astype(np.int8)
            mm["boxes"][i] = s["boxes"]
            if self.band_lo is not None:
                mm["bands"][i] = s["band"]
            if (i + 1) % 2000 == 0 or i + 1 == n:
                r = (i + 1) / (time.time() - t0)
                print(f"[SUASCells]   {i + 1}/{n} ({r:.0f} img/s, "
                      f"eta {(n - i - 1) / r:.0f}s)", flush=True)
        for a in mm.values():
            a.flush()
        del mm
        # meta.json written LAST = completion marker (crash-safe rebuild)
        meta_path.write_text(json.dumps(
            {"stems": stems, "coverage_thresh": self.coverage_thresh,
             "band_lo": self.band_lo}))
        return files

    def _maps(self):
        # (re)open memmaps once per process — spawn/fork workers each get their
        # own handles instead of a pickled 70 GB array
        if self._mm is None or self._mm_pid != os.getpid():
            self._mm = {k: np.load(f, mmap_mode="r") for k, f in self._cache_files.items()}
            self._mm_pid = os.getpid()
        return self._mm

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"], state["_mm_pid"] = None, None  # handles are per-process
        return state

    # ------------------------------------------------------------------ items
    def __len__(self):
        return len(self.items)

    def sample_weights(self):
        return torch.from_numpy(self._weights)

    def is_visdrone(self, i):
        return self.items[i].stem.startswith("vd_")

    def _load_raw(self, i):
        """Decode + letterbox + rasterize one item (uint8 image, numpy parts)."""
        path = self.items[i]
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        s, nw, nh, px, py = letterbox_params(w0, h0)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (PAD_VALUE,) * 3)
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        image_u8 = np.ascontiguousarray(np.asarray(canvas, np.uint8).transpose(2, 0, 1))

        boxes = transform_boxes(_read_label(self.label_dir / f"{path.stem}.txt"), w0, h0)
        band = None
        if self.band_lo is not None:
            grid, band = boxes_to_grid_band(boxes, self.coverage_thresh, self.band_lo)
        else:
            grid = boxes_to_grid(boxes, self.coverage_thresh)
        padded = np.full((MAX_BOXES, 5), -1.0, np.float32)
        n = min(len(boxes), MAX_BOXES)
        if n:
            padded[:n] = boxes[:n]
        return {"image_u8": image_u8, "grid": grid, "boxes": padded, "band": band}

    def __getitem__(self, i):
        if self._cache_files is not None:
            mm = self._maps()
            image_u8 = np.array(mm["images"][i])  # copy out of the memmap page
            grid = torch.from_numpy(np.array(mm["grids"][i])).long()
            padded = torch.from_numpy(np.array(mm["boxes"][i]))
            band = (torch.from_numpy(np.array(mm["bands"][i]))
                    if self.band_lo is not None else None)
        else:
            s = self._load_raw(i)
            image_u8 = s["image_u8"]
            grid = torch.from_numpy(s["grid"]).long()
            padded = torch.from_numpy(s["boxes"])
            band = torch.from_numpy(s["band"]) if s["band"] is not None else None

        if self.uint8:
            tensor = torch.from_numpy(image_u8)
        else:
            tensor = torch.from_numpy(image_u8.astype(np.float32) / 255.0)

        out = {"image": tensor, "grid": grid, "boxes": padded,
               "vd": self.is_visdrone(i), "stem": self.items[i].stem}
        if band is not None:
            out["band"] = band
        if self.center_dual:
            # v23 (D76): one rasterization per class, each on its own grid.
            # Rasterization is per-item at load time, so the memmap cache is
            # untouched by the grid change (no rebuild needed).
            b = padded.numpy()
            for pre, cls, grid, sig in (("mann", 0, (54, 96), 3.0),
                                        ("tent", 1, (27, 48), 1.5)):
                h, o, r = boxes_to_heatmap(b, sigma=sig, grid_hw=grid,
                                           classes=(cls,))
                out[f"{pre}_heat"] = torch.from_numpy(h)
                out[f"{pre}_offset"] = torch.from_numpy(o)
                out[f"{pre}_reg_mask"] = torch.from_numpy(r)
        elif self.center:
            # padded is the same canvas-normalized, -1-padded box set already
            # rasterized into `grid` above; boxes_to_heatmap ignores the -1
            # padding rows (their class index isn't 0 or 1).
            kw = {"grid_hw": self.center_grid} if self.center_grid else {}
            heat, offset, reg_mask = boxes_to_heatmap(
                padded.numpy(), sigma=self.center_sigma, **kw)
            out["heat"] = torch.from_numpy(heat)
            out["offset"] = torch.from_numpy(offset)
            out["reg_mask"] = torch.from_numpy(reg_mask)
        if self.teacher_dir is not None:
            soft = np.load(self.teacher_dir / f"{self.items[i].stem}.npz")["grid"]
            out["teacher"] = torch.from_numpy(soft)
        return out
