"""Per-image stage dump + prediction overlay for deep visual analysis.

Runs a checkpoint on N images containing a target class (tent by default) and,
for each image, writes a folder with every pipeline stage as a PNG plus a
prediction overlay (GT boxes vs predicted region boxes). Also writes a contact
sheet of all overlays for a quick scan.

  python scripts/visualize.py --ckpt runs/anet/last.pt --n 30 --cls tent
  -> runs/viz/<stem>/{00_input, 01_stem_*, 02_embedding*, 03_pathA_*,
                      04_logit_*, 05_prob_*, 06_overlay}.png + stats.txt
     runs/viz/_contact_sheet.png

Stages dumped (see ARCHITECTURE.md): stem features (colour + oriented edges for
edge_dq, or colour + high-pass for highpass), the encoder embedding map, the 3
Path-A context scales (activation magnitude), and per-class cell logits/probs.
Deps: torch + numpy + pillow only (no matplotlib).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.data.dataset import SUASCells  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402
from anet.data.rasterize import CANVAS_H, CANVAS_W, GRID_H, GRID_W  # noqa: E402

CLS = ("background", "mannequin", "tent")
CELL = CANVAS_H // GRID_H  # 10 px per cell
BOX_COLOR = {1: (255, 60, 60), 2: (60, 140, 255)}  # pred: mannequin red, tent blue
GT_COLOR = (60, 220, 60)


# ---- small image helpers (PIL only) ---------------------------------------
def chw_to_rgb(t):  # (3,H,W) in ~[0,1] -> PIL RGB
    a = t.detach().cpu().float().numpy()
    a = a - a.min((1, 2), keepdims=True)
    a = a / (a.max((1, 2), keepdims=True) + 1e-6)
    return Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB")


def norm01(a):
    a = a.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def heat(a2d, signed=False, size=None):  # 2D array -> PIL RGB heatmap
    a = a2d.detach().cpu().float().numpy() if torch.is_tensor(a2d) else a2d
    if signed:
        m = np.abs(a).max() + 1e-6
        v = a / m  # [-1,1]
        r = norm01(np.clip(v, 0, 1))
        b = norm01(np.clip(-v, 0, 1))
        rgb = np.stack([r, np.zeros_like(r), b], -1)  # red=+ blue=-
    else:
        v = norm01(a)  # hot: black->red->yellow->white
        rgb = np.stack([np.clip(3 * v, 0, 1), np.clip(3 * v - 1, 0, 1),
                        np.clip(3 * v - 2, 0, 1)], -1)
    im = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
    if size:
        im = im.resize(size, Image.NEAREST)
    return im


def montage(chans, cols=6, scale=3):  # (C,h,w) -> tiled grayscale PIL
    c, h, w = chans.shape
    rows = (c + cols - 1) // cols
    sheet = Image.new("L", (cols * w * scale, rows * h * scale), 0)
    for i in range(c):
        tile = Image.fromarray((norm01(chans[i]) * 255).astype(np.uint8), "L")
        tile = tile.resize((w * scale, h * scale), Image.NEAREST)
        sheet.paste(tile, ((i % cols) * w * scale, (i // cols) * h * scale))
    return sheet


def components(mask):  # bool (H,W) -> [(r0,c0,r1,c1)] connected-component boxes
    seen = np.zeros_like(mask, bool)
    out = []
    H, W = mask.shape
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or seen[i, j]:
                continue
            stack, r0, r1, c0, c1 = [(i, j)], i, i, j, j
            seen[i, j] = True
            while stack:
                r, c = stack.pop()
                r0, r1, c0, c1 = min(r0, r), max(r1, r), min(c0, c), max(c1, c)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append((r0, c0, r1, c1))
    return out


def overlay(base_rgb, gt_boxes, pred_grid):
    im = base_rgb.copy()
    d = ImageDraw.Draw(im)
    for box in gt_boxes:  # canvas-normalized [cls,cx,cy,w,h]
        if box[0] < 0:
            continue
        cx, cy, w, h = box[1:] * np.array([CANVAS_W, CANVAS_H, CANVAS_W, CANVAS_H])
        d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    outline=GT_COLOR, width=2)
    for cls in (1, 2):  # predicted region boxes from connected cells
        for r0, c0, r1, c1 in components(pred_grid == cls):
            d.rectangle([c0 * CELL, r0 * CELL, (c1 + 1) * CELL, (r1 + 1) * CELL],
                        outline=BOX_COLOR[cls], width=2)
    d.text((4, 4), "green=GT  red=pred-mannequin  blue=pred-tent", fill=(255, 255, 0))
    return im


# ---- staged forward --------------------------------------------------------
@torch.no_grad()
def stages(model, img):
    feat = model._features(img)                                    # (1,feat,H,W)
    m16 = model._map_dense(feat) if model.dense else model._map_windowed(feat)
    m = torch.cat([m16, model.xy_map.expand(img.shape[0], -1, -1, -1)], 1)
    maps = [p(m) for p in model.pools]                             # Path A (3 scales)
    cells = model(img)                                             # full forward -> logits
    return feat, m16, maps, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/anet/last.pt")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cls", choices=["tent", "mannequin"], default="tent")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="runs/viz")
    args = ap.parse_args()
    cfg = anet_cfg()
    device = pick_device()

    sd = torch.load(args.ckpt, map_location=device)
    model = ANetV1.from_state_dict(sd, use_checkpoint=False).to(device).eval()
    print(f"ckpt {args.ckpt} | hidden={model.encoder.hidden} stem={model.stem} | device={device}")

    ds = SUASCells(cfg.data.root, args.split, coverage_thresh=cfg.data.coverage_thresh)
    want = 2 if args.cls == "tent" else 1
    has = 1 if args.cls == "tent" else 0  # ds._has columns: (has_m, has_t)
    idx = [i for i in range(len(ds))
           if not ds.is_visdrone(i) and ds._has[i][has]][: args.n]
    print(f"{len(idx)} synthetic {args.split} images containing {args.cls}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # stem channel groups for labeling
    groups = ([("colour", 0), ("vert_edge", 3), ("horiz_edge", 6)] if model.stem == "edge_dq"
              else [("colour", 0), ("highpass", 3)])
    overlays = []

    for rank, i in enumerate(idx):
        s = ds[i]
        img = s["image"].unsqueeze(0).to(device)
        feat, m16, maps, cells = stages(model, img)
        cells = cells[0].float().cpu()
        pred = cells.argmax(0).numpy()
        probs = torch.softmax(cells, 0)
        base = chw_to_rgb(s["image"])

        d = out_root / s["stem"]
        d.mkdir(exist_ok=True)
        base.save(d / "00_input.png")
        for name, off in groups:  # stem feature groups as RGB composites
            chw_to_rgb(feat[0, off:off + 3]).save(d / f"01_stem_{name}.png")
        montage(m16[0].cpu().numpy()).save(d / "02_embedding_montage.png")
        heat(m16[0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "02_embedding_mag.png")
        for mp, k in zip(maps, (3, 7, 11)):
            heat(mp[0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / f"03_pathA_k{k}.png")
        for c in range(3):
            heat(cells[c], signed=True, size=(CANVAS_W, CANVAS_H)).save(d / f"04_logit_{CLS[c]}.png")
            heat(probs[c], size=(CANVAS_W, CANVAS_H)).save(d / f"05_prob_{CLS[c]}.png")
        ov = overlay(base, s["boxes"].numpy(), pred)
        ov.save(d / "06_overlay.png")
        overlays.append(ov)

        gt = s["grid"].numpy()
        stat = {c: (int((gt == c).sum()), int((pred == c).sum())) for c in (1, 2)}
        (d / "stats.txt").write_text(
            f"image {s['stem']}\n"
            + "".join(f"{CLS[c]}: gt_cells={g} pred_cells={p}\n" for c, (g, p) in stat.items())
        )
        print(f"[{rank + 1}/{len(idx)}] {s['stem']}  "
              + " ".join(f"{CLS[c]} gt={g}/pred={p}" for c, (g, p) in stat.items()))

    # contact sheet of all overlays
    if overlays:
        cols = 5
        rows = (len(overlays) + cols - 1) // cols
        tw, th = CANVAS_W // 2, CANVAS_H // 2
        sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
        for k, ov in enumerate(overlays):
            sheet.paste(ov.resize((tw, th)), ((k % cols) * tw, (k // cols) * th))
        sheet.save(out_root / "_contact_sheet.png")
        print(f"\ncontact sheet -> {out_root / '_contact_sheet.png'}")
    print(f"per-image stage folders -> {out_root}/")


if __name__ == "__main__":
    main()
