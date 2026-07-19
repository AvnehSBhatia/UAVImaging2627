"""Per-patch stage dump for the P1 WhiteboxDQ probe (OBSERVATIONS.md).

  cd ANetV1
  python scripts/visualize_whitebox.py --ckpt runs/whitebox/best.pt --n 8

Writes runs/viz_whitebox/<stem>/{00_input, 01_full_pred, 02_patches.png,
stats.txt} plus runs/viz_whitebox/_contact_sheet.png.
Each patch row: crop | GT mask | stage0..N RGB | pred prob | pred@0.5.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet.data.rasterize import CANVAS_H, CANVAS_W  # noqa: E402
from anet.probes import PatchCrops, WhiteboxDQ  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

CLS = ("bg", "mannequin", "tent")
ROOT_DEFAULT = "../datasets/suas-synth-50k"


def to_rgb(t):
    a = t.detach().cpu().float().numpy()
    a = np.clip(a, 0, 1)
    return Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB")


def gray01(a, size=None):
    a = a.detach().cpu().float().numpy() if torch.is_tensor(a) else a
    a = np.clip(a, 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8), "L").convert("RGB")
    if size:
        im = im.resize(size, Image.NEAREST)
    return im


def heat(a2d, size=None):
    a = a2d.detach().cpu().float().numpy() if torch.is_tensor(a2d) else a2d
    v = (a - a.min()) / (a.max() - a.min() + 1e-6)
    rgb = np.stack([np.clip(3 * v, 0, 1), np.clip(3 * v - 1, 0, 1),
                    np.clip(3 * v - 2, 0, 1)], -1)
    im = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
    if size:
        im = im.resize(size, Image.NEAREST)
    return im


def row_montage(imgs, pad=3, bg=(20, 20, 20), label=None):
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    sheet = Image.new("RGB", (w, h + (18 if label else 0)), bg)
    x = 0
    for im in imgs:
        sheet.paste(im, (x, 18 if label else 0))
        x += im.width + pad
    if label:
        ImageDraw.Draw(sheet).text((2, 2), label, fill=(230, 230, 230))
    return sheet


@torch.no_grad()
def forward_stages(model, x):
    """Return list of intermediate RGB maps + final logit (B,1,S,S)."""
    if hasattr(model, "intermediates"):
        return model.intermediates(x)
    stages = []
    h = x
    for dq, act in zip(model.dq, model.act):
        h = act(dq(h))
        stages.append(h)
    return stages, model.out(h)


@torch.no_grad()
def dump_image(model, ds, k, out_dir: Path, max_patches=12):
    item = ds[k]
    base = ds.base[ds.idx[k]]
    stem = base["stem"]
    out_dir.mkdir(parents=True, exist_ok=True)

    full = base["image"].unsqueeze(0).to(next(model.parameters()).device)
    stages_f, logit_f = forward_stages(model, full)
    to_rgb(full[0]).save(out_dir / "00_input.png")
    pred_f = torch.sigmoid(logit_f[0, 0])
    heat(pred_f, size=(CANVAS_W, CANVAS_H)).save(out_dir / "01_full_pred.png")
    ov = to_rgb(full[0]).convert("RGBA")
    mask_rgba = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    pm = (pred_f.cpu().numpy() > 0.5)
    rgba = np.zeros((CANVAS_H, CANVAS_W, 4), np.uint8)
    rgba[pm] = (255, 255, 255, 140)
    mask_rgba = Image.fromarray(rgba, "RGBA")
    Image.alpha_composite(ov, mask_rgba).convert("RGB").save(
        out_dir / "01_full_overlay.png")

    rows = []
    stats = [f"image {stem}", f"full_pred white_frac={(pred_f > 0.5).float().mean():.4f}"]
    n_obj = n_bg = iou_num = iou_den = 0.0
    device = next(model.parameters()).device

    patches = []
    for size in (40, 100):
        for crop, mask, label, org in item[size]:
            patches.append((size, crop, mask, int(label), org))
    # object crops first, then bg; cap total
    patches.sort(key=lambda t: (0 if t[3] > 0 else 1, t[0]))
    patches = patches[:max_patches]

    for i, (size, crop, mask, label, org) in enumerate(patches):
        x = crop.unsqueeze(0).to(device)
        stages, logit = forward_stages(model, x)
        prob = torch.sigmoid(logit[0, 0])
        pred = (prob > 0.5).float()
        m = mask.to(device)
        inter = (pred * m).sum().item()
        union = ((pred + m) > 0).float().sum().item()
        if label > 0:
            n_obj += 1
            iou_num += inter
            iou_den += max(union, 1.0)
        else:
            n_bg += 1
        cells = [
            to_rgb(crop).resize((size * 2, size * 2), Image.NEAREST),
            gray01(mask, size=(size * 2, size * 2)),
        ]
        for s in stages:
            cells.append(to_rgb(s[0]).resize((size * 2, size * 2), Image.NEAREST))
        cells.append(heat(prob, size=(size * 2, size * 2)))
        cells.append(gray01(pred, size=(size * 2, size * 2)))
        ox, oy = int(org[0]), int(org[1])
        lab = (f"{i:02d} {CLS[label]}@{size} ox={ox},oy={oy} "
               f"iou={inter / max(union, 1):.3f} white={float(pred.mean()):.3f}")
        rows.append(row_montage(cells, label=lab))
        stats.append(lab)

    if rows:
        pad = 4
        W = max(r.width for r in rows)
        H = sum(r.height for r in rows) + pad * (len(rows) - 1)
        sheet = Image.new("RGB", (W, H), (20, 20, 20))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + pad
        sheet.save(out_dir / "02_patches.png")

    iou = iou_num / max(iou_den, 1.0)
    stats.insert(2, f"obj_crops={int(n_obj)} bg_crops={int(n_bg)} mean_iou@0.5={iou:.4f}")
    (out_dir / "stats.txt").write_text("\n".join(stats) + "\n")
    return Image.open(out_dir / "01_full_overlay.png"), stem, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/whitebox/best.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="runs/viz_whitebox")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--max-patches", type=int, default=12)
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {}) if isinstance(ckpt, dict) else {}
    model = WhiteboxDQ(
        stages=int(arch.get("stages", os.environ.get("ANET_STAGES", "16"))),
        width=int(arch.get("width", os.environ.get("ANET_CH", "32"))),
        kernels=arch.get("kernels", os.environ.get("ANET_K", "7,11,15")),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    n_par = sum(p.numel() for p in model.parameters())
    v = ckpt.get("val", {}) if isinstance(ckpt, dict) else {}
    print(f"ckpt {args.ckpt} | WhiteboxDQ {n_par} params | device={device}  "
          f"ep={ckpt.get('epoch')}  iou={v.get('iou', float('nan')):.3f}  "
          f"bg_white={v.get('bg_white', float('nan')):.3f}")

    ds = PatchCrops(args.root, args.split, include_vd=False)
    # prefer images that have at least one object crop
    idx = []
    for k in range(len(ds)):
        item = ds[k]
        if any(lab > 0 for size in (40, 100) for _, _, lab, _ in item[size]):
            idx.append(k)
        if len(idx) >= args.n:
            break
    print(f"{len(idx)} synthetic {args.split} images with object crops")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    overlays = []
    for rank, k in enumerate(idx):
        # resolve stem via base
        stem = ds.base[ds.idx[k]]["stem"]
        ov, stem, stats = dump_image(model, ds, k, out_root / stem, args.max_patches)
        overlays.append(ov)
        print(f"[{rank + 1}/{len(idx)}] {stem}  {stats[2]}")

    if overlays:
        cols = min(4, len(overlays))
        rows = (len(overlays) + cols - 1) // cols
        tw, th = CANVAS_W // 2, CANVAS_H // 2
        sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
        for i, ov in enumerate(overlays):
            sheet.paste(ov.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
        sheet.save(out_root / "_contact_sheet.png")
        print(f"\ncontact sheet -> {out_root / '_contact_sheet.png'}")
    print(f"per-image dumps -> {out_root}/<stem>/")


if __name__ == "__main__":
    main()
