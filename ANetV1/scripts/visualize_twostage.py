"""Per-image stage dump for the v21.5 chunk detector (D71).

  cd ANetV1
  python scripts/visualize_twostage.py --ckpt runs/twostage/best.pt --n 8

Writes runs/viz_twostage/<stem>/{00_input, 01_smooth, 02_edge, 03_saliency,
04_chunks, 05_overlay}.png + stats.txt and _contact_sheet.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet.data.dataset import SUASCells  # noqa: E402
from anet.data.rasterize import CANVAS_H, CANVAS_W  # noqa: E402
from anet.model.twostage import (  # noqa: E402
    GRID_H, GRID_W, IMG_H, IMG_W, STRIDE, V21TwoStage,
)
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

BOX_COLOR = {1: (255, 60, 60), 2: (60, 140, 255)}
GT_COLOR = (60, 220, 60)
CLS = ("BG", "mannequin", "tent")


def chw_to_rgb(t):
    a = t.detach().cpu().float().numpy() if torch.is_tensor(t) else t
    a = a - a.min((1, 2), keepdims=True)
    a = a / (a.max((1, 2), keepdims=True) + 1e-6)
    return Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB")


def to_rgb_clip(t):
    a = t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t)
    a = np.clip(a, 0, 1)
    return Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB")


def norm01(a):
    a = a.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def heat(a2d, size=None):
    a = a2d.detach().cpu().float().numpy() if torch.is_tensor(a2d) else a2d
    v = norm01(a)
    rgb = np.stack([np.clip(3 * v, 0, 1), np.clip(3 * v - 1, 0, 1),
                    np.clip(3 * v - 2, 0, 1)], -1)
    im = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
    if size:
        im = im.resize(size, Image.NEAREST)
    return im


def ker_img(k, scale=8):
    """(k,k) kernel -> RGB visualization (signed: red+/blue-)."""
    a = k.detach().cpu().float().numpy()
    m = np.abs(a).max() + 1e-6
    v = a / m
    r = np.clip(v, 0, 1)
    b = np.clip(-v, 0, 1)
    rgb = np.stack([r, np.zeros_like(r), b], -1)
    im = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
    return im.resize((k.shape[-1] * scale, k.shape[-2] * scale), Image.NEAREST)


def montage_row(imgs, pad=4, bg=(20, 20, 20)):
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    sheet = Image.new("RGB", (w, h), bg)
    x = 0
    for im in imgs:
        sheet.paste(im, (x, (h - im.height) // 2))
        x += im.width + pad
    return sheet


def overlay_detect(base_rgb, gt_boxes, peaks_cls):
    """GT boxes green/yellow; classified peaks as filled (match) or X (FP)."""
    im = base_rgb.copy()
    d = ImageDraw.Draw(im)
    gt = [b for b in gt_boxes if b[0] >= 0]
    matched_gt = [False] * len(gt)
    matched_pk = [False] * len(peaks_cls)
    for gi, box in enumerate(gt):
        bx, by, bw, bh = box[1], box[2], box[3], box[4]
        for pi, (pc, px, py, p) in enumerate(peaks_cls):
            if pc == int(box[0]) + 1 and abs(px - bx) <= bw / 2 and abs(py - by) <= bh / 2:
                matched_gt[gi] = matched_pk[pi] = True
    for gi, box in enumerate(gt):
        cx, cy, w, h = box[1:] * np.array([CANVAS_W, CANVAS_H, CANVAS_W, CANVAS_H])
        color = GT_COLOR if matched_gt[gi] else (255, 220, 0)
        d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    outline=color, width=2)
    for pi, (pc, px, py, p) in enumerate(peaks_cls):
        x, y, rad = px * CANVAS_W, py * CANVAS_H, 6
        if matched_pk[pi]:
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=BOX_COLOR[pc])
        else:
            d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=BOX_COLOR[pc], width=2)
            d.line([x - rad, y - rad, x + rad, y + rad], fill=BOX_COLOR[pc], width=2)
            d.line([x - rad, y + rad, x + rad, y - rad], fill=BOX_COLOR[pc], width=2)
        d.text((x + rad + 1, y - rad), f"{CLS[pc][0]}{p:.2f}", fill=BOX_COLOR[pc])
    d.text((4, 4),
           "GT green=found yellow=MISS | peaks: fill=match X=false (red=mann blue=tent)",
           fill=(255, 255, 0))
    return im


@torch.no_grad()
def dump_one(model, sample, img, out_dir: Path, peak_thresh=0.3):
    """v21.4 pipeline dump for one image. Returns overlay PIL.
    Panels: 00 input, 01 smooth quat, 02 edge stack, 03 saliency prob
    (peaks ringed), 04 dual-scale crop tiles with CropCNN verdicts,
    05 overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = to_rgb_clip(sample["image"])
    base.save(out_dir / "00_input.png")

    out = model(img)
    to_rgb_clip(out["smooth"][0].cpu()).save(out_dir / "01_smooth.png")
    chw_to_rgb(out["edge"][0].cpu()).save(out_dir / "02_edge.png")

    sal_prob = torch.sigmoid(out["sal_logits"])          # (1,27,48)
    thresh = float(out["thresh"][0])
    sal_img = heat(sal_prob[0].cpu(), size=(CANVAS_W, CANVAS_H))
    ImageDraw.Draw(sal_img).text(
        (4, 4), f"chunk thresh={thresh:.3f}", fill=(255, 255, 255))
    sal_img.save(out_dir / "03_saliency.png")

    chunks = model.image_chunks(sal_prob[0], out["thresh"][0])
    peaks_cls = []                                       # (cls1or2,cx,cy,p)
    chunk_img = Image.new("RGB", (CANVAS_W, CANVAS_H), (10,) * 3)
    dch = ImageDraw.Draw(chunk_img)
    palette = [(240, 90, 90), (90, 140, 240), (110, 220, 110),
               (230, 200, 80), (200, 110, 220), (90, 210, 210),
               (240, 150, 60), (160, 160, 240)]
    if chunks:
        cls_p = torch.softmax(
            model.chunk_logits(0, chunks, sal_prob, out), 1).cpu()
        for k, (cells, p) in enumerate(zip(chunks, cls_p)):
            col = palette[k % len(palette)]
            for r, c in cells:
                dch.rectangle([c * STRIDE, r * STRIDE,
                               (c + 1) * STRIDE - 1, (r + 1) * STRIDE - 1],
                              fill=col)
            pred = int(p.argmax())
            r0, c0 = model.chunk_centroid(cells, sal_prob[0])
            dch.text((c0 * STRIDE + 2, r0 * STRIDE + 2),
                     f"{CLS[pred][0]}{float(p.max()):.2f}", fill=(0, 0, 0))
            if pred > 0:
                peaks_cls.append((pred, (c0 + 0.5) / GRID_W,
                                  (r0 + 0.5) / GRID_H, float(p.max())))
    chunk_img.save(out_dir / "04_chunks.png")

    ov = overlay_detect(base, sample["boxes"].numpy(), peaks_cls)
    ov.save(out_dir / "05_overlay.png")

    gt = [b for b in sample["boxes"].numpy() if b[0] >= 0]
    lines = [
        f"image {sample['stem']}",
        f"sal max={float(sal_prob.max()):.3f} thresh={thresh:.3f} "
        f"n_chunks={len(chunks)}",
        f"n_fg_cls={len(peaks_cls)}",
    ]
    for c, name in ((1, "mannequin"), (2, "tent")):
        boxes_c = [b for b in gt if int(b[0]) + 1 == c]
        pk_c = [p for p in peaks_cls if p[0] == c]
        found = sum(
            any(p[0] == c and abs(p[1] - b[1]) <= b[3] / 2
                and abs(p[2] - b[2]) <= b[4] / 2 for p in peaks_cls)
            for b in boxes_c)
        lines.append(
            f"{name}: gt={len(boxes_c)} found={found} peaks={len(pk_c)} "
            f"max_p={max((p[3] for p in pk_c), default=0.0):.3f}")
    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n")
    return ov, lines



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/twostage/best.pt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cls", choices=["tent", "mannequin", "any"], default="mannequin")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="runs/viz_twostage")
    ap.add_argument("--peak-thresh", type=float, default=0.3)
    args = ap.parse_args()

    cfg = anet_cfg()
    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    model = V21TwoStage().to(device).eval()
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    n = sum(p.numel() for p in model.parameters())
    meta = ""
    if isinstance(ckpt, dict) and "val" in ckpt:
        v = ckpt["val"]
        meta = (f"  ep={ckpt.get('epoch')}  "
                f"mann={v.get('mannequin_recall', float('nan')):.3f}  "
                f"tent={v.get('tent_recall', float('nan')):.3f}  "
                f"fp={v.get('fp_per_image', float('nan')):.3f}")
    print(f"ckpt {args.ckpt} | V21TwoStage {n:,} params | device={device}{meta}")

    ds = SUASCells(cfg.data.root, args.split, coverage_thresh=cfg.data.coverage_thresh,
                   center=True)
    if args.cls == "any":
        keep = lambda i: any(ds._has[i])  # noqa: E731
    else:
        has = 1 if args.cls == "tent" else 0
        keep = lambda i: ds._has[i][has]  # noqa: E731
    idx = [i for i in range(len(ds))
           if not ds.is_visdrone(i) and keep(i)][: args.n]
    print(f"{len(idx)} synthetic {args.split} images containing {args.cls}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    overlays = []
    for rank, i in enumerate(idx):
        s = ds[i]
        img = s["image"].unsqueeze(0).to(device)
        d = out_root / s["stem"]
        ov, lines = dump_one(model, s, img, d, args.peak_thresh)
        overlays.append(ov)
        print(f"[{rank + 1}/{len(idx)}] {s['stem']}  " + " | ".join(lines[3:]))

    if overlays:
        cols = 5
        rows = (len(overlays) + cols - 1) // cols
        tw, th = CANVAS_W // 2, CANVAS_H // 2
        sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
        for k, ov in enumerate(overlays):
            sheet.paste(ov.resize((tw, th)), ((k % cols) * tw, (k // cols) * th))
        sheet.save(out_root / "_contact_sheet.png")
        print(f"\ncontact sheet -> {out_root / '_contact_sheet.png'}")
    print(f"per-image dumps -> {out_root}/<stem>/")


if __name__ == "__main__":
    main()
