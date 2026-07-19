"""Per-image stage dump + prediction overlay for deep visual analysis.

Runs a checkpoint on N images containing a target class and, for each image,
writes a folder with every pipeline stage as a PNG plus a prediction overlay.
Also writes a contact sheet of all overlays for a quick scan.

  python scripts/visualize.py --ckpt runs/anet/best.pt --n 30 --cls mannequin
  v8/v9 -> runs/viz/<stem>/{00_input, 01_stem_*, 02_embedding*, 03_pathA_*,
                            04_logit_*, 05_prob_*, 06_overlay}.png + stats.txt
  v13   -> runs/viz/<stem>/{00_input, 01_stem_*, 02_s4_*, 03_s20_*,
                            04_heat_logit_* / 04_offset_*, 05_heat_prob_*,
                            06_overlay}.png + stats.txt
  both  -> runs/viz/_contact_sheet.png

v8/v9 stages: stem features (colour + oriented edges for edge_dq, or colour +
high-pass), the encoder embedding map, the 3 Path-A context scales, per-class
cell logits/probs, connected-cell region boxes vs GT.
v13 stages (D58 conv backbone): stem s2 map, s4 stage, every s20 block, per-
class center-heatmap logits/probs, dx/dy offset maps, and a peak-level overlay
(the same 3x3-local-max + offset de-quantization the metrics use): matched GT
green, MISSED GT yellow, matched peaks filled dots, false peaks crossed out.
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
from anet.train.metrics import V12_H, V12_W, CenterObjectMetrics  # noqa: E402

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


def prob_overlay(base_rgb, prob2d, color, alpha=0.85):  # blend a prob map onto the frame
    a = np.asarray(base_rgb, np.float32)
    p = prob2d.detach().cpu().float().numpy() if torch.is_tensor(prob2d) else prob2d
    pm = np.asarray(Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8))
                    .resize((a.shape[1], a.shape[0]), Image.BILINEAR), np.float32)[..., None] / 255
    out = a * (1 - alpha * pm) + np.asarray(color, np.float32) * (alpha * pm)
    return Image.fromarray(out.astype(np.uint8), "RGB")


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


# ---- v13 (D58 conv backbone): staged forward + peak-level overlay ----------
@torch.no_grad()
def stages_v13(model, img):
    """Re-run the backbone stage by stage (same modules, same order as
    V13Backbone.forward) so every intermediate map can be dumped."""
    bb = model.backbone
    x_stem = bb.act(bb.stem_norm(bb.stem(img)))   # (1, 16, 270, 480)
    x_s4 = bb.block4(bb.down4(x_stem))            # (1, 32, 135, 240)
    x = bb.down20(x_s4)                           # (1, 64, 27, 48)
    x_blocks = []
    for blk in bb.blocks:
        x = blk(x)
        x_blocks.append(x)
    out = bb.head(x)                              # (1, 4, 27, 48)
    return x_stem, x_s4, x_blocks, {"heat": out[:, 0:2], "offset": out[:, 2:4]}


@torch.no_grad()
def stages_v22(model, img):
    """V22Backbone stages including the peak-augmented SPD funnel branch."""
    import torch.nn.functional as F
    bb = model.backbone
    x_stem = bb.act(bb.stem_norm(bb.stem(img)))
    x_s4 = bb.block4(bb.down4(x_stem))
    spd = bb.spd_proj(x_s4)
    peak = bb.peak_proj(F.max_pool2d(x_s4, 5, 5))
    branch = bb.act(bb.spd_norm(spd + peak))
    gain = 2.0 * torch.tanh(bb.spd_gain)
    donor = bb.down20(x_s4)
    x = donor + gain * branch
    x_blocks = []
    for blk in bb.blocks:
        x = blk(x)
        x_blocks.append(x)
    out = bb.head(x)
    return {
        "stem": x_stem, "s4": x_s4, "spd": spd, "peak": peak,
        "branch": branch, "gain": gain, "donor": donor,
        "blocks": x_blocks,
        "out": {"heat": out[:, 0:2], "offset": out[:, 2:4]},
    }


def peaks_of(heat_prob, off_prob, thresh=0.3):
    """Per-class peak list [(cls, cx, cy, p)] in canvas-normalized coords —
    the exact CenterObjectMetrics pipeline (3x3 local max + offset dequant)."""
    out = []
    for c in range(2):
        rows, cols = CenterObjectMetrics._find_peaks(heat_prob[c], thresh)
        for r, cc in zip(rows, cols):
            cx = (cc + off_prob[0, r, cc]) / V12_W
            cy = (r + off_prob[1, r, cc]) / V12_H
            out.append((c + 1, float(cx), float(cy), float(heat_prob[c, r, cc])))
    return out


def overlay_center(base_rgb, gt_boxes, peaks):
    """GT boxes: green if some same-class peak lands inside, YELLOW if missed.
    Peaks: filled dot when matched to a GT box, crossed-out ring when false."""
    im = base_rgb.copy()
    d = ImageDraw.Draw(im)
    gt = [b for b in gt_boxes if b[0] >= 0]
    matched_gt, matched_pk = [False] * len(gt), [False] * len(peaks)
    for gi, box in enumerate(gt):
        bx, by, bw, bh = box[1], box[2], box[3], box[4]
        for pi, (pc, px, py, _) in enumerate(peaks):
            if pc == int(box[0]) + 1 and abs(px - bx) <= bw / 2 and abs(py - by) <= bh / 2:
                matched_gt[gi] = matched_pk[pi] = True
    for gi, box in enumerate(gt):
        cx, cy, w, h = box[1:] * np.array([CANVAS_W, CANVAS_H, CANVAS_W, CANVAS_H])
        color = GT_COLOR if matched_gt[gi] else (255, 220, 0)  # yellow = MISS
        d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    outline=color, width=2)
    for pi, (pc, px, py, p) in enumerate(peaks):
        x, y, rad = px * CANVAS_W, py * CANVAS_H, 5
        if matched_pk[pi]:
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=BOX_COLOR[pc])
        else:  # false peak: ring + cross
            d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=BOX_COLOR[pc], width=2)
            d.line([x - rad, y - rad, x + rad, y + rad], fill=BOX_COLOR[pc], width=2)
            d.line([x - rad, y + rad, x + rad, y - rad], fill=BOX_COLOR[pc], width=2)
        d.text((x + rad + 1, y - rad), f"{p:.2f}", fill=BOX_COLOR[pc])
    d.text((4, 4), "GT: green=found yellow=MISSED | peaks: dot=match ring+X=false "
                   "(red=mannequin blue=tent)", fill=(255, 255, 0))
    return im


def _dump_center_heats(d, base, out, s, extra_lines=None):
    """Shared heat/offset/overlay dump for center-heatmap archs."""
    heat_l = out["heat"][0].float().cpu()
    off_p = torch.sigmoid(out["offset"][0].float()).cpu().numpy()
    heat_p = torch.sigmoid(heat_l).numpy()
    for c, name in ((0, "mannequin"), (1, "tent")):
        heat(heat_l[c], signed=True, size=(CANVAS_W, CANVAS_H)).save(
            d / f"04_heat_logit_{name}.png")
        heat(heat_p[c], size=(CANVAS_W, CANVAS_H)).save(
            d / f"05_heat_prob_{name}.png")
        prob_overlay(base, heat_p[c], BOX_COLOR[c + 1]).save(
            d / f"05_heat_prob_{name}_overlay.png")
    heat(off_p[0], size=(CANVAS_W, CANVAS_H)).save(d / "04_offset_dx.png")
    heat(off_p[1], size=(CANVAS_W, CANVAS_H)).save(d / "04_offset_dy.png")
    pks = peaks_of(heat_p, off_p)
    ov = overlay_center(base, s["boxes"].numpy(), pks)
    ov.save(d / "06_overlay.png")
    gt = [b for b in s["boxes"].numpy() if b[0] >= 0]
    lines = [f"image {s['stem']}"]
    if extra_lines:
        lines.extend(extra_lines)
    for c, name in ((1, "mannequin"), (2, "tent")):
        boxes_c = [b for b in gt if int(b[0]) + 1 == c]
        pk_c = [p for p in pks if p[0] == c]
        found = sum(any(p[0] == c and abs(p[1] - b[1]) <= b[3] / 2
                        and abs(p[2] - b[2]) <= b[4] / 2 for p in pks)
                    for b in boxes_c)
        lines.append(
            f"{name}: gt={len(boxes_c)} found={found} peaks={len(pk_c)} "
            f"max_p={max((p[3] for p in pk_c), default=0.0):.3f} "
            f"max_heat={float(heat_p[c - 1].max()):.3f}")
    (d / "stats.txt").write_text("\n".join(lines) + "\n")
    return ov, lines[1:]


def dump_v13(model, s, img, d, base):
    """Full per-image stage dump for the conv backbone. Returns (overlay, stats)."""
    x_stem, x_s4, x_blocks, out = stages_v13(model, img)
    montage(x_stem[0].cpu().numpy(), cols=4, scale=1).save(d / "01_stem_montage.png")
    heat(x_stem[0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "01_stem_mag.png")
    montage(x_s4[0].cpu().numpy(), cols=8, scale=1).save(d / "02_s4_montage.png")
    heat(x_s4[0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "02_s4_mag.png")
    for bi, xb in enumerate(x_blocks):
        heat(xb[0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(
            d / f"03_s20_block{bi}_mag.png")
    montage(x_blocks[-1][0].cpu().numpy(), cols=8).save(d / "03_s20_final_montage.png")
    return _dump_center_heats(d, base, out, s)


def dump_v22(model, s, img, d, base):
    """v22 stage dump: donor down20 + SPD/peak branch + gain + heats."""
    st = stages_v22(model, img)
    montage(st["stem"][0].cpu().numpy(), cols=4, scale=1).save(d / "01_stem_montage.png")
    heat(st["stem"][0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "01_stem_mag.png")
    montage(st["s4"][0].cpu().numpy(), cols=8, scale=1).save(d / "02_s4_montage.png")
    heat(st["s4"][0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "02_s4_mag.png")
    heat(st["donor"][0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "03_donor_down20_mag.png")
    heat(st["spd"][0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "03_spd_proj_mag.png")
    heat(st["peak"][0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "03_peak_proj_mag.png")
    heat(st["branch"][0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / "03_spd_branch_mag.png")
    g = st["gain"][0, :, 0, 0].detach().cpu().float().numpy()
    for bi, xb in enumerate(st["blocks"]):
        heat(xb[0].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(
            d / f"03_s20_block{bi}_mag.png")
    montage(st["blocks"][-1][0].cpu().numpy(), cols=8).save(d / "03_s20_final_montage.png")
    extra = [
        f"spd_gain: mean={float(g.mean()):.4f} |g|_mean={float(np.abs(g).mean()):.4f} "
        f"max|g|={float(np.abs(g).max()):.4f}",
        f"branch/donor mag ratio="
        f"{float(st['branch'][0].norm(dim=0).mean() / (st['donor'][0].norm(dim=0).mean() + 1e-6)):.3f}",
    ]
    return _dump_center_heats(d, base, st["out"], s, extra_lines=extra)


# ---- staged forward (v8/v9) -------------------------------------------------
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
    ap.add_argument("--cls", choices=["tent", "mannequin", "any"], default="tent")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="runs/viz")
    args = ap.parse_args()
    cfg = anet_cfg()
    device = pick_device()

    sd = torch.load(args.ckpt, map_location=device)
    model = ANetV1.from_state_dict(sd, use_checkpoint=False).to(device).eval()
    enc = getattr(model, "encoder", None)  # v13 has no window encoder
    print(f"ckpt {args.ckpt} | arch={model.arch} "
          + (f"hidden={enc.hidden} " if enc is not None else "")
          + f"stem={model.stem} | device={device}")

    ds = SUASCells(cfg.data.root, args.split, coverage_thresh=cfg.data.coverage_thresh)
    if args.cls == "any":
        keep = lambda i: any(ds._has[i])  # noqa: E731
    else:
        has = 1 if args.cls == "tent" else 0  # ds._has columns: (has_m, has_t)
        keep = lambda i: ds._has[i][has]  # noqa: E731
    idx = [i for i in range(len(ds))
           if not ds.is_visdrone(i) and keep(i)][: args.n]
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
        if model.arch in ("v13", "v14", "v16", "v17", "v18", "v19", "v22"):
            d = out_root / s["stem"]
            d.mkdir(exist_ok=True)
            base = chw_to_rgb(s["image"])
            base.save(d / "00_input.png")
            dump = dump_v22 if model.arch == "v22" else dump_v13
            ov, stat_lines = dump(model, s, img, d, base)
            overlays.append(ov)
            print(f"[{rank + 1}/{len(idx)}] {s['stem']}  " + " | ".join(stat_lines))
            continue
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
        # Path A carries the 2 (x,y) coord channels (ramps 0..1) after the hidden
        # feature channels — exclude them or their gradient swamps the norm.
        h = model.encoder.hidden
        for mp, k in zip(maps, (3, 7, 11)):
            heat(mp[0, :h].norm(dim=0), size=(CANVAS_W, CANVAS_H)).save(d / f"03_pathA_k{k}.png")
        for c in range(3):
            heat(cells[c], signed=True, size=(CANVAS_W, CANVAS_H)).save(d / f"04_logit_{CLS[c]}.png")
            heat(probs[c], size=(CANVAS_W, CANVAS_H)).save(d / f"05_prob_{CLS[c]}.png")
        # prob blended onto the frame — localization against the real image
        prob_overlay(base, probs[1], BOX_COLOR[1]).save(d / "05_prob_mannequin_overlay.png")
        prob_overlay(base, probs[2], BOX_COLOR[2]).save(d / "05_prob_tent_overlay.png")
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
