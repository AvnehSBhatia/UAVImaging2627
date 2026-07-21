"""Side-by-side checkpoint comparison on the preserved real web scenes.

`runs/viz_web_scenes/infer/*/00_input.png` holds 14 real 960x540 photographs
— the cases that triggered the v23 redesign (§18.1) and the only real-imagery
evidence the project has. `visualize.py` dumps one checkpoint's internals on
DATASET frames; this dumps N checkpoints' mannequin heatmaps side by side on
those real frames, which is the comparison that actually shows whether a
change transfers off the renderer.

  python scripts/visualize_scenes.py --ckpts v13_best d85_best v22_d85_best

Per scene one row: input, then one heat overlay per checkpoint with the same
peak-finding the metrics use (3x3 local max above --peak-thresh) drawn as
circles. Red channel = mannequin probability, so a clean model shows a dark
frame with one bright dot and a noisy one shows a red haze — the D80 tail
made visible. Per-panel caption carries max p, count above threshold, and
the background p99, so the numbers §20.5 reports are readable off the image.

There are NO labels on these frames (they are web photographs), so nothing
here is a recall measurement — it is qualitative evidence, and the caption
numbers are distribution statistics, not scores against ground truth.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet.model.anet import ANetV1  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

W, H = 960, 540
SCALE = 0.5           # panel downscale; 14 rows x 4 panels is a big sheet
PAD, CAP = 6, 26


def load(ckpt, device):
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))
    m = ANetV1.from_state_dict(sd)
    m.load_state_dict(sd, strict=False)
    return m.eval().to(device)


def peaks(prob, thresh):
    """3x3 local maxima above thresh — the same rule CenterObjectMetrics uses."""
    p = torch.from_numpy(prob)[None, None]
    mx = torch.nn.functional.max_pool2d(p, 3, 1, 1)[0, 0].numpy()
    return np.argwhere((prob >= mx) & (prob > thresh))


def overlay(img, prob, thresh):
    """Input dimmed, mannequin probability painted into red, peaks circled."""
    grid = np.asarray(Image.fromarray((prob * 255).astype(np.uint8))
                      .resize((W, H), Image.NEAREST), np.float32) / 255.0
    base = np.asarray(img, np.float32) * 0.45
    base[..., 0] = np.clip(base[..., 0] + grid * 255.0, 0, 255)
    out = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(out)
    gh, gw = prob.shape
    for r, c in peaks(prob, thresh):
        x, y = (c + 0.5) * W / gw, (r + 0.5) * H / gh
        d.ellipse([x - 13, y - 13, x + 13, y + 13], outline=(0, 255, 0), width=3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["v13_best", "v22_d85_best"],
                    help="names under runs/anet/ (without .pt) or full paths")
    ap.add_argument("--scenes", default="runs/viz_web_scenes/infer")
    ap.add_argument("--peak-thresh", type=float, default=0.30)
    ap.add_argument("--out", default="runs/viz_scenes_compare.png")
    ap.add_argument("--per-scene", metavar="DIR",
                    help="also write FULL-RESOLUTION 960x540 overlays, one PNG "
                         "per (scene, checkpoint), into DIR. The contact sheet "
                         "is 0.5x and 4-up, which is unreadable for judging "
                         "individual detections.")
    args = ap.parse_args()

    device = pick_device()
    paths = [p if Path(p).exists() else f"runs/anet/{p}.pt" for p in args.ckpts]
    names = [Path(p).stem for p in paths]
    models = [load(p, device) for p in paths]
    for n, m in zip(names, models):
        print(f"{n:<16} arch={m.arch} params={sum(q.numel() for q in m.parameters()):,}")

    scenes = sorted(Path(args.scenes).glob("*/00_input.png"))
    print(f"{len(scenes)} scenes | peak_thresh={args.peak_thresh}")

    pw, ph = int(W * SCALE), int(H * SCALE)
    cols = 1 + len(models)
    sheet = Image.new("RGB", (cols * (pw + PAD) + PAD,
                              len(scenes) * (ph + CAP + PAD) + PAD + CAP),
                      (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    for j, lab in enumerate(["input"] + names):
        dr.text((PAD + j * (pw + PAD) + 4, 6), lab, fill=(235, 235, 235))

    per = Path(args.per_scene) if args.per_scene else None
    if per:
        per.mkdir(parents=True, exist_ok=True)
    print(f"\n{'scene':<28}" + "".join(f"{n[:13]:>26}" for n in names))
    print(f"{'':<28}" + "".join(f"{'max / pk / bgp99':>26}" for _ in names))
    print("-" * (28 + 26 * len(names)))

    for i, sp in enumerate(scenes):
        img = Image.open(sp).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        x = torch.from_numpy(
            np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0)[None].to(device)
        y0 = PAD + CAP + i * (ph + CAP + PAD)
        sheet.paste(img.resize((pw, ph)), (PAD, y0))
        dr.text((PAD + 4, y0 + ph + 5), sp.parent.name[:34], fill=(200, 200, 200))
        row = f"{sp.parent.name[:27]:<28}"
        for j, m in enumerate(models):
            with torch.no_grad():
                prob = torch.sigmoid(m(x)["heat"])[0, 0].cpu().numpy()
            full = overlay(img, prob, args.peak_thresh)
            sheet.paste(full.resize((pw, ph)), (PAD + (j + 1) * (pw + PAD), y0))
            n_pk = len(peaks(prob, args.peak_thresh))
            p99 = float(np.quantile(prob, .99))
            dr.text((PAD + (j + 1) * (pw + PAD) + 4, y0 + ph + 5),
                    f"max {prob.max():.2f}  peaks>{args.peak_thresh:g}: {n_pk}"
                    f"  bg p99 {p99:.2f}", fill=(200, 200, 200))
            row += f"{prob.max():>10.2f}{n_pk:>7d}{p99:>9.2f}"
            if per:
                full.save(per / f"{sp.parent.name}__{names[j]}.png")
        print(row, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"-> {args.out}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
