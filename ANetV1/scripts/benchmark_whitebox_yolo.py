"""Class-agnostic object-presence bench: WhiteboxDQ vs YOLO26n.

Ignores class — a GT object is "found" if any predicted box's centre falls
inside it (same spirit as CenterObjectMetrics). Pred boxes that hit no GT
are FPs. Whitebox masks (full-frame, thresh 0.5) are converted to boxes via
8-connected components.

  cd ANetV1
  python scripts/benchmark_whitebox_yolo.py \
      --whitebox runs/whitebox/best.pt \
      --yolo runs/yolo/yolo26n/weights/best.pt \
      --split val --synth-only

Writes:
  runs/bench_whitebox_yolo/results.json
  runs/bench_whitebox_yolo/REPORT.md
  runs/bench_whitebox_yolo/boxes/<stem>/{gt,whitebox,yolo}.txt
  runs/bench_whitebox_yolo/overlays/<stem>.png   (first --n-overlay images)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet.data.dataset import SUASCells  # noqa: E402
from anet.data.rasterize import (  # noqa: E402
    CANVAS_H, CANVAS_W, transform_boxes,
)
from anet.probes import WhiteboxDQ  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import pick_device, yolo_device  # noqa: E402

OUT = Path("runs/bench_whitebox_yolo")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def mask_to_boxes(mask: np.ndarray, min_area=16):
    """Binary (H,W) -> list of canvas-normalized [cx,cy,w,h] via 8-cc."""
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    boxes = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < H and 0 <= xx < W
                            and mask[yy, xx] and not seen[yy, xx]):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        if len(cells) < min_area:
            continue
        rr = [c[0] for c in cells]
        cc = [c[1] for c in cells]
        y_min, y_max = min(rr), max(rr) + 1
        x_min, x_max = min(cc), max(cc) + 1
        bw, bh = x_max - x_min, y_max - y_min
        cx = (x_min + x_max) / 2.0 / W
        cy = (y_min + y_max) / 2.0 / H
        boxes.append([cx, cy, bw / W, bh / H])
    return boxes


def gt_boxes(sample):
    """Canvas-normalized [cx,cy,w,h] — class stripped."""
    out = []
    for b in sample["boxes"].numpy():
        if b[0] < 0:
            continue
        out.append([float(b[1]), float(b[2]), float(b[3]), float(b[4])])
    return out


def match_agnostic(gt, pred):
    """Centre-in-GT matching, class-agnostic. Returns (n_found, n_fp, found_flags)."""
    found = [False] * len(gt)
    matched_pred = [False] * len(pred)
    for pi, (cx, cy, _, _) in enumerate(pred):
        for gi, (gx, gy, gw, gh) in enumerate(gt):
            if found[gi]:
                continue
            if abs(cx - gx) <= gw / 2 and abs(cy - gy) <= gh / 2:
                found[gi] = True
                matched_pred[pi] = True
                break
    n_found = sum(found)
    n_fp = sum(1 for m in matched_pred if not m)
    return n_found, n_fp, found


def write_boxes(path: Path, boxes, confs=None):
    """One line per box: cx cy w h [conf]  (canvas-normalized, classless)."""
    lines = []
    for i, b in enumerate(boxes):
        if confs is not None:
            lines.append(f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {confs[i]:.4f}")
        else:
            lines.append(f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def draw_overlay(rgb, gt, wb, yolo, path: Path):
    im = rgb.copy()
    d = ImageDraw.Draw(im)

    def rect(box, color, width=2):
        cx, cy, w, h = box
        x0 = (cx - w / 2) * CANVAS_W
        y0 = (cy - h / 2) * CANVAS_H
        x1 = (cx + w / 2) * CANVAS_W
        y1 = (cy + h / 2) * CANVAS_H
        d.rectangle([x0, y0, x1, y1], outline=color, width=width)

    for b in gt:
        rect(b, (60, 220, 60), 2)
    for b in wb:
        rect(b, (255, 220, 40), 2)
    for b in yolo:
        rect(b, (60, 140, 255), 2)
    d.text((4, 4), "GT green | whitebox yellow | YOLO blue", fill=(255, 255, 0))
    im.save(path)


@torch.no_grad()
def run_whitebox(model, ds, idx, device, out_boxes: Path, min_area, batch=8):
    model.eval()
    per = {}
    # timed full pass (warmup separate)
    imgs = []
    meta = []
    for i in idx:
        s = ds[i]
        imgs.append(s["image"])
        meta.append((s["stem"], gt_boxes(s), s["image"]))
    # warmup
    x0 = imgs[0].unsqueeze(0).to(device)
    for _ in range(5):
        model(x0)
    _sync(device)

    t_infer = 0.0
    n_img = 0
    for start in range(0, len(imgs), batch):
        chunk = imgs[start:start + batch]
        x = torch.stack(chunk).to(device)
        _sync(device)
        t0 = time.perf_counter()
        logits = model(x)[:, 0]
        prob = torch.sigmoid(logits)
        _sync(device)
        t_infer += time.perf_counter() - t0
        masks = (prob > 0.5).cpu().numpy()
        for j, mask in enumerate(masks):
            stem, gt, rgb = meta[start + j]
            boxes = mask_to_boxes(mask, min_area=min_area)
            # conf = mean prob inside each box (recompute cheaply)
            confs = []
            H, W = mask.shape
            p = prob[j].cpu().numpy()
            for cx, cy, w, h in boxes:
                x0 = max(int((cx - w / 2) * W), 0)
                x1 = min(int((cx + w / 2) * W) + 1, W)
                y0 = max(int((cy - h / 2) * H), 0)
                y1 = min(int((cy + h / 2) * H) + 1, H)
                confs.append(float(p[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0)
            d = out_boxes / stem
            d.mkdir(parents=True, exist_ok=True)
            write_boxes(d / "gt.txt", gt)
            write_boxes(d / "whitebox.txt", boxes, confs)
            n_found, n_fp, _ = match_agnostic(gt, boxes)
            per[stem] = {
                "n_gt": len(gt), "n_pred": len(boxes),
                "n_found": n_found, "n_fp": n_fp,
                "boxes": boxes, "gt": gt,
                "rgb": rgb,
            }
            n_img += 1
        if (start // batch + 1) % 20 == 0:
            print(f"  whitebox {min(start + batch, len(imgs))}/{len(imgs)}",
                  flush=True)
    ms = (t_infer / max(n_img, 1)) * 1000
    return per, ms


def run_yolo(weights, ds, idx, cfg, conf, out_boxes: Path):
    from ultralytics import YOLO

    model = YOLO(weights)
    dev = yolo_device()
    # warmup
    p0 = str(ds.items[idx[0]])
    for _ in range(5):
        model.predict(p0, imgsz=cfg.yolo.imgsz, conf=conf, device=dev, verbose=False)

    per = {}
    t_infer = 0.0
    for k, i in enumerate(idx):
        path = str(ds.items[i])
        stem = Path(path).stem
        t0 = time.perf_counter()
        res = model.predict(path, imgsz=cfg.yolo.imgsz, conf=conf,
                            device=dev, verbose=False)[0]
        t_infer += time.perf_counter() - t0
        h0, w0 = res.orig_shape
        rows, confs = [], []
        if res.boxes is not None and len(res.boxes):
            xywhn = res.boxes.xywhn.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            sc = res.boxes.conf.cpu().numpy()
            for (cx, cy, w, h), c, s in zip(xywhn, cls, sc):
                rows.append([c, cx, cy, w, h])
                confs.append(float(s))
        canvas = transform_boxes(
            np.asarray(rows, np.float32).reshape(-1, 5), w0, h0)
        # strip class -> [cx,cy,w,h]
        boxes = [[float(b[1]), float(b[2]), float(b[3]), float(b[4])]
                 for b in canvas]
        sample = ds[i]
        gt = gt_boxes(sample)
        d = out_boxes / stem
        d.mkdir(parents=True, exist_ok=True)
        write_boxes(d / "yolo.txt", boxes, confs if confs else None)
        if not (d / "gt.txt").exists():
            write_boxes(d / "gt.txt", gt)
        n_found, n_fp, _ = match_agnostic(gt, boxes)
        per[stem] = {
            "n_gt": len(gt), "n_pred": len(boxes),
            "n_found": n_found, "n_fp": n_fp,
            "boxes": boxes, "gt": gt,
        }
        if (k + 1) % 50 == 0:
            print(f"  yolo {k + 1}/{len(idx)}", flush=True)
    ms = (t_infer / max(len(idx), 1)) * 1000
    # params
    n_par = sum(p.numel() for p in model.model.parameters())
    return per, ms, n_par


def summarize(per):
    n_gt = sum(v["n_gt"] for v in per.values())
    n_found = sum(v["n_found"] for v in per.values())
    n_fp = sum(v["n_fp"] for v in per.values())
    n_pred = sum(v["n_pred"] for v in per.values())
    n_img = len(per)
    recall = n_found / max(n_gt, 1)
    precision = n_found / max(n_found + n_fp, 1)
    return {
        "n_images": n_img,
        "n_gt_objects": n_gt,
        "n_pred_boxes": n_pred,
        "n_found": n_found,
        "n_fp": n_fp,
        "object_recall": recall,
        "precision": precision,
        "fp_per_image": n_fp / max(n_img, 1),
        "preds_per_image": n_pred / max(n_img, 1),
    }


def write_report(path: Path, results):
    wb, yo = results["whitebox"], results["yolo"]
    lines = [
        "# WhiteboxDQ vs YOLO26n — class-agnostic object presence",
        "",
        f"Split: `{results['meta']['split']}` · "
        f"synth_only={results['meta']['synth_only']} · "
        f"n={results['meta']['n_images']} · "
        f"device={results['meta']['device']} · "
        f"match=centre-in-GT · YOLO conf={results['meta']['yolo_conf']}",
        "",
        "| metric | WhiteboxDQ | YOLO26n |",
        "|---|---:|---:|",
        f"| params | {wb['params']:,} | {yo['params']:,} |",
        f"| latency ms/img | {wb['latency_ms']:.2f} | {yo['latency_ms']:.2f} |",
        f"| throughput img/s | {wb['throughput_img_s']:.1f} | {yo['throughput_img_s']:.1f} |",
        f"| object recall | {wb['object_recall']:.3f} | {yo['object_recall']:.3f} |",
        f"| precision | {wb['precision']:.3f} | {yo['precision']:.3f} |",
        f"| FP / image | {wb['fp_per_image']:.3f} | {yo['fp_per_image']:.3f} |",
        f"| preds / image | {wb['preds_per_image']:.2f} | {yo['preds_per_image']:.2f} |",
        f"| GT objects | {wb['n_gt_objects']} | {yo['n_gt_objects']} |",
        f"| found | {wb['n_found']} | {yo['n_found']} |",
        "",
        "Per-image boxes: `boxes/<stem>/{gt,whitebox,yolo}.txt` "
        "(cx cy w h [conf], canvas-normalized).",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whitebox", default="runs/whitebox/best.pt")
    ap.add_argument("--yolo", default="runs/yolo/yolo26n/weights/best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--synth-only", action="store_true", default=True)
    ap.add_argument("--include-vd", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-area", type=int, default=16,
                    help="min whitebox component area in px")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-overlay", type=int, default=24)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    synth_only = not args.include_vd

    cfg = anet_cfg()
    device = pick_device()
    root = os.environ.get("DATA_ROOT", "../datasets/suas-synth-50k")
    ds = SUASCells(root, args.split)
    idx = [i for i in range(len(ds))
           if (not synth_only) or (not ds.is_visdrone(i))]
    if args.limit:
        idx = idx[: args.limit]
    print(f"{args.split}: {len(idx)} images "
          f"(synth_only={synth_only}) device={device}")

    out = Path(args.out)
    boxes_dir = out / "boxes"
    ov_dir = out / "overlays"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    # --- whitebox ---
    sd = torch.load(args.whitebox, map_location=device, weights_only=False)
    arch = sd.get("arch", {}) if isinstance(sd, dict) else {}
    wb_model = WhiteboxDQ(
        stages=int(arch.get("stages", os.environ.get("ANET_STAGES", "16"))),
        width=int(arch.get("width", os.environ.get("ANET_CH", "32"))),
        kernels=arch.get("kernels", os.environ.get("ANET_K", "7,11,15")),
    ).to(device)
    wb_model.load_state_dict(sd["model"] if "model" in sd else sd)
    wb_params = sum(p.numel() for p in wb_model.parameters())
    print(f"WhiteboxDQ {wb_params} params — running…", flush=True)
    wb_per, wb_ms = run_whitebox(
        wb_model, ds, idx, device, boxes_dir, args.min_area, args.batch)
    wb = summarize(wb_per)
    wb["params"] = wb_params
    wb["latency_ms"] = wb_ms
    wb["throughput_img_s"] = 1000.0 / max(wb_ms, 1e-9)
    print(f"  whitebox recall={wb['object_recall']:.3f} "
          f"fp/img={wb['fp_per_image']:.2f}  {wb_ms:.2f} ms/img", flush=True)

    # --- yolo ---
    print(f"YOLO {args.yolo} — running…", flush=True)
    yo_per, yo_ms, yo_params = run_yolo(
        args.yolo, ds, idx, cfg, args.conf, boxes_dir)
    yo = summarize(yo_per)
    yo["params"] = yo_params
    yo["latency_ms"] = yo_ms
    yo["throughput_img_s"] = 1000.0 / max(yo_ms, 1e-9)
    print(f"  yolo recall={yo['object_recall']:.3f} "
          f"fp/img={yo['fp_per_image']:.2f}  {yo_ms:.2f} ms/img", flush=True)

    # --- overlays (first N with GT) ---
    n_ov = 0
    for stem in sorted(wb_per):
        if n_ov >= args.n_overlay:
            break
        if wb_per[stem]["n_gt"] == 0:
            continue
        rgb = wb_per[stem]["rgb"]
        a = rgb.numpy().transpose(1, 2, 0)
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")
        draw_overlay(im, wb_per[stem]["gt"],
                     wb_per[stem]["boxes"],
                     yo_per.get(stem, {}).get("boxes", []),
                     ov_dir / f"{stem}.png")
        n_ov += 1

    results = {
        "meta": {
            "split": args.split,
            "synth_only": synth_only,
            "n_images": len(idx),
            "device": str(device),
            "whitebox_ckpt": args.whitebox,
            "yolo_weights": args.yolo,
            "yolo_conf": args.conf,
            "match": "centre-in-GT (class-agnostic)",
            "whitebox_min_area_px": args.min_area,
            "canvas": [CANVAS_H, CANVAS_W],
        },
        "whitebox": wb,
        "yolo": yo,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    write_report(out / "REPORT.md", results)

    # console table
    print("\n" + "=" * 64)
    print(f"{'metric':22s} {'WhiteboxDQ':>14s} {'YOLO26n':>14s}")
    print("-" * 64)
    rows = [
        ("params", f"{wb['params']:,}", f"{yo['params']:,}"),
        ("latency_ms/img", f"{wb['latency_ms']:.2f}", f"{yo['latency_ms']:.2f}"),
        ("throughput_img/s", f"{wb['throughput_img_s']:.1f}",
         f"{yo['throughput_img_s']:.1f}"),
        ("object_recall", f"{wb['object_recall']:.3f}", f"{yo['object_recall']:.3f}"),
        ("precision", f"{wb['precision']:.3f}", f"{yo['precision']:.3f}"),
        ("fp_per_image", f"{wb['fp_per_image']:.3f}", f"{yo['fp_per_image']:.3f}"),
        ("preds_per_image", f"{wb['preds_per_image']:.2f}",
         f"{yo['preds_per_image']:.2f}"),
        ("n_gt_objects", str(wb["n_gt_objects"]), str(yo["n_gt_objects"])),
        ("n_found", str(wb["n_found"]), str(yo["n_found"])),
    ]
    for name, a, b in rows:
        print(f"{name:22s} {a:>14s} {b:>14s}")
    print("=" * 64)
    print(f"\nresults -> {out / 'results.json'}")
    print(f"report  -> {out / 'REPORT.md'}")
    print(f"boxes   -> {boxes_dir}/<stem>/{{gt,whitebox,yolo}}.txt")
    print(f"overlays-> {ov_dir}/ ({n_ov} images)")


if __name__ == "__main__":
    main()
