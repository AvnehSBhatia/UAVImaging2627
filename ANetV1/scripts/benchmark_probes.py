"""Benchmark the P1/P2 probes against YOLO26n on the IDENTICAL crops.

Run from inside ANetV1/. YOLO predicts on the full frame (its native
regime); its boxes are letterbox-remapped and then rasterized/classified
inside each PatchCrops window, so all three columns answer the same two
questions on the same pixels:

  mask metrics (P1's game): IoU@0.5 of white-painted GT boxes on object
    crops + white-fraction on background crops. YOLO's "mask" is its
    predicted boxes rendered white through the same rect_mask geometry.
  patch metrics (P2's game): bg accuracy + per-class recall of the crop's
    centre-object class. YOLO's patch label is the class of its predicted
    box with the largest overlap on the crop (bg when none intersects).

  python scripts/benchmark_probes.py \
      --whitebox runs/whitebox/best.pt --fivestack runs/fivestack/best.pt \
      --yolo runs/yolo/yolo26n/weights/best.pt [--split test] [--limit N] \
      [--latency]

Results ledger: OBSERVATIONS.md. Output json: runs/probe_benchmark.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet.probes import FiveStack, PatchCrops, WhiteboxDQ, collate_patches  # noqa: E402
from anet.probes.patches import _rects, rect_mask  # noqa: E402
from anet.data.rasterize import transform_boxes  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import pick_device, yolo_device  # noqa: E402


class MaskStats:
    def __init__(self):
        self.inter = self.union = self.bg_white = self.bg_px = 0.0
        self.n_obj = 0

    def add(self, pred, gt_mask, is_obj):
        if is_obj:
            self.inter += float((pred * gt_mask).sum())
            self.union += float(((pred + gt_mask) > 0).float().sum())
            self.n_obj += 1
        else:
            self.bg_white += float(pred.sum())
            self.bg_px += pred.numel()

    def summary(self):
        iou = self.inter / max(self.union, 1.0)
        bgw = self.bg_white / max(self.bg_px, 1.0)
        return {"mask_iou": iou, "bg_white": bgw, "n_obj": self.n_obj}


class ClassStats:
    def __init__(self):
        self.hit = {0: 0, 1: 0, 2: 0}
        self.tot = {0: 0, 1: 0, 2: 0}

    def add(self, pred, label):
        self.tot[label] += 1
        self.hit[label] += int(pred == label)

    def summary(self):
        r = {c: self.hit[c] / max(self.tot[c], 1) for c in (0, 1, 2)}
        return {"bg_acc": r[0], "mann_r": r[1], "tent_r": r[2]}


def eval_whitebox(ckpt, ds, device):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    arch = sd.get("arch", {}) if isinstance(sd, dict) else {}
    model = WhiteboxDQ(
        stages=int(arch.get("stages", os.environ.get("ANET_STAGES", "16"))),
        width=int(arch.get("width", os.environ.get("ANET_CH", "32"))),
        kernels=arch.get("kernels", os.environ.get("ANET_K", "7,11,15")),
    ).to(device)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    ms = MaskStats()
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_patches)
    with torch.no_grad():
        for batch in loader:
            for size, (imgs, masks, labels, _) in batch.items():
                p = (torch.sigmoid(model(imgs.to(device))[:, 0]) > 0.5).float().cpu()
                for j in range(len(labels)):
                    ms.add(p[j], masks[j], bool(labels[j] > 0))
    out = ms.summary()
    out["params"] = sum(p.numel() for p in model.parameters())
    return out, model


def eval_fivestack(ckpt, ds, device):
    model = FiveStack(embed=int(os.environ.get("ANET_CH", "16"))).to(device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    cs = ClassStats()
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_patches)
    with torch.no_grad():
        for batch in loader:
            for size, (imgs, _, labels, _) in batch.items():
                pred = model(imgs.to(device)).argmax(1).cpu()
                for j in range(len(labels)):
                    cs.add(int(pred[j]), int(labels[j]))
    out = cs.summary()
    out["params"] = sum(p.numel() for p in model.parameters())
    return out, model


def eval_yolo(weights, ds, cfg, conf=0.25):
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(weights)
    dev = yolo_device()
    ms, cs = MaskStats(), ClassStats()
    for k in range(len(ds)):
        path = ds.base.items[ds.idx[k]]
        res = model.predict(str(path), imgsz=cfg.yolo.imgsz, conf=conf,
                            device=dev, verbose=False)[0]
        rows = []
        if res.boxes is not None and len(res.boxes):
            xywhn = res.boxes.xywhn.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            for (cx, cy, w, h), c in zip(xywhn, cls):
                rows.append([c, cx, cy, w, h])
        w0, h0 = Image.open(path).size
        pred_rects = _rects(transform_boxes(
            np.asarray(rows, np.float32).reshape(-1, 5), w0, h0))
        for size, quads in ds[k].items():
            for crop, gt_mask, label, org in quads:
                ox, oy = int(org[0]), int(org[1])
                ms.add(rect_mask(pred_rects, ox, oy, size), gt_mask, label > 0)
                # patch label = class of the pred box overlapping most
                best_c, best_a = 0, 0.0
                for c, x0, y0, x1, y1 in pred_rects:
                    a = (max(0.0, min(x1, ox + size) - max(x0, ox))
                         * max(0.0, min(y1, oy + size) - max(y0, oy)))
                    if a > best_a:
                        best_c, best_a = c + 1, a
                cs.add(best_c, int(label))
        if (k + 1) % 200 == 0:
            print(f"  yolo {k + 1}/{len(ds)} frames", flush=True)
    out = {**ms.summary(), **cs.summary()}
    out["params"] = sum(p.numel() for p in model.model.parameters())
    return out


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def latency_probe(model, device, size=40, batch=256, n=30):
    x = torch.rand(batch, 3, size, size, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        _sync(device)
        t0 = time.time()
        for _ in range(n):
            model(x)
        _sync(device)
    return batch * n / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whitebox")
    ap.add_argument("--fivestack")
    ap.add_argument("--yolo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--out", default="runs/probe_benchmark.json")
    args = ap.parse_args()

    cfg = anet_cfg()
    device = pick_device()
    root = os.environ.get("DATA_ROOT", "../datasets/suas-synth-50k")
    ds = PatchCrops(root, args.split,
                    include_vd=os.environ.get("ANET_VD") == "1",
                    limit=args.limit)
    print(f"{args.split}: {len(ds)} frames, device={device}")

    results = {}
    if args.whitebox:
        results["whitebox"], m = eval_whitebox(args.whitebox, ds, device)
        if args.latency:
            results["whitebox"]["patches40_per_s"] = latency_probe(m, device)
    if args.fivestack:
        results["fivestack"], m = eval_fivestack(args.fivestack, ds, device)
        if args.latency:
            results["fivestack"]["patches40_per_s"] = latency_probe(m, device)
    if args.yolo:
        results["yolo"] = eval_yolo(args.yolo, ds, cfg, args.conf)

    keys = ["mask_iou", "bg_white", "bg_acc", "mann_r", "tent_r", "params"]
    if args.latency:
        keys.append("patches40_per_s")
    header = f"{'metric':20s}" + "".join(f"{n:>14s}" for n in results)
    print("\n" + header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:20s}"
        for name in results:
            v = results[name].get(k, float("nan"))
            row += f"{v:14,d}" if k == "params" else f"{v:14.3f}"
        print(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nfull results -> {args.out}")


if __name__ == "__main__":
    main()
