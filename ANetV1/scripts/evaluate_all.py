"""Three-way comparison on the test split: YOLO26n vs ANetV1 vs ANetV1-distilled.

YOLO boxes are rasterized through the identical cell pipeline so every number
is apples-to-apples. Slices: source (synthetic/VisDrone) + smallest-decile GT
boxes (worst-decile proxy — dataset exports no per-image GSD meta).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.data.dataset import SUASCells  # noqa: E402
from anet.data.rasterize import boxes_to_grid, transform_boxes  # noqa: E402
from anet.train.metrics import (CellConfusion, CenterObjectMetrics,  # noqa: E402
                                ObjectMetrics, confident_pred)
from anet.train.trainer import pick_device, yolo_device  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def latency_anet(ckpt, device, n=50, throughput_batch=16):
    """Median batch-1 latency (ms) + throughput (img/s) at a service batch size."""
    model = _load_anet(ckpt, device)
    out = {}
    with torch.no_grad():
        for b, key in ((1, "latency_ms_b1"), (throughput_batch, "throughput_img_s")):
            x = torch.rand(b, 3, 540, 960, device=device)
            for _ in range(10):
                model(x)
            _sync(device)
            t0 = time.time()
            for _ in range(n):
                model(x)
            _sync(device)
            dt = (time.time() - t0) / n
            out[key] = dt * 1000 if b == 1 else b / dt
    out["params"] = sum(p.numel() for p in model.parameters())
    return out


def latency_yolo(weights, ds, cfg, n=50):
    from ultralytics import YOLO

    model = YOLO(weights)
    img = str(ds.items[0])
    dev = yolo_device()
    for _ in range(10):
        model.predict(img, imgsz=cfg.yolo.imgsz, device=dev, verbose=False)
    t0 = time.time()
    for _ in range(n):
        model.predict(img, imgsz=cfg.yolo.imgsz, device=dev, verbose=False)
    dt = (time.time() - t0) / n
    return {"latency_ms_b1": dt * 1000, "throughput_img_s": 1.0 / dt,
            "params": sum(p.numel() for p in model.model.parameters())}


def _load_anet(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    model = ANetV1.from_state_dict(sd, use_checkpoint=False).to(device)  # width/stem from ckpt
    model.eval()
    return model


def eval_anet(ckpt, ds, device, conf_thresh=0.0, peak_thresh=0.3):
    model = _load_anet(ckpt, device)
    loader = DataLoader(ds, batch_size=4, num_workers=4)
    # every arch since v12 is a center-heatmap detector returning the
    # {"heat","offset"} dict; only the v8/v9 cell archs return cell logits.
    # Route by that contract, not an arch whitelist — the whitelist version
    # silently excluded v14/v15/v16 and crashed on 'dict has no .float()'.
    if model.arch not in ("v8", "v9"):
        # center-heatmap archs return {"heat","offset"} dicts, and their
        # object metrics come from peak-finding, not cell argmax. No cell
        # table — the summary keys mirror ObjectMetrics by design, so the
        # comparison rows print apples-to-apples and the cell rows show nan.
        obj_m = CenterObjectMetrics(peak_thresh=peak_thresh)
        with torch.no_grad():
            for batch in loader:
                out = model(batch["image"].to(device))
                heat = torch.sigmoid(out["heat"].float()).cpu()
                off = torch.sigmoid(out["offset"].float()).cpu()
                for i in range(heat.shape[0]):
                    obj_m.update(heat[i], off[i],
                                 batch["boxes"][i].numpy(), bool(batch["vd"][i]))
        return obj_m.summary()
    cells_m, obj_m = CellConfusion(), ObjectMetrics()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).float()
            pred = confident_pred(logits, conf_thresh).cpu().numpy()
            cells_m.update(pred, batch["grid"].numpy())
            for i in range(pred.shape[0]):
                obj_m.update(pred[i], batch["boxes"][i].numpy(), bool(batch["vd"][i]))
    out = obj_m.summary()
    out["cells"] = cells_m.summary()
    return out


def eval_yolo(weights, ds, cfg, conf=0.25):
    from ultralytics import YOLO

    model = YOLO(weights)
    cells_m, obj_m = CellConfusion(), ObjectMetrics()
    for i in range(len(ds)):
        path = ds.items[i]
        res = model.predict(str(path), imgsz=cfg.yolo.imgsz, conf=conf,
                            device=yolo_device(), verbose=False)[0]
        h0, w0 = res.orig_shape
        rows = []
        if res.boxes is not None and len(res.boxes):
            xywhn = res.boxes.xywhn.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            for (cx, cy, w, h), k in zip(xywhn, cls):
                rows.append([k, cx, cy, w, h])
        pred_boxes = transform_boxes(np.asarray(rows, np.float32).reshape(-1, 5), w0, h0)
        pred_grid = boxes_to_grid(pred_boxes, coverage_thresh=0.0)
        sample = ds[i]
        cells_m.update(pred_grid[None], sample["grid"].numpy()[None])
        obj_m.update(pred_grid, sample["boxes"].numpy(), sample["vd"])
    out = obj_m.summary()
    out["cells"] = cells_m.summary()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", help="trained YOLO weights .pt")
    ap.add_argument("--anet", help="ANetV1 checkpoint (runs/anet/best.pt)")
    ap.add_argument("--anet-distill", help="distilled checkpoint (runs/anet_distill/best.pt)")
    ap.add_argument("--latency", action="store_true",
                    help="also measure b1 latency and throughput per model")
    ap.add_argument("--split", default="test",
                    help="dataset split (train/val/test). --split train answers "
                         "the capacity-vs-data question: train recall ~= test "
                         "recall means the model UNDERFITS (capacity ceiling); "
                         "train >> test means a generalization gap (data lever)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images for the ANet eval (0 = all; the full train "
                         "split is ~14k images). YOLO eval ignores the cap.")
    ap.add_argument("--peak-thresh", type=float, default=None,
                    help="override the center-arch peak threshold (presets "
                         "0.3) — sweep it to map the recall-vs-fp operating "
                         "curve of a checkpoint")
    ap.add_argument("--out", default="runs/comparison.json")
    args = ap.parse_args()
    cfg = anet_cfg()

    ds = SUASCells(cfg.data.root, args.split,
                   coverage_thresh=cfg.data.coverage_thresh)
    ds_anet = ds
    if args.limit and args.limit < len(ds):
        from torch.utils.data import Subset
        ds_anet = Subset(ds, range(args.limit))
        print(f"ANet eval capped at first {args.limit}/{len(ds)} "
              f"{args.split} images")
    device = pick_device()
    results = {}
    if args.yolo:
        results["yolo"] = eval_yolo(args.yolo, ds, cfg)
        if args.latency:
            results["yolo"].update(latency_yolo(args.yolo, ds, cfg))
    if args.anet:
        results["anet"] = eval_anet(args.anet, ds_anet, device,
                                    getattr(cfg.train, "conf_thresh", 0.0),
                                    args.peak_thresh
                                    or getattr(cfg.train, "peak_thresh", 0.3))
        if args.latency:
            results["anet"].update(latency_anet(args.anet, device))
    if args.anet_distill:
        results["anet_distill"] = eval_anet(args.anet_distill, ds_anet, device,
                                            getattr(cfg.train, "conf_thresh", 0.0),
                                            args.peak_thresh
                                            or getattr(cfg.train, "peak_thresh", 0.3))
        if args.latency:
            results["anet_distill"].update(latency_anet(args.anet_distill, device))

    # hoist cell-level P/R/F1 so the table shows accuracy at both granularities
    for r in results.values():
        for cls in ("mannequin", "tent"):
            c = r.get("cells", {}).get(cls, {})
            for m in ("precision", "recall", "f1"):
                r[f"{cls}_cell_{m}"] = c.get(m, float("nan"))

    keys = ["mannequin_recall", "tent_recall", "fp_per_image",
            "mannequin_recall_synthetic", "mannequin_recall_visdrone",
            # D82: the pooled decile is 100% VisDrone at <=13.1 px^2 — read the
            # _synthetic one for anything mission-related.
            "mannequin_recall_smallest_decile_synthetic",
            "mannequin_recall_smallest_decile",
            "mannequin_cell_precision", "mannequin_cell_recall", "mannequin_cell_f1",
            "tent_cell_precision", "tent_cell_recall", "tent_cell_f1"]
    if args.latency:
        keys += ["latency_ms_b1", "throughput_img_s", "params"]
    header = f"{'metric':38s}" + "".join(f"{name:>14s}" for name in results)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:38s}"
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
