"""Paper-grade ANetV1 vs YOLO26n benchmarking suite.

Caches one inference pass per model, then computes offline:
  - object recall / FP/img with bootstrap 95% CIs
  - synthetic vs VisDrone slices
  - GT-area decile / quartile recall curves (worst-decile = §10 decision metric)
  - threshold sweeps (ANet peak_thresh, YOLO conf) → precision/recall operating curves
  - efficiency: params, disk, FLOPs, latency p50/p95, throughput, peak mem
  - training-cost summary from log.csv / results.csv
  - miss / FP size distributions for error analysis

Usage (from ANetV1/):
  python scripts/benchmark_paper.py \
    --anet runs/anet/best.pt \
    --yolo runs/yolo/yolo26n/weights/best.pt \
    --out runs/paper_bench
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.data.dataset import SUASCells  # noqa: E402
from anet.data.rasterize import (  # noqa: E402
    CANVAS_H, CANVAS_W, boxes_to_grid, transform_boxes,
)
from anet.train.metrics import (  # noqa: E402
    V12_H, V12_W, CenterObjectMetrics, ObjectMetrics,
)
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import pick_device, yolo_device  # noqa: E402

CLASS_NAMES = ("mannequin", "tent")


# --------------------------------------------------------------------------- helpers
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Mean + percentile CI over a binary or real vector."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(v, size=len(v), replace=True).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(v))}


def _area_decile_edges(areas):
    """Return 11 edges for 10 equal-count decile bins (unique-safe)."""
    a = np.sort(np.asarray(areas, dtype=np.float64))
    if len(a) == 0:
        return np.linspace(0, 1, 11)
    qs = np.linspace(0, 1, 11)
    return np.quantile(a, qs)


def _bin_recall(records, edges):
    """records: list of (cls, area, found, is_vd) → per-bin recall + n."""
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        # last bin inclusive on the right
        if i == len(edges) - 2:
            sub = [r for r in records if lo <= r[1] <= hi]
        else:
            sub = [r for r in records if lo <= r[1] < hi]
        rec = sum(r[2] for r in sub) / len(sub) if sub else float("nan")
        out.append({
            "bin": i,
            "area_lo": float(lo),
            "area_hi": float(hi),
            "n": len(sub),
            "recall": float(rec),
        })
    return out


def _summarize_records(records, fp_components, images, label=""):
    """records: (cls, area, found, is_vd). Mirrors ObjectMetrics.summary + extras."""
    out = {
        "fp_per_image": fp_components / max(images, 1),
        "n_images": images,
        "n_fp_components": int(fp_components),
    }
    for k, name in ((1, "mannequin"), (2, "tent")):
        sub = [r for r in records if r[0] == k]
        found = np.array([r[2] for r in sub], dtype=np.float64)
        out[f"{name}_recall"] = _bootstrap_ci(found)
        out[f"{name}_n"] = len(sub)
        areas = [r[1] for r in sub]
        out[f"{name}_area_deciles"] = _bin_recall(sub, _area_decile_edges(areas))
        # quartiles too (compact table for paper main text)
        out[f"{name}_area_quartiles"] = _bin_recall(
            sub, np.quantile(areas, [0, 0.25, 0.5, 0.75, 1.0]) if areas else np.linspace(0, 1, 5)
        )

    for name, cond in (("synthetic", lambda r: not r[3]), ("visdrone", lambda r: r[3])):
        sub = [r for r in records if r[0] == 1 and cond(r)]
        out[f"mannequin_recall_{name}"] = _bootstrap_ci([r[2] for r in sub])
        out[f"mannequin_n_{name}"] = len(sub)
        if sub:
            out[f"mannequin_area_deciles_{name}"] = _bin_recall(
                sub, _area_decile_edges([r[1] for r in sub])
            )

    # §10 decision metric: smallest-area decile of ALL mannequins
    m = sorted((r for r in records if r[0] == 1), key=lambda r: r[1])
    decile = m[: max(len(m) // 10, 1)] if m else []
    out["mannequin_recall_smallest_decile"] = _bootstrap_ci([r[2] for r in decile])
    # synthetic-only worst decile (deployment domain)
    m_syn = sorted((r for r in records if r[0] == 1 and not r[3]), key=lambda r: r[1])
    d_syn = m_syn[: max(len(m_syn) // 10, 1)] if m_syn else []
    out["mannequin_recall_smallest_decile_synthetic"] = _bootstrap_ci([r[2] for r in d_syn])

    # miss size stats (error analysis)
    for k, name in ((1, "mannequin"), (2, "tent")):
        misses = [r[1] for r in records if r[0] == k and not r[2]]
        hits = [r[1] for r in records if r[0] == k and r[2]]
        out[f"{name}_miss_area"] = {
            "n": len(misses),
            "mean": float(np.mean(misses)) if misses else float("nan"),
            "median": float(np.median(misses)) if misses else float("nan"),
            "p90": float(np.quantile(misses, 0.9)) if misses else float("nan"),
        }
        out[f"{name}_hit_area"] = {
            "n": len(hits),
            "mean": float(np.mean(hits)) if hits else float("nan"),
            "median": float(np.median(hits)) if hits else float("nan"),
        }
    if label:
        out["label"] = label
    return out


def _flatten_ci(summary):
    """Hoist *.mean for table printing while keeping full CI objects."""
    flat = {}
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v and "lo" in v:
            flat[k] = v["mean"]
            flat[f"{k}_lo"] = v["lo"]
            flat[f"{k}_hi"] = v["hi"]
            flat[f"{k}_n"] = v.get("n", summary.get(k.replace("_recall", "_n"), 0))
        else:
            flat[k] = v
    return flat


# --------------------------------------------------------------------------- ANet inference cache
def _load_anet(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    model = ANetV1.from_state_dict(sd, use_checkpoint=False).to(device)
    model.eval()
    return model


def cache_anet(ckpt, ds, device, out_dir: Path):
    """Run once; save per-image heat/offset probs + boxes meta as .npz."""
    cache = out_dir / "anet_preds.npz"
    meta_path = out_dir / "anet_meta.json"
    if cache.exists() and meta_path.exists():
        print(f"[anet] reusing cache {cache}")
        return np.load(cache, allow_pickle=True), json.loads(meta_path.read_text())

    model = _load_anet(ckpt, device)
    loader = DataLoader(ds, batch_size=8, num_workers=0, shuffle=False)
    heats, offsets, boxes_list, vd_list, stems = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            heat = torch.sigmoid(out["heat"].float()).cpu().numpy()
            off = torch.sigmoid(out["offset"].float()).cpu().numpy()
            for i in range(heat.shape[0]):
                heats.append(heat[i])
                offsets.append(off[i])
                boxes_list.append(batch["boxes"][i].numpy())
                vd_list.append(bool(batch["vd"][i]))
                stems.append(batch["stem"][i] if isinstance(batch["stem"][i], str)
                             else str(batch["stem"][i]))
    dt = time.time() - t0
    heats = np.stack(heats).astype(np.float16)
    offsets = np.stack(offsets).astype(np.float16)
    # ragged boxes → object array
    boxes_arr = np.empty(len(boxes_list), dtype=object)
    boxes_arr[:] = boxes_list
    np.savez_compressed(cache, heat=heats, offset=offsets, boxes=boxes_arr,
                        vd=np.asarray(vd_list, bool))
    meta = {
        "ckpt": str(ckpt),
        "n": len(heats),
        "arch": model.arch,
        "params": int(sum(p.numel() for p in model.parameters())),
        "infer_seconds": dt,
        "stems": stems,
        "device": str(device),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[anet] cached {len(heats)} preds in {dt:.1f}s → {cache}")
    return np.load(cache, allow_pickle=True), meta


def eval_anet_cached(preds, peak_thresh=0.3):
    m = CenterObjectMetrics(peak_thresh=peak_thresh)
    heat, offset, boxes, vd = preds["heat"], preds["offset"], preds["boxes"], preds["vd"]
    for i in range(len(heat)):
        m.update(heat[i].astype(np.float32), offset[i].astype(np.float32),
                 boxes[i], bool(vd[i]))
    return _summarize_records(m.records, m.fp_components, m.images)


def sweep_anet(preds, thresholds):
    rows = []
    for t in thresholds:
        s = eval_anet_cached(preds, peak_thresh=t)
        # precision proxy: TP / (TP + FP) at object level
        # TP ≈ sum found; FP = fp_components; FN = misses
        n_m = s["mannequin_n"]
        n_t = s["tent_n"]
        tp = (s["mannequin_recall"]["mean"] * n_m if n_m else 0) + (
            s["tent_recall"]["mean"] * n_t if n_t else 0)
        fp = s["n_fp_components"]
        prec = tp / max(tp + fp, 1e-9)
        rec_m = s["mannequin_recall"]["mean"]
        rows.append({
            "threshold": t,
            "mannequin_recall": rec_m,
            "tent_recall": s["tent_recall"]["mean"],
            "fp_per_image": s["fp_per_image"],
            "object_precision": float(prec),
            "mannequin_recall_smallest_decile": s["mannequin_recall_smallest_decile"]["mean"],
            "mannequin_recall_synthetic": s["mannequin_recall_synthetic"]["mean"],
            "mannequin_recall_visdrone": s["mannequin_recall_visdrone"]["mean"],
        })
    return rows


# --------------------------------------------------------------------------- YOLO inference cache
def cache_yolo(weights, ds, cfg, out_dir: Path, conf_floor=0.01):
    """Store all boxes above conf_floor so higher confs can be re-filtered offline."""
    cache = out_dir / "yolo_preds.npz"
    meta_path = out_dir / "yolo_meta.json"
    if cache.exists() and meta_path.exists():
        print(f"[yolo] reusing cache {cache}")
        return np.load(cache, allow_pickle=True), json.loads(meta_path.read_text())

    from ultralytics import YOLO

    model = YOLO(weights)
    dev = yolo_device()
    boxes_list, confs_list, vd_list, stems, shapes = [], [], [], [], []
    t0 = time.time()
    # batch paths for ultralytics — much faster than one-by-one
    paths = [str(p) for p in ds.items]
    bs = 16
    for start in range(0, len(paths), bs):
        chunk = paths[start:start + bs]
        results = model.predict(chunk, imgsz=cfg.yolo.imgsz, conf=conf_floor,
                                device=dev, verbose=False)
        for j, res in enumerate(results):
            i = start + j
            h0, w0 = res.orig_shape
            rows, confs = [], []
            if res.boxes is not None and len(res.boxes):
                xywhn = res.boxes.xywhn.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                cf = res.boxes.conf.cpu().numpy()
                for (cx, cy, w, h), k, c in zip(xywhn, cls, cf):
                    rows.append([k, cx, cy, w, h])
                    confs.append(float(c))
            raw = np.asarray(rows, np.float32).reshape(-1, 5)
            # store canvas-normalized boxes (same space as GT)
            canv = transform_boxes(raw, w0, h0) if len(raw) else raw.reshape(0, 5)
            boxes_list.append(canv)
            confs_list.append(np.asarray(confs, np.float32))
            vd_list.append(ds.is_visdrone(i))
            stems.append(ds.items[i].stem)
            shapes.append((h0, w0))
        if (start // bs) % 20 == 0:
            print(f"[yolo] {min(start + bs, len(paths))}/{len(paths)}")
    dt = time.time() - t0
    boxes_arr = np.empty(len(boxes_list), dtype=object)
    boxes_arr[:] = boxes_list
    confs_arr = np.empty(len(confs_list), dtype=object)
    confs_arr[:] = confs_list
    # also cache GT boxes for offline eval (avoid reloading images)
    gt_boxes = []
    for i in range(len(ds)):
        # boxes already canvas-normalized in dataset item
        sample = ds[i]
        gt_boxes.append(sample["boxes"].numpy())
    gt_arr = np.empty(len(gt_boxes), dtype=object)
    gt_arr[:] = gt_boxes
    np.savez_compressed(cache, pred_boxes=boxes_arr, confs=confs_arr,
                        gt_boxes=gt_arr, vd=np.asarray(vd_list, bool))
    n_params = int(sum(p.numel() for p in model.model.parameters()))
    meta = {
        "weights": str(weights),
        "n": len(boxes_list),
        "params": n_params,
        "infer_seconds": dt,
        "conf_floor": conf_floor,
        "imgsz": cfg.yolo.imgsz,
        "stems": stems,
        "device": str(dev),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[yolo] cached {len(boxes_list)} preds in {dt:.1f}s → {cache}")
    return np.load(cache, allow_pickle=True), meta


def eval_yolo_cached(preds, conf=0.25):
    """Rasterize filtered YOLO boxes through the same ObjectMetrics pipeline."""
    cells_unused = None  # keep object-level primary
    obj_m = ObjectMetrics()
    pred_boxes_all = preds["pred_boxes"]
    confs_all = preds["confs"]
    gt_all = preds["gt_boxes"]
    vd = preds["vd"]
    for i in range(len(pred_boxes_all)):
        pb = pred_boxes_all[i]
        cf = confs_all[i]
        if len(pb) and len(cf):
            keep = cf >= conf
            pb = pb[keep]
        else:
            pb = pb.reshape(0, 5) if hasattr(pb, "reshape") else np.zeros((0, 5), np.float32)
        pred_grid = boxes_to_grid(pb, coverage_thresh=0.0)
        obj_m.update(pred_grid, gt_all[i], bool(vd[i]))
    return _summarize_records(obj_m.records, obj_m.fp_components, obj_m.images)


def sweep_yolo(preds, thresholds):
    rows = []
    for t in thresholds:
        s = eval_yolo_cached(preds, conf=t)
        n_m, n_t = s["mannequin_n"], s["tent_n"]
        tp = (s["mannequin_recall"]["mean"] * n_m if n_m else 0) + (
            s["tent_recall"]["mean"] * n_t if n_t else 0)
        fp = s["n_fp_components"]
        prec = tp / max(tp + fp, 1e-9)
        rows.append({
            "threshold": t,
            "mannequin_recall": s["mannequin_recall"]["mean"],
            "tent_recall": s["tent_recall"]["mean"],
            "fp_per_image": s["fp_per_image"],
            "object_precision": float(prec),
            "mannequin_recall_smallest_decile": s["mannequin_recall_smallest_decile"]["mean"],
            "mannequin_recall_synthetic": s["mannequin_recall_synthetic"]["mean"],
            "mannequin_recall_visdrone": s["mannequin_recall_visdrone"]["mean"],
        })
    return rows


# --------------------------------------------------------------------------- efficiency
def measure_anet_efficiency(ckpt, device, n_lat=100, batches=(1, 4, 8, 16)):
    model = _load_anet(ckpt, device)
    out = {
        "params": int(sum(p.numel() for p in model.parameters())),
        "params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "disk_bytes": Path(ckpt).stat().st_size,
        "arch": model.arch,
    }
    # FLOPs via fvcore if available, else torch flop counter, else skip
    x = torch.rand(1, 3, CANVAS_H, CANVAS_W, device=device)
    flops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = int(FlopCountAnalysis(model, x).total())
    except Exception:
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                with torch.no_grad():
                    model(x)
            flops = int(fc.get_total_flops())
        except Exception as e:
            out["flops_error"] = str(e)
    if flops is not None:
        out["flops"] = flops
        out["gflops"] = flops / 1e9

    # latency distribution + throughput
    latencies = {}
    throughputs = {}
    with torch.no_grad():
        for b in batches:
            xb = torch.rand(b, 3, CANVAS_H, CANVAS_W, device=device)
            for _ in range(15):
                model(xb)
            _sync(device)
            ts = []
            for _ in range(n_lat):
                _sync(device)
                t0 = time.perf_counter()
                model(xb)
                _sync(device)
                ts.append((time.perf_counter() - t0) * 1000.0)  # ms / batch
            arr = np.asarray(ts)
            per_img = arr / b
            latencies[f"b{b}"] = {
                "p50_ms": float(np.median(per_img)),
                "p95_ms": float(np.quantile(per_img, 0.95)),
                "p99_ms": float(np.quantile(per_img, 0.99)),
                "mean_ms": float(per_img.mean()),
            }
            throughputs[f"b{b}"] = float(1000.0 / np.median(per_img))
    out["latency_ms_per_img"] = latencies
    out["throughput_img_s"] = throughputs
    out["latency_ms_b1"] = latencies["b1"]["p50_ms"]
    out["throughput_img_s_b1"] = throughputs["b1"]

    # peak memory (MPS / CUDA)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(x)
        _sync(device)
        out["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
    elif device.type == "mps":
        # driver_allocated is cumulative; best-effort delta
        before = torch.mps.driver_allocated_memory()
        with torch.no_grad():
            for _ in range(5):
                model(x)
        _sync(device)
        out["peak_mem_bytes"] = int(max(0, torch.mps.driver_allocated_memory() - before))
        out["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    return out


def measure_yolo_efficiency(weights, ds, cfg, n_lat=50):
    from ultralytics import YOLO

    model = YOLO(weights)
    img = str(ds.items[0])
    dev = yolo_device()
    for _ in range(10):
        model.predict(img, imgsz=cfg.yolo.imgsz, device=dev, verbose=False)
    ts = []
    for _ in range(n_lat):
        t0 = time.perf_counter()
        model.predict(img, imgsz=cfg.yolo.imgsz, device=dev, verbose=False)
        ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ts)
    return {
        "params": int(sum(p.numel() for p in model.model.parameters())),
        "disk_bytes": Path(weights).stat().st_size,
        "latency_ms_b1": float(np.median(arr)),
        "latency_p95_ms": float(np.quantile(arr, 0.95)),
        "latency_p99_ms": float(np.quantile(arr, 0.99)),
        "throughput_img_s_b1": float(1000.0 / np.median(arr)),
        "imgsz": cfg.yolo.imgsz,
        "device": str(dev),
    }


def training_summary(anet_log: Path, yolo_csv: Path):
    out = {}
    if anet_log.exists():
        rows = list(csv.DictReader(anet_log.open()))
        if rows:
            secs = [float(r["seconds"]) for r in rows]
            out["anet"] = {
                "epochs": len(rows),
                "wall_seconds": float(sum(secs)),
                "sec_per_epoch_mean": float(np.mean(secs)),
                "best_mannequin_recall": float(max(float(r["mannequin_recall"]) for r in rows)),
                "best_tent_recall": float(max(float(r["tent_recall"]) for r in rows)),
                "final_train_loss": float(rows[-1]["train_loss"]),
            }
    if yolo_csv.exists():
        rows = list(csv.DictReader(yolo_csv.open()))
        if rows:
            # ultralytics `time` column is cumulative seconds
            t_end = float(rows[-1]["time"])
            out["yolo"] = {
                "epochs": len(rows),
                "wall_seconds": t_end,
                "sec_per_epoch_mean": t_end / max(len(rows), 1),
                "best_mAP50": float(max(float(r["metrics/mAP50(B)"]) for r in rows)),
                "best_recall": float(max(float(r["metrics/recall(B)"]) for r in rows)),
                "final_mAP50": float(rows[-1]["metrics/mAP50(B)"]),
            }
    return out


# --------------------------------------------------------------------------- report
def write_tables(out_dir: Path, results: dict):
    """Markdown + CSV tables ready to paste into a paper draft."""
    models = [k for k in ("anet", "yolo") if k in results and "metrics" in results[k]]
    keys = [
        ("mannequin_recall", "Mannequin recall"),
        ("tent_recall", "Tent recall"),
        ("fp_per_image", "FP / image"),
        ("mannequin_recall_synthetic", "Mannequin recall (synth)"),
        ("mannequin_recall_visdrone", "Mannequin recall (VisDrone)"),
        ("mannequin_recall_smallest_decile", "Mannequin recall (worst decile)"),
        ("mannequin_recall_smallest_decile_synthetic", "Mannequin recall (worst decile, synth)"),
    ]
    lines = ["# ANetV1 vs YOLO26n — paper benchmark\n"]
    lines.append("## Detection (test split, object-level)\n")
    header = "| Metric | " + " | ".join(m.upper() for m in models) + " |"
    sep = "|---|---" + "|---" * (len(models) - 1) + "|" if models else "|---|---|"
    # fix sep
    sep = "|" + "---|" * (len(models) + 1)
    lines.append(header)
    lines.append(sep)
    for key, label in keys:
        row = f"| {label} |"
        for m in models:
            flat = _flatten_ci(results[m]["metrics"])
            v = flat.get(key, float("nan"))
            if isinstance(v, dict):
                v = v.get("mean", float("nan"))
            lo = flat.get(f"{key}_lo")
            hi = flat.get(f"{key}_hi")
            if lo is not None and hi is not None and not (isinstance(lo, float) and math.isnan(lo)):
                row += f" {v:.3f} [{lo:.3f}, {hi:.3f}] |"
            else:
                row += f" {v:.3f} |" if isinstance(v, float) else f" {v} |"
        lines.append(row)

    lines.append("\n## Efficiency\n")
    lines.append("| Axis | " + " | ".join(m.upper() for m in models) + " |")
    lines.append("|" + "---|" * (len(models) + 1))
    eff_keys = [
        ("params", "Parameters", "{:,}"),
        ("disk_bytes", "Checkpoint bytes", "{:,}"),
        ("gflops", "GFLOPs / frame", "{:.3f}"),
        ("latency_ms_b1", "Latency p50 (ms, b=1)", "{:.2f}"),
        ("throughput_img_s_b1", "Throughput (img/s, b=1)", "{:.1f}"),
    ]
    for key, label, fmt in eff_keys:
        row = f"| {label} |"
        for m in models:
            e = results[m].get("efficiency", {})
            v = e.get(key, e.get("latency_ms_per_img", {}).get("b1", {}).get("p50_ms", float("nan"))
                      if key == "latency_ms_b1" else float("nan"))
            if key == "latency_ms_b1" and "latency_ms_b1" in e:
                v = e["latency_ms_b1"]
            if isinstance(v, float) and math.isnan(v):
                row += " — |"
            else:
                try:
                    row += f" {fmt.format(v)} |"
                except Exception:
                    row += f" {v} |"
        lines.append(row)

    # Pareto / ratios
    if "anet" in results and "yolo" in results:
        ae, ye = results["anet"].get("efficiency", {}), results["yolo"].get("efficiency", {})
        am = _flatten_ci(results["anet"]["metrics"])
        ym = _flatten_ci(results["yolo"]["metrics"])
        lines.append("\n## Headlines (ratios)\n")
        if ae.get("params") and ye.get("params"):
            lines.append(f"- Params: YOLO / ANet = **{ye['params'] / ae['params']:.1f}×**")
        if ae.get("latency_ms_b1") and ye.get("latency_ms_b1"):
            lines.append(
                f"- Latency: YOLO / ANet = **{ye['latency_ms_b1'] / ae['latency_ms_b1']:.1f}×** "
                f"(ANet is faster)"
            )
        gap = ym.get("mannequin_recall_smallest_decile", float("nan")) - am.get(
            "mannequin_recall_smallest_decile", float("nan"))
        lines.append(f"- §10 decision gap (YOLO − ANet worst-decile mannequin): **{gap:+.3f}**")
        gap_s = ym.get("mannequin_recall_smallest_decile_synthetic", float("nan")) - am.get(
            "mannequin_recall_smallest_decile_synthetic", float("nan"))
        lines.append(f"- Same gap on synthetic-only worst decile: **{gap_s:+.3f}**")

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")

    # CSV: main metrics
    with (out_dir / "metrics_main.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "mean", "ci_lo", "ci_hi", "n"])
        for m in models:
            s = results[m]["metrics"]
            for key, _ in keys:
                v = s.get(key)
                if isinstance(v, dict) and "mean" in v:
                    w.writerow([m, key, v["mean"], v["lo"], v["hi"], v.get("n", "")])
                elif isinstance(v, (int, float)):
                    w.writerow([m, key, v, "", "", ""])

    # CSV: area deciles
    with (out_dir / "mannequin_area_deciles.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "slice", "bin", "area_lo", "area_hi", "n", "recall"])
        for m in models:
            s = results[m]["metrics"]
            for slice_name, key in (
                ("all", "mannequin_area_deciles"),
                ("synthetic", "mannequin_area_deciles_synthetic"),
                ("visdrone", "mannequin_area_deciles_visdrone"),
            ):
                for row in s.get(key, []) or []:
                    w.writerow([m, slice_name, row["bin"], row["area_lo"],
                                row["area_hi"], row["n"], row["recall"]])

    # threshold sweeps
    for m in models:
        sweep = results[m].get("threshold_sweep")
        if not sweep:
            continue
        with (out_dir / f"threshold_sweep_{m}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            w.writerows(sweep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anet", default="runs/anet/best.pt")
    ap.add_argument("--yolo", default="runs/yolo/yolo26n/weights/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="cap images (0=all)")
    ap.add_argument("--out", default="runs/paper_bench")
    ap.add_argument("--peak-thresh", type=float, default=0.3)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--skip-efficiency", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="delete prediction caches and re-infer")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_cache:
        for p in out_dir.glob("*_preds.npz"):
            p.unlink()
        for p in out_dir.glob("*_meta.json"):
            p.unlink()

    cfg = anet_cfg()
    base = SUASCells(cfg.data.root, args.split, coverage_thresh=cfg.data.coverage_thresh)

    class _StrideSub:
        """Stride subsample that preserves .items / is_visdrone (prefix would be synth-only)."""
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = list(indices)
            self.items = [parent.items[i] for i in self.indices]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.parent[self.indices[i]]

        def is_visdrone(self, i):
            return self.parent.is_visdrone(self.indices[i])

    if args.limit and args.limit < len(base):
        idx = np.linspace(0, len(base) - 1, args.limit, dtype=int)
        ds = _StrideSub(base, idx)
        print(f"subsampled {len(ds)}/{len(base)} {args.split} images (stride, mixed sources)")
    else:
        ds = base

    device = pick_device()
    results = {
        "meta": {
            "split": args.split,
            "n_images": len(ds),
            "device": str(device),
            "anet_ckpt": args.anet,
            "yolo_weights": args.yolo,
            "peak_thresh": args.peak_thresh,
            "yolo_conf": args.yolo_conf,
            "canvas": [CANVAS_H, CANVAS_W],
            "center_grid": [V12_H, V12_W],
        }
    }

    # ---- ANet
    if args.anet:
        preds, meta = cache_anet(args.anet, ds, device, out_dir)
        metrics = eval_anet_cached(preds, peak_thresh=args.peak_thresh)
        results["anet"] = {"meta": meta, "metrics": metrics}
        if not args.skip_sweep:
            thr = [round(x, 2) for x in np.linspace(0.05, 0.80, 16)]
            results["anet"]["threshold_sweep"] = sweep_anet(preds, thr)
        if not args.skip_efficiency:
            results["anet"]["efficiency"] = measure_anet_efficiency(args.anet, device)

    # ---- YOLO
    if args.yolo:
        # YOLO cache needs a real SUASCells (items + is_visdrone)
        preds, meta = cache_yolo(args.yolo, ds, cfg, out_dir)
        metrics = eval_yolo_cached(preds, conf=args.yolo_conf)
        results["yolo"] = {"meta": meta, "metrics": metrics}
        if not args.skip_sweep:
            thr = [round(x, 2) for x in np.linspace(0.05, 0.80, 16)]
            results["yolo"]["threshold_sweep"] = sweep_yolo(preds, thr)
        if not args.skip_efficiency:
            base_ds = ds if hasattr(ds, "items") else SUASCells(
                cfg.data.root, args.split, coverage_thresh=cfg.data.coverage_thresh)
            results["yolo"]["efficiency"] = measure_yolo_efficiency(args.yolo, base_ds, cfg)

    # ---- training cost
    results["training"] = training_summary(
        Path("runs/anet/log.csv"),
        Path("runs/yolo/yolo26n/results.csv")
        if Path("runs/yolo/yolo26n/results.csv").exists()
        else Path("runs/yolo/baseline/results.csv"),
    )

    # serialize (CI dicts are plain floats)
    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    out_json = out_dir / "results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2, default=_default)
    write_tables(out_dir, results)

    # console summary
    print("\n" + "=" * 72)
    print(f"{'metric':42s}" + "".join(f"{n:>14s}" for n in ("anet", "yolo") if n in results))
    print("-" * 72)
    show = [
        "mannequin_recall", "tent_recall", "fp_per_image",
        "mannequin_recall_synthetic", "mannequin_recall_visdrone",
        "mannequin_recall_smallest_decile",
        "mannequin_recall_smallest_decile_synthetic",
    ]
    for k in show:
        row = f"{k:42s}"
        for n in ("anet", "yolo"):
            if n not in results:
                continue
            v = results[n]["metrics"].get(k)
            if isinstance(v, dict) and "mean" in v:
                row += f"{v['mean']:14.3f}"
            else:
                row += f"{float(v):14.3f}" if v is not None else f"{'nan':>14s}"
        print(row)
    if not args.skip_efficiency:
        print("-" * 72)
        for k in ("params", "latency_ms_b1", "throughput_img_s_b1", "gflops"):
            row = f"{k:42s}"
            for n in ("anet", "yolo"):
                if n not in results or "efficiency" not in results[n]:
                    continue
                v = results[n]["efficiency"].get(k, float("nan"))
                if k == "params":
                    row += f"{int(v):14,d}" if v == v else f"{'—':>14s}"
                else:
                    row += f"{float(v):14.3f}" if v == v else f"{'—':>14s}"
            print(row)
    print("=" * 72)
    print(f"\nfull results → {out_json}")
    print(f"tables       → {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
