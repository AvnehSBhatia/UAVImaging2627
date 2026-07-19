"""Three-way miss analysis: v13 vs v22 vs YOLO26n on synthetic test.

Caches v13 preds next to the existing v22/YOLO caches, then offline:
  - per-GT-object found flags (class-matched centre-in-box for all three)
  - contingency tables (who finds what the others miss)
  - area / heat-at-GT-center slices for v22-only misses
  - miss gallery overlays under runs/paper_bench_tri/miss_gallery/

  cd ANetV1
  python scripts/analyze_v13_v22_misses.py \
      --v13 runs/anet/v13_best.pt --v22 runs/anet/best.pt \
      --yolo-cache runs/paper_bench_v22 --out runs/paper_bench_tri
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet.data.dataset import SUASCells  # noqa: E402
from anet.data.rasterize import CANVAS_H, CANVAS_W  # noqa: E402
from anet.train.metrics import CenterObjectMetrics, V12_H, V12_W  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

import importlib.util  # noqa: E402

_bp = Path(__file__).resolve().parent / "benchmark_paper.py"
_spec = importlib.util.spec_from_file_location("benchmark_paper", _bp)
_bp_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bp_mod)
cache_anet = _bp_mod.cache_anet
eval_anet_cached = _bp_mod.eval_anet_cached
eval_yolo_cached = _bp_mod.eval_yolo_cached
measure_anet_efficiency = _bp_mod.measure_anet_efficiency
measure_yolo_efficiency = _bp_mod.measure_yolo_efficiency


def peaks(heat_c, off, thresh=0.3):
    rows, cols = CenterObjectMetrics._find_peaks(heat_c, thresh)
    if len(rows) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    dx, dy = off[0, rows, cols], off[1, rows, cols]
    cx = (cols + dx) / V12_W
    cy = (rows + dy) / V12_H
    p = heat_c[rows, cols]
    return cx, cy, p


def anet_found(heat, off, box, thresh=0.3):
    """Same-class peak centre inside GT box. Returns (found, max_heat_in_box, max_peak_p)."""
    c = int(box[0])
    bx, by, bw, bh = map(float, box[1:])
    cx, cy, p = peaks(heat[c], off, thresh)
    inside = ((cx >= bx - bw / 2) & (cx <= bx + bw / 2)
              & (cy >= by - bh / 2) & (cy <= by + bh / 2))
    # heat at GT centre cell (dilution diagnostic)
    r = min(max(int(by * V12_H), 0), V12_H - 1)
    cc = min(max(int(bx * V12_W), 0), V12_W - 1)
    h_center = float(heat[c, r, cc])
    # max heat in box footprint
    r0 = max(int((by - bh / 2) * V12_H), 0)
    r1 = min(int(np.ceil((by + bh / 2) * V12_H)), V12_H)
    c0 = max(int((bx - bw / 2) * V12_W), 0)
    c1 = min(int(np.ceil((bx + bw / 2) * V12_W)), V12_W)
    h_box = float(heat[c, r0:r1, c0:c1].max()) if r1 > r0 and c1 > c0 else h_center
    max_p = float(p[inside].max()) if inside.any() else 0.0
    return bool(inside.any()), h_center, h_box, max_p


def yolo_found(pred_boxes, confs, box, conf=0.25):
    """Same-class pred whose centre lands in the GT box."""
    c = int(box[0])
    bx, by, bw, bh = map(float, box[1:])
    if len(pred_boxes) == 0:
        return False, 0.0
    keep = confs >= conf if len(confs) else np.zeros(0, bool)
    pb = pred_boxes[keep] if len(keep) else pred_boxes.reshape(0, 5)
    best = 0.0
    hit = False
    for row in pb:
        if int(row[0]) != c:
            continue
        px, py = float(row[1]), float(row[2])
        if abs(px - bx) <= bw / 2 and abs(py - by) <= bh / 2:
            hit = True
            # conf of this box
            # find matching conf — approximate by scanning kept
        # IoU fallback for large YOLO boxes whose centre drifts
        # (centre-in-box is primary; IoU>0.1 as secondary)
        x0 = max(bx - bw / 2, row[1] - row[3] / 2)
        y0 = max(by - bh / 2, row[2] - row[4] / 2)
        x1 = min(bx + bw / 2, row[1] + row[3] / 2)
        y1 = min(by + bh / 2, row[2] + row[4] / 2)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = bw * bh + row[3] * row[4] - inter
        iou = inter / max(union, 1e-9)
        if abs(px - bx) <= bw / 2 and abs(py - by) <= bh / 2:
            hit = True
        elif iou >= 0.1:
            hit = True
        best = max(best, iou)
    return hit, best


def draw_gallery(rgb_path, gt_box, stem, tag, out_path, notes):
    im = Image.open(rgb_path).convert("RGB").resize((CANVAS_W, CANVAS_H))
    d = ImageDraw.Draw(im)
    c, cx, cy, w, h = gt_box
    x0 = (cx - w / 2) * CANVAS_W
    y0 = (cy - h / 2) * CANVAS_H
    x1 = (cx + w / 2) * CANVAS_W
    y1 = (cy + h / 2) * CANVAS_H
    d.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=3)
    d.text((4, 4), f"{stem} {tag} cls={int(c)} area={w*h:.5f}", fill=(255, 255, 0))
    d.text((4, 20), notes[:120], fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v13", default="runs/anet/v13_best.pt")
    ap.add_argument("--v22", default="runs/anet/best.pt")
    ap.add_argument("--yolo-cache", default="runs/paper_bench_v22",
                    help="dir with yolo_preds.npz (+ optional v22 anet_preds)")
    ap.add_argument("--out", default="runs/paper_bench_tri")
    ap.add_argument("--split", default="test")
    ap.add_argument("--peak-thresh", type=float, default=0.3)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--n-gallery", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.yolo_cache)
    # reuse YOLO (+ v22 if present) caches
    for name in ("yolo_preds.npz", "yolo_meta.json"):
        if (src / name).exists() and not (out / name).exists():
            shutil.copy2(src / name, out / name)
    # v22 preds: copy as v22_preds.npz; also keep anet_preds as v22 for clarity
    if (src / "anet_preds.npz").exists():
        if not (out / "v22_preds.npz").exists():
            shutil.copy2(src / "anet_preds.npz", out / "v22_preds.npz")
            shutil.copy2(src / "anet_meta.json", out / "v22_meta.json")

    cfg = anet_cfg()
    device = pick_device()
    ds = SUASCells(cfg.data.root, args.split, coverage_thresh=cfg.data.coverage_thresh)

    # cache v13 under out/v13_*
    v13_dir = out / "_v13_cache"
    v13_dir.mkdir(exist_ok=True)
    if not (out / "v13_preds.npz").exists():
        preds13, meta13 = cache_anet(args.v13, ds, device, v13_dir)
        shutil.copy2(v13_dir / "anet_preds.npz", out / "v13_preds.npz")
        shutil.copy2(v13_dir / "anet_meta.json", out / "v13_meta.json")
    else:
        preds13 = np.load(out / "v13_preds.npz", allow_pickle=True)
        meta13 = json.loads((out / "v13_meta.json").read_text())

    if not (out / "v22_preds.npz").exists():
        preds22, meta22 = cache_anet(args.v22, ds, device, out)
        # cache_anet writes anet_preds.npz
        shutil.move(str(out / "anet_preds.npz"), str(out / "v22_preds.npz"))
        shutil.move(str(out / "anet_meta.json"), str(out / "v22_meta.json"))
    else:
        preds22 = np.load(out / "v22_preds.npz", allow_pickle=True)
        meta22 = json.loads((out / "v22_meta.json").read_text())

    yolo = np.load(out / "yolo_preds.npz", allow_pickle=True)
    yolo_meta = json.loads((out / "yolo_meta.json").read_text())

    # aggregate metrics (standard paper tables)
    m13 = eval_anet_cached(preds13, peak_thresh=args.peak_thresh)
    m22 = eval_anet_cached(preds22, peak_thresh=args.peak_thresh)
    my = eval_yolo_cached(yolo, conf=args.yolo_conf)

    # per-object rows on synthetic only
    rows = []
    stems = yolo_meta["stems"]
    for i in range(len(stems)):
        if bool(yolo["vd"][i]):
            continue
        gt = yolo["gt_boxes"][i]
        h13, o13 = preds13["heat"][i].astype(np.float32), preds13["offset"][i].astype(np.float32)
        h22, o22 = preds22["heat"][i].astype(np.float32), preds22["offset"][i].astype(np.float32)
        pb, cf = yolo["pred_boxes"][i], yolo["confs"][i]
        for box in gt:
            if box[0] < 0:
                continue
            f13, hc13, hb13, p13 = anet_found(h13, o13, box, args.peak_thresh)
            f22, hc22, hb22, p22 = anet_found(h22, o22, box, args.peak_thresh)
            fy, iou = yolo_found(pb, cf, box, args.yolo_conf)
            area = float(box[3] * box[4])
            rows.append({
                "stem": stems[i], "img_i": i, "cls": int(box[0]),
                "area": area, "box": [float(x) for x in box],
                "v13": f13, "v22": f22, "yolo": fy,
                "h_center_v13": hc13, "h_box_v13": hb13, "peak_v13": p13,
                "h_center_v22": hc22, "h_box_v22": hb22, "peak_v22": p22,
                "yolo_iou": iou,
            })

    def filt(pred):
        return [r for r in rows if pred(r)]

    mann = [r for r in rows if r["cls"] == 0]
    tent = [r for r in rows if r["cls"] == 1]

    def recall(subset, key):
        return sum(r[key] for r in subset) / max(len(subset), 1)

    # contingencies on synthetic mannequins
    yolo_hit_v22_miss = filt(lambda r: r["cls"] == 0 and r["yolo"] and not r["v22"])
    v13_hit_v22_miss = filt(lambda r: r["cls"] == 0 and r["v13"] and not r["v22"])
    v22_hit_v13_miss = filt(lambda r: r["cls"] == 0 and r["v22"] and not r["v13"])
    all_miss = filt(lambda r: r["cls"] == 0 and not r["v13"] and not r["v22"] and not r["yolo"])
    yolo_only = filt(lambda r: r["cls"] == 0 and r["yolo"] and not r["v13"] and not r["v22"])
    both_anet_miss_yolo_hit = filt(
        lambda r: r["cls"] == 0 and r["yolo"] and not r["v13"] and not r["v22"])

    # heat dilution on YOLO-hit / v22-miss
    dil = yolo_hit_v22_miss
    heat_stats = {
        "n": len(dil),
        "h_box_v22_mean": float(np.mean([r["h_box_v22"] for r in dil])) if dil else None,
        "h_box_v22_median": float(np.median([r["h_box_v22"] for r in dil])) if dil else None,
        "h_center_v22_mean": float(np.mean([r["h_center_v22"] for r in dil])) if dil else None,
        "frac_h_box_below_thresh": (
            sum(r["h_box_v22"] <= args.peak_thresh for r in dil) / max(len(dil), 1)),
        "frac_h_box_in_02_03": (
            sum(0.2 <= r["h_box_v22"] < 0.3 for r in dil) / max(len(dil), 1)),
        "area_median": float(np.median([r["area"] for r in dil])) if dil else None,
        "area_p90": float(np.quantile([r["area"] for r in dil], 0.9)) if dil else None,
    }

    # area deciles of synth mannequin: recall per model
    areas = sorted(r["area"] for r in mann)
    edges = np.quantile(areas, np.linspace(0, 1, 11)) if areas else np.linspace(0, 1, 11)
    decile_rows = []
    for d in range(10):
        lo, hi = float(edges[d]), float(edges[d + 1])
        sub = [r for r in mann if (lo <= r["area"] <= hi if d == 9 else lo <= r["area"] < hi)]
        decile_rows.append({
            "decile": d, "lo": lo, "hi": hi, "n": len(sub),
            "v13": recall(sub, "v13"), "v22": recall(sub, "v22"), "yolo": recall(sub, "yolo"),
        })

    # efficiency (v13 + reuse v22/yolo if present)
    eff = {}
    eff["v13"] = measure_anet_efficiency(args.v13, device)
    if (src / "results.json").exists():
        prev = json.loads((src / "results.json").read_text())
        eff["v22"] = prev.get("anet", {}).get("efficiency") or measure_anet_efficiency(
            args.v22, device)
        eff["yolo"] = prev.get("yolo", {}).get("efficiency") or measure_yolo_efficiency(
            "runs/yolo/yolo26n/weights/best.pt", ds, cfg)
    else:
        eff["v22"] = measure_anet_efficiency(args.v22, device)
        eff["yolo"] = measure_yolo_efficiency(
            "runs/yolo/yolo26n/weights/best.pt", ds, cfg)

    # gallery: YOLO-hit v22-miss, smallest first
    gal = out / "miss_gallery"
    gal.mkdir(exist_ok=True)
    ranked = sorted(yolo_hit_v22_miss, key=lambda r: r["area"])
    for k, r in enumerate(ranked[: args.n_gallery]):
        path = ds.items[r["img_i"]]
        tag = "yolo_hit_v22_miss"
        if r["v13"]:
            tag += "_v13_hit"
        else:
            tag += "_v13_miss"
        notes = (f"h_box_v22={r['h_box_v22']:.3f} h_box_v13={r['h_box_v13']:.3f} "
                 f"v13={'Y' if r['v13'] else 'N'}")
        draw_gallery(path, r["box"], r["stem"], tag,
                     gal / f"{k:02d}_{r['stem']}_{tag}.png", notes)

    # also v13-hit v22-miss (regression from donor)
    ranked2 = sorted(v13_hit_v22_miss, key=lambda r: r["area"])
    for k, r in enumerate(ranked2[: min(12, args.n_gallery)]):
        path = ds.items[r["img_i"]]
        notes = f"REGRESSION h_box_v22={r['h_box_v22']:.3f} h_box_v13={r['h_box_v13']:.3f}"
        draw_gallery(path, r["box"], r["stem"], "v13_hit_v22_miss",
                     gal / f"reg_{k:02d}_{r['stem']}.png", notes)

    summary = {
        "meta": {
            "split": args.split, "synth_only_objects": True,
            "peak_thresh": args.peak_thresh, "yolo_conf": args.yolo_conf,
            "n_synth_mannequin": len(mann), "n_synth_tent": len(tent),
            "v13_ckpt": args.v13, "v22_ckpt": args.v22,
        },
        "recall_synth": {
            "mannequin": {
                "v13": recall(mann, "v13"),
                "v22": recall(mann, "v22"),
                "yolo": recall(mann, "yolo"),
                "n": len(mann),
            },
            "tent": {
                "v13": recall(tent, "v13"),
                "v22": recall(tent, "v22"),
                "yolo": recall(tent, "yolo"),
                "n": len(tent),
            },
        },
        "paper_metrics": {
            "v13": {k: m13[k] for k in m13 if "miss" not in k and "area_decile" not in k},
            "v22": {k: m22[k] for k in m22 if "miss" not in k and "area_decile" not in k},
            "yolo": {k: my[k] for k in my if "miss" not in k and "area_decile" not in k},
        },
        "contingency_synth_mannequin": {
            "yolo_hit_v22_miss": len(yolo_hit_v22_miss),
            "v13_hit_v22_miss": len(v13_hit_v22_miss),
            "v22_hit_v13_miss": len(v22_hit_v13_miss),
            "yolo_only": len(yolo_only),
            "all_three_miss": len(all_miss),
            "both_anet_miss_yolo_hit": len(both_anet_miss_yolo_hit),
        },
        "yolo_hit_v22_miss_heat": heat_stats,
        "area_deciles_synth_mannequin": decile_rows,
        "efficiency": {
            "v13": {k: eff["v13"].get(k) for k in
                    ("params", "latency_ms_b1", "throughput_img_s_b1", "gflops")},
            "v22": {k: eff["v22"].get(k) for k in
                    ("params", "latency_ms_b1", "throughput_img_s_b1", "gflops")},
            "yolo": {k: eff["yolo"].get(k) for k in
                     ("params", "latency_ms_b1", "throughput_img_s_b1", "gflops")},
        },
        "gallery": str(gal),
    }
    (out / "miss_analysis.json").write_text(json.dumps(summary, indent=2, default=float))

    # markdown report
    lines = [
        "# v13 vs v22 vs YOLO — miss analysis (synthetic test)",
        "",
        f"Objects: {len(mann)} mannequin, {len(tent)} tent. "
        f"Match = same-class centre-in-GT (ANet peaks / YOLO boxes).",
        "",
        "## Recall (synthetic)",
        "",
        "| | v13 | v22 | YOLO |",
        "|---|---:|---:|---:|",
        f"| Mannequin | {recall(mann,'v13'):.3f} | {recall(mann,'v22'):.3f} | {recall(mann,'yolo'):.3f} |",
        f"| Tent | {recall(tent,'v13'):.3f} | {recall(tent,'v22'):.3f} | {recall(tent,'yolo'):.3f} |",
        f"| Worst mann decile | {decile_rows[0]['v13']:.3f} | {decile_rows[0]['v22']:.3f} | {decile_rows[0]['yolo']:.3f} |",
        "",
        "## Where v22 falls short (synth mannequin)",
        "",
        f"- YOLO finds, v22 misses: **{len(yolo_hit_v22_miss)}** "
        f"({100*len(yolo_hit_v22_miss)/max(len(mann),1):.1f}% of GT)",
        f"- v13 finds, v22 misses (regression): **{len(v13_hit_v22_miss)}**",
        f"- v22 finds, v13 misses (gain): **{len(v22_hit_v13_miss)}**",
        f"- YOLO-only (both ANets miss): **{len(yolo_only)}**",
        f"- All three miss: **{len(all_miss)}**",
        "",
        "### Heat on YOLO-hit / v22-miss",
        "",
        f"- n={heat_stats['n']}, median area={heat_stats['area_median']}",
        f"- median max-heat in GT box (v22)={heat_stats['h_box_v22_median']:.3f}",
        f"- fraction with h_box ≤ {args.peak_thresh}: "
        f"{heat_stats['frac_h_box_below_thresh']:.3f}",
        f"- fraction with h_box in [0.2, 0.3) (D62 dilution band): "
        f"{heat_stats['frac_h_box_in_02_03']:.3f}",
        "",
        "## Area deciles (synth mannequin recall)",
        "",
        "| decile | n | v13 | v22 | YOLO |",
        "|---:|---:|---:|---:|---:|",
    ]
    for d in decile_rows:
        lines.append(
            f"| {d['decile']} | {d['n']} | {d['v13']:.3f} | {d['v22']:.3f} | {d['yolo']:.3f} |")
    lines += [
        "",
        "## Efficiency",
        "",
        "| | v13 | v22 | YOLO |",
        "|---|---:|---:|---:|",
        f"| params | {eff['v13']['params']:,} | {eff['v22']['params']:,} | "
        f"{eff['yolo']['params']:,} |",
        f"| latency ms | {eff['v13']['latency_ms_b1']:.2f} | "
        f"{eff['v22']['latency_ms_b1']:.2f} | {eff['yolo']['latency_ms_b1']:.2f} |",
        "",
        f"Miss gallery → `{gal}/`",
        "",
    ]
    (out / "MISS_REPORT.md").write_text("\n".join(lines))

    print("\n" + (out / "MISS_REPORT.md").read_text())
    print(f"json → {out / 'miss_analysis.json'}")


if __name__ == "__main__":
    main()
