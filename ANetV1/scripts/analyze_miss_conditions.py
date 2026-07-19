"""Miss-condition autopsy: v13 vs v22 vs YOLO26n, synth + VisDrone, thresh sweeps.

Uses cached preds in runs/paper_bench_tri/ (from analyze_v13_v22_misses.py).

  cd ANetV1
  python scripts/analyze_miss_conditions.py --out runs/paper_bench_tri

Writes CONDITIONS.md, conditions.json, sweep_*.csv, and refreshes the
miss-condition breakdown the canvas reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet.train.metrics import CenterObjectMetrics, V12_H, V12_W  # noqa: E402

import importlib.util

_bp = Path(__file__).resolve().parent / "benchmark_paper.py"
_spec = importlib.util.spec_from_file_location("benchmark_paper", _bp)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
eval_anet_cached = _mod.eval_anet_cached
eval_yolo_cached = _mod.eval_yolo_cached
sweep_anet = _mod.sweep_anet
sweep_yolo = _mod.sweep_yolo


def peaks(heat_c, off, thresh):
    rows, cols = CenterObjectMetrics._find_peaks(heat_c, thresh)
    if len(rows) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    dx, dy = off[0, rows, cols], off[1, rows, cols]
    return (cols + dx) / V12_W, (rows + dy) / V12_H, heat_c[rows, cols]


def anet_diag(heat, off, box, thresh=0.3):
    c = int(box[0])
    bx, by, bw, bh = map(float, box[1:])
    cx, cy, p = peaks(heat[c], off, thresh)
    inside = ((cx >= bx - bw / 2) & (cx <= bx + bw / 2)
              & (cy >= by - bh / 2) & (cy <= by + bh / 2))
    r0 = max(int((by - bh / 2) * V12_H), 0)
    r1 = min(int(np.ceil((by + bh / 2) * V12_H)), V12_H)
    c0 = max(int((bx - bw / 2) * V12_W), 0)
    c1 = min(int(np.ceil((bx + bw / 2) * V12_W)), V12_W)
    patch = heat[c, r0:r1, c0:c1] if r1 > r0 and c1 > c0 else heat[c:c + 1, 0:1, 0:1]
    h_box = float(patch.max()) if patch.size else 0.0
    r = min(max(int(by * V12_H), 0), V12_H - 1)
    cc = min(max(int(bx * V12_W), 0), V12_W - 1)
    h_center = float(heat[c, r, cc])
    # nearest peak of this class (even if outside box) — localization miss?
    if len(p):
        d2 = (cx - bx) ** 2 + (cy - by) ** 2
        j = int(np.argmin(d2))
        near_p, near_d = float(p[j]), float(np.sqrt(d2[j]))
    else:
        near_p, near_d = 0.0, float("nan")
    return {
        "found": bool(inside.any()),
        "h_box": h_box,
        "h_center": h_center,
        "near_peak_p": near_p,
        "near_peak_dist": near_d,
        "n_peaks": int(len(p)),
    }


def yolo_diag(pred_boxes, confs, box, conf=0.25):
    c = int(box[0])
    bx, by, bw, bh = map(float, box[1:])
    if len(pred_boxes) == 0 or len(confs) == 0:
        return {"found": False, "best_iou": 0.0, "best_conf": 0.0,
                "n_cls_preds": 0, "near_dist": float("nan")}
    keep = confs >= conf
    pb, cf = pred_boxes[keep], confs[keep]
    best_iou = best_conf = 0.0
    near_dist = float("inf")
    n_cls = 0
    hit = False
    for row, sc in zip(pb, cf):
        if int(row[0]) != c:
            continue
        n_cls += 1
        px, py, pw, ph = map(float, row[1:])
        dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        near_dist = min(near_dist, dist)
        x0 = max(bx - bw / 2, px - pw / 2)
        y0 = max(by - bh / 2, py - ph / 2)
        x1 = min(bx + bw / 2, px + pw / 2)
        y1 = min(by + bh / 2, py + ph / 2)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = bw * bh + pw * ph - inter
        iou = inter / max(union, 1e-9)
        best_iou = max(best_iou, iou)
        if (abs(px - bx) <= bw / 2 and abs(py - by) <= bh / 2) or iou >= 0.1:
            hit = True
            best_conf = max(best_conf, float(sc))
    if near_dist == float("inf"):
        near_dist = float("nan")
    return {"found": hit, "best_iou": best_iou, "best_conf": best_conf,
            "n_cls_preds": n_cls, "near_dist": near_dist}


def heat_band(h):
    if h < 0.1:
        return "h<0.1 (dead)"
    if h < 0.2:
        return "h∈[0.1,0.2)"
    if h < 0.3:
        return "h∈[0.2,0.3) dilution"
    if h < 0.5:
        return "h∈[0.3,0.5) peak-capable"
    return "h≥0.5"


def edge_bin(box):
    """How close is the box centre to the canvas edge (margin in norm units)."""
    _, cx, cy, _, _ = box
    m = min(cx, 1 - cx, cy, 1 - cy)
    if m < 0.05:
        return "edge(<5%)"
    if m < 0.15:
        return "near-edge(5-15%)"
    return "interior"


def aspect_bin(box):
    _, _, _, w, h = box
    ar = w / max(h, 1e-9)
    if ar < 0.5:
        return "tall"
    if ar > 2.0:
        return "wide"
    return "squareish"


def pct(n, d):
    return 100.0 * n / max(d, 1)


def cond_table(rows, key_fn, models=("v13", "v22", "yolo")):
    """Group rows by condition; report n and recall per model."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)
    out = []
    for k in sorted(buckets, key=lambda x: -len(buckets[x])):
        sub = buckets[k]
        row = {"condition": k, "n": len(sub)}
        for m in models:
            row[m] = sum(r[m] for r in sub) / len(sub)
            row[f"{m}_miss"] = sum(not r[m] for r in sub)
        out.append(row)
    return out


def miss_condition_mix(misses, key_fn):
    """Among misses, fraction in each condition bucket."""
    buckets = defaultdict(int)
    for r in misses:
        buckets[key_fn(r)] += 1
    total = max(len(misses), 1)
    return [{"condition": k, "n": buckets[k], "pct": pct(buckets[k], total)}
            for k in sorted(buckets, key=lambda x: -buckets[x])]


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/paper_bench_tri")
    ap.add_argument("--peak-thresh", type=float, default=0.3)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.out)

    p13 = np.load(out / "v13_preds.npz", allow_pickle=True)
    p22 = np.load(out / "v22_preds.npz", allow_pickle=True)
    yolo = np.load(out / "yolo_preds.npz", allow_pickle=True)
    stems = json.loads((out / "yolo_meta.json").read_text())["stems"]

    # ---- threshold sweeps (full + will slice synth/VD in report via cached metrics)
    thr = [round(x, 2) for x in np.linspace(0.05, 0.80, 16)]
    print("sweeping v13…", flush=True)
    sw13 = sweep_anet(p13, thr)
    print("sweeping v22…", flush=True)
    sw22 = sweep_anet(p22, thr)
    print("sweeping yolo…", flush=True)
    swy = sweep_yolo(yolo, thr)
    fields = list(sw13[0].keys())
    write_csv(out / "sweep_v13.csv", sw13, fields)
    write_csv(out / "sweep_v22.csv", sw22, fields)
    write_csv(out / "sweep_yolo.csv", swy, fields)

    # ---- per-object rows (all sources)
    rows = []
    for i in range(len(stems)):
        is_vd = bool(yolo["vd"][i])
        gt = yolo["gt_boxes"][i]
        h13 = p13["heat"][i].astype(np.float32)
        o13 = p13["offset"][i].astype(np.float32)
        h22 = p22["heat"][i].astype(np.float32)
        o22 = p22["offset"][i].astype(np.float32)
        pb, cf = yolo["pred_boxes"][i], yolo["confs"][i]
        for box in gt:
            if box[0] < 0:
                continue
            d13 = anet_diag(h13, o13, box, args.peak_thresh)
            d22 = anet_diag(h22, o22, box, args.peak_thresh)
            dy = yolo_diag(pb, cf, box, args.yolo_conf)
            area = float(box[3] * box[4])
            rows.append({
                "stem": stems[i], "cls": int(box[0]), "vd": is_vd,
                "area": area, "w": float(box[3]), "h": float(box[4]),
                "aspect": float(box[3]) / max(float(box[4]), 1e-9),
                "edge": edge_bin(box), "shape": aspect_bin(box),
                "v13": d13["found"], "v22": d22["found"], "yolo": dy["found"],
                "h_box_v13": d13["h_box"], "h_box_v22": d22["h_box"],
                "h_center_v22": d22["h_center"],
                "heat_band_v22": heat_band(d22["h_box"]),
                "near_dist_v22": d22["near_peak_dist"],
                "near_p_v22": d22["near_peak_p"],
                "yolo_iou": dy["best_iou"], "yolo_conf": dy["best_conf"],
                "yolo_n_cls": dy["n_cls_preds"],
            })

    # area decile edges from synth mannequin (stable reference) + VD-specific
    mann_all = [r for r in rows if r["cls"] == 0]
    mann_syn = [r for r in mann_all if not r["vd"]]
    mann_vd = [r for r in mann_all if r["vd"]]
    tent_all = [r for r in rows if r["cls"] == 1]

    def assign_decile(subset, edges):
        for r in subset:
            a = r["area"]
            d = 9
            for i in range(10):
                lo, hi = edges[i], edges[i + 1]
                if (lo <= a <= hi) if i == 9 else (lo <= a < hi):
                    d = i
                    break
            r["area_decile"] = d

    edges_syn = np.quantile([r["area"] for r in mann_syn], np.linspace(0, 1, 11))
    edges_vd = np.quantile([r["area"] for r in mann_vd], np.linspace(0, 1, 11))
    assign_decile(mann_syn, edges_syn)
    assign_decile(mann_vd, edges_vd)
    for r in tent_all:
        r["area_decile"] = -1  # unused

    def recall(sub, m):
        return sum(r[m] for r in sub) / max(len(sub), 1)

    # ---- miss sets at operating point
    def miss_set(sub, model):
        return [r for r in sub if not r[model]]

    # conditions among v22 misses, split by source
    def autopsy(label, sub):
        misses = miss_set(sub, "v22")
        yolo_hit = [r for r in misses if r["yolo"]]
        v13_hit = [r for r in misses if r["v13"]]
        return {
            "label": label,
            "n_gt": len(sub),
            "recall": {m: recall(sub, m) for m in ("v13", "v22", "yolo")},
            "n_v22_miss": len(misses),
            "n_yolo_hit_v22_miss": len(yolo_hit),
            "n_v13_hit_v22_miss": len(v13_hit),
            "miss_by_heat_band": miss_condition_mix(misses, lambda r: r["heat_band_v22"]),
            "miss_by_edge": miss_condition_mix(misses, lambda r: r["edge"]),
            "miss_by_shape": miss_condition_mix(misses, lambda r: r["shape"]),
            "miss_by_area_decile": miss_condition_mix(
                misses, lambda r: f"decile_{r.get('area_decile', -1)}"),
            "yolo_hit_miss_by_heat": miss_condition_mix(yolo_hit, lambda r: r["heat_band_v22"]),
            "heat_on_yolo_hit_miss": {
                "n": len(yolo_hit),
                "median_h_box": float(np.median([r["h_box_v22"] for r in yolo_hit])) if yolo_hit else None,
                "frac_below_0.3": pct(sum(r["h_box_v22"] < 0.3 for r in yolo_hit), len(yolo_hit)) / 100,
                "frac_dilution_0.2_0.3": pct(
                    sum(0.2 <= r["h_box_v22"] < 0.3 for r in yolo_hit), len(yolo_hit)) / 100,
                "median_area": float(np.median([r["area"] for r in yolo_hit])) if yolo_hit else None,
                "frac_near_peak_outside_box": pct(
                    sum((not np.isnan(r["near_dist_v22"])) and r["near_dist_v22"] > 0
                        and r["near_p_v22"] >= args.peak_thresh and not r["v22"]
                        for r in yolo_hit), len(yolo_hit)) / 100,
            },
            "recall_by_heat_band": cond_table(sub, lambda r: r["heat_band_v22"]),
            "recall_by_edge": cond_table(sub, lambda r: r["edge"]),
            "recall_by_shape": cond_table(sub, lambda r: r["shape"]),
            "recall_by_area_decile": cond_table(
                sub, lambda r: f"decile_{r.get('area_decile', -1)}"),
        }

    reports = {
        "mannequin_synthetic": autopsy("mannequin_synthetic", mann_syn),
        "mannequin_visdrone": autopsy("mannequin_visdrone", mann_vd),
        "tent_all": autopsy("tent_all", tent_all),  # tents are synth-only in this dataset
    }

    # localization vs absence: among v22 misses with h_box>=0.3, is there a peak nearby?
    loc_miss = [r for r in miss_set(mann_syn, "v22")
                if r["h_box_v22"] >= args.peak_thresh]
    reports["mannequin_synthetic"]["localization_misses"] = {
        "n_miss_with_h_box_ge_thresh": len(loc_miss),
        "frac_of_v22_misses": pct(len(loc_miss), len(miss_set(mann_syn, "v22"))) / 100,
        "note": "heat in box clears thresh but no peak centre landed inside — "
                "peak NMS/offset localization failure, not absence of evidence",
    }

    # ---- operating-point paper metrics for FP context
    m13 = eval_anet_cached(p13, args.peak_thresh)
    m22 = eval_anet_cached(p22, args.peak_thresh)
    my = eval_yolo_cached(yolo, args.yolo_conf)

    summary = {
        "meta": {
            "peak_thresh": args.peak_thresh,
            "yolo_conf": args.yolo_conf,
            "n_mann_syn": len(mann_syn),
            "n_mann_vd": len(mann_vd),
            "n_tent": len(tent_all),
            "models": ["v13", "v22", "yolo26n"],
        },
        "operating_point": {
            "v13": {
                "mann_syn": m13["mannequin_recall_synthetic"]["mean"],
                "mann_vd": m13["mannequin_recall_visdrone"]["mean"],
                "tent": m13["tent_recall"]["mean"],
                "fp_per_image": m13["fp_per_image"],
                "worst_decile_syn": m13.get(
                    "mannequin_recall_smallest_decile_synthetic", {}).get("mean"),
            },
            "v22": {
                "mann_syn": m22["mannequin_recall_synthetic"]["mean"],
                "mann_vd": m22["mannequin_recall_visdrone"]["mean"],
                "tent": m22["tent_recall"]["mean"],
                "fp_per_image": m22["fp_per_image"],
                "worst_decile_syn": m22.get(
                    "mannequin_recall_smallest_decile_synthetic", {}).get("mean"),
            },
            "yolo": {
                "mann_syn": my["mannequin_recall_synthetic"]["mean"],
                "mann_vd": my["mannequin_recall_visdrone"]["mean"],
                "tent": my["tent_recall"]["mean"],
                "fp_per_image": my["fp_per_image"],
                "worst_decile_syn": my.get(
                    "mannequin_recall_smallest_decile_synthetic", {}).get("mean"),
            },
        },
        "sweeps": {"v13": sw13, "v22": sw22, "yolo": swy},
        "conditions": reports,
    }
    (out / "conditions.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- markdown
    def fmt_mix(mix, top=6):
        lines = []
        for m in mix[:top]:
            lines.append(f"  - {m['condition']}: {m['n']} ({m['pct']:.1f}%)")
        return "\n".join(lines)

    def fmt_recall_table(tbl, title):
        lines = [f"### {title}", "",
                 "| condition | n | v13 | v22 | YOLO |",
                 "|---|---:|---:|---:|---:|"]
        for r in tbl:
            lines.append(
                f"| {r['condition']} | {r['n']} | {r['v13']:.3f} | "
                f"{r['v22']:.3f} | {r['yolo']:.3f} |")
        return "\n".join(lines)

    def fmt_sweep(name, sw):
        lines = [
            f"### {name}",
            "",
            "| thr | mann_syn | mann_VD | tent | worst_dec | fp/img | prec |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sw:
            lines.append(
                f"| {r['threshold']:.2f} | {r['mannequin_recall_synthetic']:.3f} | "
                f"{r['mannequin_recall_visdrone']:.3f} | {r['tent_recall']:.3f} | "
                f"{r['mannequin_recall_smallest_decile']:.3f} | "
                f"{r['fp_per_image']:.2f} | {r['object_precision']:.3f} |")
        return "\n".join(lines)

    op = summary["operating_point"]
    syn = reports["mannequin_synthetic"]
    vd = reports["mannequin_visdrone"]
    md = [
        "# Miss conditions — v13 vs v22 vs YOLO26n",
        "",
        f"Operating point: ANet peak_thresh={args.peak_thresh}, "
        f"YOLO conf={args.yolo_conf}.",
        "",
        "## Operating-point recall",
        "",
        "| slice | v13 | v22 | YOLO26n |",
        "|---|---:|---:|---:|",
        f"| Mannequin synth (n={len(mann_syn)}) | {op['v13']['mann_syn']:.3f} | "
        f"{op['v22']['mann_syn']:.3f} | {op['yolo']['mann_syn']:.3f} |",
        f"| Mannequin VisDrone (n={len(mann_vd)}) | {op['v13']['mann_vd']:.3f} | "
        f"{op['v22']['mann_vd']:.3f} | {op['yolo']['mann_vd']:.3f} |",
        f"| Tent | {op['v13']['tent']:.3f} | {op['v22']['tent']:.3f} | "
        f"{op['yolo']['tent']:.3f} |",
        f"| Worst-decile mann synth | {op['v13']['worst_decile_syn']:.3f} | "
        f"{op['v22']['worst_decile_syn']:.3f} | {op['yolo']['worst_decile_syn']:.3f} |",
        f"| FP/image (pooled) | {op['v13']['fp_per_image']:.2f} | "
        f"{op['v22']['fp_per_image']:.2f} | {op['yolo']['fp_per_image']:.2f} |",
        "",
        "## When does v22 miss? — synthetic mannequins",
        "",
        f"v22 misses {syn['n_v22_miss']}/{syn['n_gt']} "
        f"({pct(syn['n_v22_miss'], syn['n_gt']):.1f}%). "
        f"Of those, YOLO still finds {syn['n_yolo_hit_v22_miss']}; "
        f"v13 still finds {syn['n_v13_hit_v22_miss']}.",
        "",
        "**Among all v22 misses, by in-box heat (the dominant condition):**",
        fmt_mix(syn["miss_by_heat_band"]),
        "",
        "**Among YOLO-hit / v22-miss only (the closable gap):**",
        fmt_mix(syn["yolo_hit_miss_by_heat"]),
        "",
        f"- median h_box={syn['heat_on_yolo_hit_miss']['median_h_box']}",
        f"- frac h_box<0.3={syn['heat_on_yolo_hit_miss']['frac_below_0.3']:.3f}",
        f"- frac in dilution [0.2,0.3)="
        f"{syn['heat_on_yolo_hit_miss']['frac_dilution_0.2_0.3']:.3f}",
        f"- localization misses (h_box≥thresh but peak outside): "
        f"{syn['localization_misses']['n_miss_with_h_box_ge_thresh']} "
        f"({100*syn['localization_misses']['frac_of_v22_misses']:.1f}% of v22 misses)",
        "",
        "**Miss mix by area decile / edge / shape:**",
        "Area:",
        fmt_mix(syn["miss_by_area_decile"]),
        "Edge:",
        fmt_mix(syn["miss_by_edge"]),
        "Shape:",
        fmt_mix(syn["miss_by_shape"]),
        "",
        fmt_recall_table(syn["recall_by_heat_band"],
                         "Recall by v22 in-box heat band (synth mann)"),
        "",
        fmt_recall_table(syn["recall_by_area_decile"],
                         "Recall by area decile (synth mann)"),
        "",
        fmt_recall_table(syn["recall_by_edge"],
                         "Recall by frame position (synth mann)"),
        "",
        "## When does v22 miss? — VisDrone mannequins",
        "",
        f"v22 misses {vd['n_v22_miss']}/{vd['n_gt']} "
        f"({pct(vd['n_v22_miss'], vd['n_gt']):.1f}%). "
        f"YOLO-hit among those: {vd['n_yolo_hit_v22_miss']}. "
        f"Both models are near floor — VisDrone people are far below SUAS GSD.",
        "",
        "**v22 miss mix by heat band:**",
        fmt_mix(vd["miss_by_heat_band"]),
        "",
        "**v22 miss mix by VD area decile:**",
        fmt_mix(vd["miss_by_area_decile"]),
        "",
        fmt_recall_table(vd["recall_by_area_decile"],
                         "Recall by area decile (VisDrone mann)"),
        "",
        fmt_recall_table(vd["recall_by_heat_band"],
                         "Recall by v22 in-box heat (VisDrone mann)"),
        "",
        "## Confidence / peak threshold sweeps",
        "",
        "Same thresholds for ANet peak_thresh and YOLO conf. "
        "`worst_dec` is pooled (VisDrone-dominated); use `mann_syn` for SUAS.",
        "",
        fmt_sweep("v13", sw13),
        "",
        fmt_sweep("v22", sw22),
        "",
        fmt_sweep("YOLO26n", swy),
        "",
        "## Reading the conditions",
        "",
        "1. **Synthetic misses are heat-starvation, not localization.** "
        f"Only {syn['localization_misses']['n_miss_with_h_box_ge_thresh']} synth "
        "v22 misses have h_box ≥ 0.3; the rest never produce a legal peak.",
        "2. **Small + diluted:** YOLO-recoverable misses cluster in decile 0–1 "
        "with h_box in [0.1, 0.3) — the D62 dilution regime the SPD/peak "
        "branch was supposed to fix, and did not on net.",
        "3. **VisDrone is a different problem:** both ANet and YOLO sit near "
        f"{op['v22']['mann_vd']:.1%} / {op['yolo']['mann_vd']:.1%} — almost all "
        "VD boxes are h<0.1 dead for v22. Do not tune SUAS on pooled recall.",
        "4. **Sweep tradeoff:** lowering peak_thresh lifts synth mann recall "
        "but FP/image explodes (see sweeps). YOLO's conf sweep is far "
        "cleaner on the precision axis.",
        "",
    ]
    (out / "CONDITIONS.md").write_text("\n".join(md))
    print((out / "CONDITIONS.md").read_text())
    print(f"\n→ {out / 'conditions.json'}")
    print(f"→ {out / 'sweep_v13.csv'}, sweep_v22.csv, sweep_yolo.csv")


if __name__ == "__main__":
    main()
