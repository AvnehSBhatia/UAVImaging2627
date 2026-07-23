"""Real-scene object RECALL on the 14 hand-labeled web frames (D93).

The §24.5 lesson: on label-free frames the `webscene_check` PEAK is background,
so a model that fires harder on brush scores higher while detecting nothing. This
script uses the D93 ground truth (`runs/viz_web_scenes/labels.json`, centres
verified by zoom) to measure the two things that actually matter, per object:

  1. response AT the GT centre (max in a +-1 cell window on the 27x48 grid) —
     threshold-free, the direct object-appearance signal.
  2. HIT at a peak threshold: is there a 3x3-local-max peak >= t within R cells
     of the GT centre? -> real object recall. FALSE POSITIVES are peaks >= t not
     near any GT of that class.

Mannequin and tent are separate heat channels, scored against their own GT.
`ood` frames (oblique / composited stress tests) are excluded from the
mission-geometry aggregate and reported separately. This is the project's first
real-recall yardstick; rank real-scene changes on THIS, never on a scene peak.

  python scripts/eval_real_scenes.py --ckpts v13_best v22g_r2_best auto_v22_b7_ft_102k
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anet.model.anet import ANetV1  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

W, H = 960, 540
R_CELLS = 2          # centre-match tolerance (grid cells; ~40 px at stride 20)
ROOT = Path("runs/viz_web_scenes")


def load(ckpt, device):
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))
    m = ANetV1.from_state_dict(sd)
    m.load_state_dict(sd, strict=False)
    return m.eval().to(device)


@torch.no_grad()
def heat(model, scene, device):
    im = Image.open(ROOT / "infer" / scene / "00_input.png").convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)
    return torch.sigmoid(model(x.unsqueeze(0).to(device))["heat"])[0].cpu().numpy()


def peaks(prob, t):
    """3x3 local maxima >= t — the CenterObjectMetrics rule. Returns (r,c) list."""
    p = torch.from_numpy(prob)[None, None]
    mx = F.max_pool2d(p, 3, 1, 1)[0, 0].numpy()
    return list(zip(*np.where((prob >= mx) & (prob >= t))))


def at_gt(prob, cx, cy):
    gh, gw = prob.shape
    r, c = int(cy * gh), int(cx * gw)
    return float(prob[max(0, r - 1):r + 2, max(0, c - 1):c + 2].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+",
                    default=["v13_best", "v22g_r2_best", "auto_v22_b7_ft_102k"])
    ap.add_argument("--thresh", type=float, nargs="+", default=[0.30, 0.40])
    args = ap.parse_args()
    device = pick_device()

    lab = json.loads((ROOT / "labels.json").read_text())["scenes"]
    paths = [p if Path(p).exists() else f"runs/anet/{p}.pt" for p in args.ckpts]
    names = [Path(p).stem for p in paths]
    models = {n: load(p, device) for n, p in zip(names, paths)}

    # ---- gather GT ----
    mann_gt, tent_gt = [], []   # (scene, cx, cy, diff/ood)
    for s, d in lab.items():
        ood = d.get("ood", False)
        for cls, cx, cy, bw, bh in d["boxes"]:
            tag = "ood" if ood else d.get("diff", {}).get(f"{cx},{cy}", "-")
            (mann_gt if cls == 0 else tent_gt).append((s, cx, cy, tag))
    n_mann = sum(1 for g in mann_gt if g[3] != "ood")
    n_tent = sum(1 for g in tent_gt if g[3] != "ood")
    print(f"GT: {n_mann} mannequins, {n_tent} tents (mission geometry) + "
          f"{len(tent_gt) - n_tent} ood tents\n")

    # ---- response AT GT (threshold-free) ----
    print("=" * 78)
    print("RESPONSE AT GT CENTRE (max +-1 cell) — threshold-free object signal")
    hdr = f"{'object':<34}{'diff':<8}" + "".join(f"{n[:13]:>15}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    prob_cache = {n: {} for n in names}
    for gt_list, cls, lbl in ((mann_gt, 0, "MANNEQUIN"), (tent_gt, 1, "TENT")):
        print(f"[{lbl}]")
        for s, cx, cy, tag in gt_list:
            row = f"  {s[:32]:<32}{tag:<8}"
            for n in names:
                if s not in prob_cache[n]:
                    prob_cache[n][s] = heat(models[n], s, device)
                row += f"{at_gt(prob_cache[n][s][cls], cx, cy):>15.3f}"
            print(row)
    # aggregate medians
    print("-" * len(hdr))

    def med(gt_list, cls, pred):
        vals = {n: [] for n in names}
        for s, cx, cy, tag in gt_list:
            if not pred(tag):
                continue
            for n in names:
                vals[n].append(at_gt(prob_cache[n][s][cls], cx, cy))
        return {n: (np.median(v) if v else float("nan"), len(v)) for n, v in vals.items()}

    for lbl, gt, cls, pred in (
            ("mann median (all mission)", mann_gt, 0, lambda t: t != "ood"),
            ("  mann median (hard only)", mann_gt, 0, lambda t: t == "hard"),
            ("  mann median (easy/med)", mann_gt, 0, lambda t: t in ("easy", "medium")),
            ("tent median (all mission)", tent_gt, 1, lambda t: t != "ood")):
        m = med(gt, cls, pred)
        row = f"  {lbl:<30}" + "".join(f"{m[n][0]:>13.3f}(n{m[n][1]})"[:15] for n in names)
        print(row)

    # ---- recall + fp at thresholds ----
    for t in args.thresh:
        print("\n" + "=" * 78)
        print(f"RECALL @ peak_thresh={t}  (hit = peak within {R_CELLS} cells of GT)")
        print(f"{'metric':<30}" + "".join(f"{n[:13]:>15}" for n in names))
        print("-" * (30 + 15 * len(names)))
        stats = {n: {} for n in names}
        for n in names:
            for cls, gt_list, nlab in ((0, mann_gt, "mann"), (1, tent_gt, "tent")):
                hit = miss = fp = 0
                gt_by_scene = {}
                for s, cx, cy, tag in gt_list:
                    if tag == "ood":
                        continue
                    gt_by_scene.setdefault(s, []).append((cx, cy))
                for s, gts in gt_by_scene.items():
                    pk = peaks(prob_cache[n][s][cls], t)
                    gh, gw = prob_cache[n][s][cls].shape
                    used = set()
                    for cx, cy in gts:
                        gr, gc = int(cy * gh), int(cx * gw)
                        found = [p for p in pk if max(abs(p[0] - gr), abs(p[1] - gc)) <= R_CELLS]
                        if found:
                            hit += 1
                            used.add(found[0])
                        else:
                            miss += 1
                    # fp = peaks in this scene not matched to a gt
                    for p in pk:
                        if any(max(abs(p[0] - int(cy * gh)), abs(p[1] - int(cx * gw))) <= R_CELLS
                               for cx, cy in gts):
                            continue
                        fp += 1
                # fp from scenes with NO gt of this class (pure background for it)
                for s, d in lab.items():
                    if d.get("ood"):
                        continue
                    if s in gt_by_scene:
                        continue
                    if s not in prob_cache[n]:
                        prob_cache[n][s] = heat(models[n], s, device)
                    fp += len(peaks(prob_cache[n][s][cls], t))
                stats[n][nlab] = (hit, hit + miss, fp)
        n_frames = sum(1 for s, d in lab.items() if not d.get("ood"))
        for nlab in ("mann", "tent"):
            row = f"  {nlab} recall{'':<18}"
            for n in names:
                h, tot, _ = stats[n][nlab]
                row += f"{h}/{tot}={h/tot:.2f}"[:15].rjust(15)
            print(row)
            row = f"  {nlab} fp/frame{'':<16}"
            for n in names:
                _, _, fp = stats[n][nlab]
                row += f"{fp / n_frames:>15.2f}"
            print(row)


if __name__ == "__main__":
    main()
