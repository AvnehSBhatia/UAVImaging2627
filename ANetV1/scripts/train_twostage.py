"""v21.4 trainer (D71 line, ARCHITECTURE.md 16.9). Run from ANetV1/.

  python scripts/train_twostage.py            # train
  python scripts/train_twostage.py --smoke    # fwd/bwd/detect sanity
  python scripts/train_twostage.py --eval runs/twostage/best.pt [--split test]

Four losses:
  center : class-agnostic center focal on the 1-ch saliency,
           pos_weight=3 (ANET_POS_W)
  smooth : the smoothing quat's DEDICATED background-TV term
           (ANET_SMOOTH_W, default 0.1)
  crop   : CE over dual-scale (50+100) crops — GT centers, 2 random bg,
           <=4 unmatched-peak hard negatives per image
  ent    : the isolated EntropyProbe (detached input; ent_auc in the
           epoch line is its entire purpose)

Family protocol: ANET_* knobs, pick_device, per-epoch val through
CenterObjectMetrics, sel = mann_synth + 0.5*tent gated to -1 above
25 fp/img (max_sel_fp), best.pt/last.pt in runs/twostage/.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from anet.data.dataset import SUASCells  # noqa: E402
from anet.model.twostage import (CROP_L, CROP_S, EntropyProbe,  # noqa: E402
                                 IMG_H, IMG_W, STRIDE, V21TwoStage)
from anet.train.losses import center_focal_loss  # noqa: E402
from anet.train.metrics import CenterObjectMetrics  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "../datasets/suas-synth-50k")
OUT = Path("runs/twostage")
MAX_SEL_FP = 25.0


def _gather_crops(model, stacked, prob, means, batch, peaks, device):
    """Training crop set: GT centers (label = class), 2 random bg
    (label 0), <=4 unmatched predicted peaks (label 0). Dual-scale:
    every entry yields a 50 AND a 100 crop; ctx = [saliency prob at the
    crop's cell, frame mean RGB], built with one fancy-index at the
    end."""
    cs, cl, cells, labels = [], [], [], []
    boxes_all = batch["boxes"].numpy()
    rng = np.random.default_rng(int(batch["seed"]))

    def _add(b, cx_px, cy_px, label):
        r = min(max(int(cy_px / STRIDE), 0), prob.shape[1] - 1)
        c = min(max(int(cx_px / STRIDE), 0), prob.shape[2] - 1)
        cs.append(model.crop_at(stacked[b], cx_px, cy_px, CROP_S))
        cl.append(model.crop_at(stacked[b], cx_px, cy_px, CROP_L))
        cells.append((b, r, c))
        labels.append(label)

    for b in range(stacked.shape[0]):
        boxes = [x for x in boxes_all[b] if x[0] >= 0]
        for cls, cx, cy, w, h in boxes:
            _add(b, cx * IMG_W, cy * IMG_H, int(cls) + 1)
        for _ in range(2):
            for _try in range(10):
                cx = rng.uniform(CROP_L / 2, IMG_W - CROP_L / 2)
                cy = rng.uniform(CROP_L / 2, IMG_H - CROP_L / 2)
                near = any(abs(cx / IMG_W - x[1]) < (x[3] / 2 + CROP_L / IMG_W / 2)
                           and abs(cy / IMG_H - x[2]) < (x[4] / 2 + CROP_L / IMG_H / 2)
                           for x in boxes)
                if not near:
                    _add(b, cx, cy, 0)
                    break
        hard = 0
        for r, c, _ in peaks[b]:
            if hard >= 4:
                break
            cx, cy = (c + 0.5) / 48, (r + 0.5) / 27
            hit = any(abs(cx - x[1]) <= x[3] / 2 and abs(cy - x[2]) <= x[4] / 2
                      for x in boxes)
            if not hit:
                _add(b, (c + 0.5) * STRIDE, (r + 0.5) * STRIDE, 0)
                hard += 1
    if not cs:
        return None, None, None, None
    bi, ri, ci = (torch.tensor(x, device=device) for x in zip(*cells))
    ctx = torch.cat([prob[bi, ri, ci].unsqueeze(1), means[bi]], 1)
    return (torch.stack(cs), torch.stack(cl), ctx.detach(),
            torch.tensor(labels, device=device))


@torch.no_grad()
def eval_entropy(probe, loader, device):
    """Rank AUC of the entropy probe's slice logits vs object-in-slice
    targets — the single number answering "is entropy useful". ~0.5 =
    carries nothing; >>0.5 = worth wiring into a future model."""
    probe.eval()
    lg, tg = [], []
    for batch in loader:
        lg.append(probe(batch["image"].to(device)).cpu())
        tg.append(probe.targets(batch["boxes"]))
    lg, tg = torch.cat(lg).flatten(), torch.cat(tg).flatten()
    pos, neg = lg[tg > 0.5], lg[tg <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
    eq = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean()
    return float(gt + 0.5 * eq)


@torch.no_grad()
def evaluate(model, loader, device, peak_thresh=0.3):
    model.eval()
    m = CenterObjectMetrics(peak_thresh=peak_thresh)
    for batch in loader:
        heat, offset = model.detect(batch["image"].to(device), peak_thresh)
        for i in range(heat.shape[0]):
            m.update(heat[i], offset[i], batch["boxes"][i].numpy(),
                     bool(batch["vd"][i]))
    s = m.summary()
    fp = s.get("fp_per_image", float("inf"))
    sel = (s.get("mannequin_recall_synthetic", 0.0) or 0.0) \
        + 0.5 * (s.get("tent_recall", 0.0) or 0.0)
    s["sel"] = sel if fp <= MAX_SEL_FP else -1.0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval", metavar="CKPT")
    ap.add_argument("--split", default="test")
    ap.add_argument("--peak-thresh", type=float, default=0.3)
    args = ap.parse_args()

    device = pick_device()
    model = V21TwoStage().to(device)
    probe = EntropyProbe().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_probe = sum(p.numel() for p in probe.parameters())
    print(f"V21TwoStage (v21.4): {n_par:,} params + EntropyProbe {n_probe} "
          f"(side-only), device={device}")
    assert n_par < 40_000

    if args.smoke:
        model.train()
        x = torch.rand(2, 3, IMG_H, IMG_W, device=device)
        out = model(x)
        assert out["sal_logits"].shape == (2, 27, 48)
        assert all(v.isfinite().all() for v in out.values())
        heat_t = torch.zeros(2, 1, 27, 48, device=device)
        heat_t[:, 0, 5, 5] = 1.0
        loss = center_focal_loss(out["sal_logits"].unsqueeze(1), heat_t,
                                 pos_weight=3.0)
        stacked = model.stack_maps(x, out)
        crop_s = model.crop_at(stacked[0], 30.0, 520.0, CROP_S)  # corner
        crop_l = model.crop_at(stacked[0], 30.0, 520.0, CROP_L)
        assert crop_s.shape == (9, CROP_S, CROP_S)
        assert crop_l.shape == (9, CROP_L, CROP_L)
        ctx = torch.cat([torch.tensor([0.5], device=device),
                         out["means"][0]]).unsqueeze(0)
        loss = loss + F.cross_entropy(
            model.crop_cnn(crop_s.unsqueeze(0), crop_l.unsqueeze(0), ctx),
            torch.tensor([1], device=device))
        loss.backward()
        assert not [k for k, p in model.named_parameters() if p.grad is None]
        model.eval()
        heat, offset = model.detect(x)
        assert heat.shape == (2, 2, 27, 48) and offset.shape == (2, 2, 27, 48)
        # probe: correct shapes, own grads live, main model UNTOUCHED
        model.zero_grad(set_to_none=True)
        lg = probe(x.detach())
        assert lg.shape == (2, probe.n_slices)
        F.binary_cross_entropy_with_logits(
            lg, torch.zeros_like(lg)).backward()
        assert not [k for k, p in probe.named_parameters() if p.grad is None]
        assert all(p.grad is None for p in model.parameters()), \
            "entropy probe leaked gradient into the main model"
        print("PASS — fwd/bwd all-live grads, detect contract, "
              "probe isolated")
        return

    if args.eval:
        sd = torch.load(args.eval, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        ds = SUASCells(ROOT, args.split, center=True)
        loader = DataLoader(ds, batch_size=8, num_workers=0)
        s = evaluate(model, loader, device, args.peak_thresh)
        for k in ("mannequin_recall", "tent_recall", "fp_per_image",
                  "mannequin_recall_synthetic",
                  "mannequin_recall_smallest_decile"):
            print(f"{k:38s}{s.get(k, float('nan')):10.3f}")
        return

    lr = float(os.environ.get("ANET_LR", "1.5e-3"))
    epochs = int(os.environ.get("ANET_EPOCHS", "15"))
    batch = int(os.environ.get("ANET_BATCH", "32"))
    limit = int(os.environ.get("ANET_SAMPLES", "0"))
    smooth_w = float(os.environ.get("ANET_SMOOTH_W", "0.1"))
    pos_w = float(os.environ.get("ANET_POS_W", "3.0"))
    workers = 0 if torch.version.hip else 4
    cache = os.environ.get("ANET_CACHE") == "1"

    tr = SUASCells(ROOT, "train", center=True, cache=cache)
    va = SUASCells(ROOT, "val", center=True)
    from torch.utils.data import Subset
    tr_ds = Subset(tr, list(range(min(limit, len(tr))))) if limit else tr
    va_ds = Subset(va, range(min(800, len(va))))
    train_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True,
                              num_workers=workers, drop_last=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(va_ds, batch_size=8, num_workers=workers,
                            persistent_workers=workers > 0)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()),
        lr=lr, weight_decay=1e-4)
    ent_pos_w = torch.tensor(4.0, device=device)  # ~2-3 object slices of 19
    cls_w = torch.tensor([0.5, 1.0, 1.0], device=device)  # bg down-weight
    steps = max(len(train_loader) * epochs, 1)
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    log_every = int(os.environ.get("ANET_LOG_EVERY", "50"))
    n_steps = len(train_loader)
    print(f"train: {n_steps} steps/epoch x {epochs} epochs (batch {batch}), "
          f"smooth_w={smooth_w} pos_w={pos_w}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(epochs):
        model.train()
        probe.train()
        t0, nb = time.time(), 0
        gstep = ep * n_steps
        tot = torch.zeros((), device=device)
        parts = torch.zeros(4, device=device)
        for batch_d in train_loader:
            img = batch_d["image"].to(device)
            heat_t = batch_d["heat"].to(device)          # (B,2,27,48)
            out = model(img)
            # class-agnostic center loss on the 1-ch saliency (classes
            # come from the crop stage)
            sal_t = heat_t.max(1, keepdim=True).values
            l_center = center_focal_loss(out["sal_logits"].unsqueeze(1),
                                         sal_t, pos_weight=pos_w)
            s4 = F.avg_pool2d(out["smooth"], 4)          # (B,3,135,240)
            bg = (F.interpolate(sal_t, scale_factor=5, mode="nearest")
                  < 0.05).float()
            tv = (s4 - F.avg_pool2d(s4, 3, 1, 1)).abs()
            l_smooth = (tv * bg).sum() / bg.sum().clamp_min(1.0) / 3.0
            # dual-scale crop classification
            prob = torch.sigmoid(out["sal_logits"].detach())
            with torch.no_grad():
                peaks = model.find_peaks(prob, 0.3)
            batch_d["seed"] = gstep + nb
            stacked = model.stack_maps(img, out)
            crops_s, crops_l, ctx, labels = _gather_crops(
                model, stacked, prob, out["means"].detach(), batch_d,
                peaks, device)
            l_crop = (F.cross_entropy(
                model.crop_cnn(crops_s, crops_l, ctx), labels, weight=cls_w)
                if crops_s is not None else l_center * 0.0)
            # side probe: DETACHED input — its gradient can only reach
            # its own params, never the main model
            l_ent = F.binary_cross_entropy_with_logits(
                probe(img.detach()),
                probe.targets(batch_d["boxes"]).to(device),
                pos_weight=ent_pos_w)
            loss = l_center + smooth_w * l_smooth + l_crop + l_ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            nb += 1
            tot += loss.detach()
            parts += torch.stack([l_center.detach(), l_smooth.detach(),
                                  l_crop.detach(), l_ent.detach()])
            if nb % log_every == 0:
                r = nb * batch / (time.time() - t0)
                pc, ps, pk, pe = (parts / nb).tolist()
                print(f"  ep {ep} step {nb}/{n_steps}: "
                      f"loss={float(tot) / nb:.4f} (center {pc:.3f} "
                      f"smooth {ps:.3f} crop {pk:.3f} ent {pe:.3f}, "
                      f"{r:.0f} img/s)", flush=True)
        tot = float(tot)
        s = evaluate(model, val_loader, device, args.peak_thresh)
        auc = eval_entropy(probe, val_loader, device)
        flag = ""
        if s["sel"] > best:
            best = s["sel"]
            torch.save({"model": model.state_dict(),
                        "probe": probe.state_dict(), "arch": "v21.4",
                        "epoch": ep, "val": s, "ent_auc": auc},
                       OUT / "best.pt")
            flag = "  *best*"
        torch.save({"model": model.state_dict(),
                    "probe": probe.state_dict(), "arch": "v21.4",
                    "epoch": ep, "val": s, "ent_auc": auc},
                   OUT / "last.pt")
        print(f"epoch {ep:3d}: loss={tot / max(nb, 1):.4f} "
              f"mann_r={s.get('mannequin_recall', 0):.3f} "
              f"tent_r={s.get('tent_recall', 0):.3f} "
              f"fp={s.get('fp_per_image', 0):.3f} sel={s['sel']:.3f} "
              f"ent_auc={auc:.3f} ({time.time() - t0:.0f}s){flag}",
              flush=True)
    print(f"done — best sel {best:.3f} -> {OUT / 'best.pt'}")


if __name__ == "__main__":
    main()
