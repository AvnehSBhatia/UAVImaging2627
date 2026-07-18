"""v21.2 trainer (D71 ablation, ARCHITECTURE.md 16.9). Run from ANetV1/.

  python scripts/train_twostage.py            # train
  python scripts/train_twostage.py --smoke    # fwd/bwd/detect sanity
  python scripts/train_twostage.py --eval runs/twostage/best.pt [--split test]

Two losses (the crop stage is REMOVED per owner direction):
  center : 2-class center_focal_loss (D57) directly on the heat logits,
           pos_weight=3 (ANET_POS_W) — the v12 slow-climb fix
  smooth : the smoothing quat's DEDICATED background-TV term
           (ANET_SMOOTH_W, default 0.1)

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

from anet.data.dataset import SUASCells  # noqa: E402
from anet.model.twostage import IMG_H, IMG_W, V21TwoStage  # noqa: E402
from anet.train.losses import center_focal_loss  # noqa: E402
from anet.train.metrics import CenterObjectMetrics  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "../datasets/suas-synth-50k")
OUT = Path("runs/twostage")
MAX_SEL_FP = 25.0


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
    n_par = sum(p.numel() for p in model.parameters())
    print(f"V21TwoStage (v21.2 ablation): {n_par} params, device={device}")
    assert n_par < 40_000

    if args.smoke:
        model.train()
        x = torch.rand(2, 3, IMG_H, IMG_W, device=device)
        out = model(x)
        assert out["heat_logits"].shape == (2, 2, 27, 48)
        assert all(v.isfinite().all() for v in out.values())
        heat_t = torch.zeros(2, 2, 27, 48, device=device)
        heat_t[:, 0, 5, 5] = 1.0
        loss = center_focal_loss(out["heat_logits"], heat_t, pos_weight=3.0)
        loss.backward()
        assert not [k for k, p in model.named_parameters() if p.grad is None]
        model.eval()
        heat, offset = model.detect(x)
        assert heat.shape == (2, 2, 27, 48) and offset.shape == (2, 2, 27, 48)
        print("PASS — fwd/bwd all-live grads, detect contract")
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

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        t0, nb = time.time(), 0
        tot = torch.zeros((), device=device)
        parts = torch.zeros(2, device=device)
        for batch_d in train_loader:
            img = batch_d["image"].to(device)
            heat_t = batch_d["heat"].to(device)          # (B,2,27,48)
            out = model(img)
            l_center = center_focal_loss(out["heat_logits"], heat_t,
                                         pos_weight=pos_w)
            s4 = F.avg_pool2d(out["smooth"], 4)          # (B,3,135,240)
            obj = heat_t.max(1, keepdim=True).values
            bg = (F.interpolate(obj, scale_factor=5, mode="nearest")
                  < 0.05).float()
            tv = (s4 - F.avg_pool2d(s4, 3, 1, 1)).abs()
            l_smooth = (tv * bg).sum() / bg.sum().clamp_min(1.0) / 3.0
            loss = l_center + smooth_w * l_smooth
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            nb += 1
            tot += loss.detach()
            parts += torch.stack([l_center.detach(), l_smooth.detach()])
            if nb % log_every == 0:
                r = nb * batch / (time.time() - t0)
                pc, ps = (parts / nb).tolist()
                print(f"  ep {ep} step {nb}/{n_steps}: "
                      f"loss={float(tot) / nb:.4f} (center {pc:.3f} "
                      f"smooth {ps:.3f}, {r:.0f} img/s)", flush=True)
        tot = float(tot)
        s = evaluate(model, val_loader, device, args.peak_thresh)
        flag = ""
        if s["sel"] > best:
            best = s["sel"]
            torch.save({"model": model.state_dict(), "arch": "v21.2",
                        "epoch": ep, "val": s}, OUT / "best.pt")
            flag = "  *best*"
        torch.save({"model": model.state_dict(), "arch": "v21.2",
                    "epoch": ep, "val": s}, OUT / "last.pt")
        print(f"epoch {ep:3d}: loss={tot / max(nb, 1):.4f} "
              f"mann_r={s.get('mannequin_recall', 0):.3f} "
              f"tent_r={s.get('tent_recall', 0):.3f} "
              f"fp={s.get('fp_per_image', 0):.3f} sel={s['sel']:.3f} "
              f"({time.time() - t0:.0f}s){flag}", flush=True)
    print(f"done — best sel {best:.3f} -> {OUT / 'best.pt'}")


if __name__ == "__main__":
    main()
