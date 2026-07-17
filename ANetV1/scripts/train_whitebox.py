"""P1 probe trainer: WhiteboxDQ — colour algebra paints GT boxes white.

Run from inside ANetV1/. Follows the family train protocol: ANET_* env
knobs, device pick, per-epoch val, best.pt on the selection score. Results
ledger: OBSERVATIONS.md P1.

  python scripts/train_whitebox.py            # train
  python scripts/train_whitebox.py --smoke    # fwd/bwd + dataset sanity
  python scripts/train_whitebox.py --eval runs/whitebox/best.pt [--split test]

Selection: mean IoU@0.5 over object crops MINUS the white-pixel fraction on
background crops (a probe that paints everything white scores ~0, matching
the max_sel_fp spirit — over-predictors must not win the checkpoint).
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

from anet.probes import PatchCrops, WhiteboxDQ, collate_patches  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "../datasets/suas-synth-50k")
OUT = Path("runs/whitebox")


def _loaders(batch, limit, val_limit, include_vd, cache):
    workers = 0 if torch.version.hip else 2
    tr = PatchCrops(ROOT, "train", include_vd, limit=limit, cache=cache)
    va = PatchCrops(ROOT, "val", include_vd, limit=val_limit)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch, shuffle=sh,
                                   num_workers=workers,
                                   collate_fn=collate_patches)
    return mk(tr, True), mk(va, False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = union = 0.0
    n_obj = 0
    bg_white = bg_px = 0.0
    for batch in loader:
        for size, (imgs, masks, labels, _) in batch.items():
            p = (torch.sigmoid(model(imgs.to(device))[:, 0]) > 0.5).float().cpu()
            obj = labels > 0
            if obj.any():
                po, mo = p[obj], masks[obj]
                inter += (po * mo).sum().item()
                union += ((po + mo) > 0).float().sum().item()
                n_obj += int(obj.sum())
            if (~obj).any():
                bg_white += p[~obj].sum().item()
                bg_px += p[~obj].numel()
    iou = inter / max(union, 1.0)
    bgw = bg_white / max(bg_px, 1.0)
    return {"iou": iou, "bg_white": bgw, "n_obj": n_obj, "sel": iou - bgw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval", metavar="CKPT")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = pick_device()
    model = WhiteboxDQ(stages=int(os.environ.get("ANET_STAGES", "4"))).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"WhiteboxDQ: {n_par} params, device={device}")

    if args.smoke:
        for size in (40, 100):
            x = torch.rand(2, 3, size, size, device=device)
            y = model(x)
            assert y.shape == (2, 1, size, size) and torch.isfinite(y).all()
            y.mean().backward()
        ds = PatchCrops(ROOT, "val", limit=3)
        b = collate_patches([ds[i] for i in range(len(ds))])
        assert b, "no crops from 3 val images"
        print("PASS —", {k: tuple(v[0].shape) for k, v in b.items()})
        return

    if args.eval:
        sd = torch.load(args.eval, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        ds = PatchCrops(ROOT, args.split,
                        include_vd=os.environ.get("ANET_VD") == "1")
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_patches)
        m = evaluate(model, loader, device)
        print(f"{args.split}: iou={m['iou']:.3f} bg_white={m['bg_white']:.4f} "
              f"sel={m['sel']:.3f} (n_obj={m['n_obj']})")
        return

    lr = float(os.environ.get("ANET_LR", "5e-3"))
    epochs = int(os.environ.get("ANET_EPOCHS", "15"))
    batch = int(os.environ.get("ANET_BATCH", "12"))  # images per batch
    limit = int(os.environ.get("ANET_SAMPLES", "0"))
    pos_w = float(os.environ.get("ANET_POS_W", "2.0"))
    train_loader, val_loader = _loaders(
        batch, limit, val_limit=800,
        include_vd=os.environ.get("ANET_VD") == "1",
        cache=os.environ.get("ANET_CACHE") == "1")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(len(train_loader) * epochs, 1)
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    pw = torch.tensor(pos_w, device=device)

    OUT.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch_d in train_loader:
            if not batch_d:
                continue
            loss = 0.0
            for size, (imgs, masks, _, _) in batch_d.items():
                logits = model(imgs.to(device))[:, 0]
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits, masks.to(device), pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        m = evaluate(model, val_loader, device)
        flag = ""
        if m["sel"] > best:
            best = m["sel"]
            torch.save({"model": model.state_dict(), "probe": "whitebox",
                        "epoch": ep, "val": m}, OUT / "best.pt")
            flag = "  *best*"
        torch.save({"model": model.state_dict(), "probe": "whitebox",
                    "epoch": ep, "val": m}, OUT / "last.pt")
        print(f"epoch {ep:3d}: loss={tot / max(nb, 1):.4f} "
              f"iou={m['iou']:.3f} bg_white={m['bg_white']:.4f} "
              f"sel={m['sel']:.3f} ({time.time() - t0:.0f}s){flag}",
              flush=True)
    print(f"done — best sel {best:.3f} -> {OUT / 'best.pt'}")


if __name__ == "__main__":
    main()
