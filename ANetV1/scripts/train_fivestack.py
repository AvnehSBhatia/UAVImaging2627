"""P2 probe trainer: FiveStack — 5-filter bank + 10x10 window scorer.

Run from inside ANetV1/. Follows the family train protocol: ANET_* env
knobs, device pick, per-epoch val, best.pt on the selection score. Results
ledger: OBSERVATIONS.md P2.

  python scripts/train_fivestack.py            # train
  python scripts/train_fivestack.py --smoke    # fwd/bwd + dataset sanity
  python scripts/train_fivestack.py --eval runs/fivestack/best.pt [--split test]

Selection: mannequin recall + 0.5*tent recall (the family's sel), GATED to
-1 when background accuracy drops below 0.8 — the max_sel_fp lesson: a
predict-everything probe must not win the checkpoint.
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

from anet.probes import FiveStack, PatchCrops, collate_patches  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "../datasets/suas-synth-50k")
OUT = Path("runs/fivestack")


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
    hit = {0: 0, 1: 0, 2: 0}
    tot = {0: 0, 1: 0, 2: 0}
    for batch in loader:
        for size, (imgs, _, labels) in batch.items():
            pred = model(imgs.to(device)).argmax(1).cpu()
            for c in (0, 1, 2):
                sel = labels == c
                tot[c] += int(sel.sum())
                hit[c] += int((pred[sel] == c).sum())
    r = {c: hit[c] / max(tot[c], 1) for c in (0, 1, 2)}
    sel = r[1] + 0.5 * r[2] if r[0] >= 0.8 else -1.0
    return {"bg_acc": r[0], "mann_r": r[1], "tent_r": r[2], "sel": sel,
            "n": dict(tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval", metavar="CKPT")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = pick_device()
    model = FiveStack(embed=int(os.environ.get("ANET_CH", "16"))).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"FiveStack: {n_par} params, device={device}")

    if args.smoke:
        for size in (40, 100):
            x = torch.rand(2, 3, size, size, device=device)
            y = model(x)
            assert y.shape == (2, 3) and torch.isfinite(y).all()
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
        print(f"{args.split}: bg_acc={m['bg_acc']:.3f} mann_r={m['mann_r']:.3f} "
              f"tent_r={m['tent_r']:.3f} sel={m['sel']:.3f} (n={m['n']})")
        return

    lr = float(os.environ.get("ANET_LR", "1.5e-3"))
    epochs = int(os.environ.get("ANET_EPOCHS", "15"))
    batch = int(os.environ.get("ANET_BATCH", "12"))  # images per batch
    limit = int(os.environ.get("ANET_SAMPLES", "0"))
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
    # bg crops outnumber objects ~2:1 per image and per class; mild reweight
    cls_w = torch.tensor([0.5, 1.0, 1.0], device=device)

    OUT.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch_d in train_loader:
            if not batch_d:
                continue
            loss = 0.0
            for size, (imgs, _, labels) in batch_d.items():
                logits = model(imgs.to(device))
                loss = loss + F.cross_entropy(logits, labels.to(device),
                                              weight=cls_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        m = evaluate(model, val_loader, device)
        flag = ""
        if m["sel"] > best:
            best = m["sel"]
            torch.save({"model": model.state_dict(), "probe": "fivestack",
                        "epoch": ep, "val": m}, OUT / "best.pt")
            flag = "  *best*"
        torch.save({"model": model.state_dict(), "probe": "fivestack",
                    "epoch": ep, "val": m}, OUT / "last.pt")
        print(f"epoch {ep:3d}: loss={tot / max(nb, 1):.4f} "
              f"bg_acc={m['bg_acc']:.3f} mann_r={m['mann_r']:.3f} "
              f"tent_r={m['tent_r']:.3f} sel={m['sel']:.3f} "
              f"({time.time() - t0:.0f}s){flag}", flush=True)
    print(f"done — best sel {best:.3f} -> {OUT / 'best.pt'}")


if __name__ == "__main__":
    main()
