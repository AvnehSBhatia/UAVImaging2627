"""Why is a class at zero recall? Loads a checkpoint, runs synthetic val, and
reports per-class logit statistics, prediction counts, and near-miss analysis.

  python scripts/diagnose.py --ckpt runs/anet/last.pt [--n 200]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.data.dataset import SUASCells  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402

CLS = ("background", "mannequin", "tent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200, help="synthetic val images to scan")
    args = ap.parse_args()
    cfg = anet_cfg()
    device = pick_device()

    sd = torch.load(args.ckpt, map_location=device)
    model = ANetV1.from_state_dict(sd, use_checkpoint=False).to(device)
    model.eval()
    hidden = model.encoder.hidden
    print(f"ckpt {args.ckpt} | hidden={hidden} stem={model.stem} | "
          f"params={sum(p.numel() for p in model.parameters()):,} | device={device}")

    ds = SUASCells(cfg.data.root, "val", coverage_thresh=cfg.data.coverage_thresh)
    idx = [i for i in range(len(ds)) if not ds.is_visdrone(i)][: args.n]
    loader = DataLoader(Subset(ds, idx), batch_size=8, num_workers=2)

    conf = np.zeros((3, 3), np.int64)          # gt x pred cells
    logit_sum = torch.zeros(3, 3, dtype=torch.float64)  # gt -> mean logits
    logit_n = torch.zeros(3, dtype=torch.float64)
    prob_at_gt = {1: [], 2: []}                 # softmax P(cls) at that class's GT cells
    rank_at_gt = {1: np.zeros(3, np.int64), 2: np.zeros(3, np.int64)}  # winner at GT cells

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).float().cpu()  # (B,3,54,96)
            pred = logits.argmax(1)
            gt = batch["grid"]
            probs = torch.softmax(logits, 1)
            for g in range(3):
                mask = gt == g
                if mask.any():
                    sel = logits.permute(0, 2, 3, 1)[mask]  # (n,3)
                    logit_sum[g] += sel.double().sum(0)
                    logit_n[g] += mask.sum()
            idxs = (gt.reshape(-1) * 3 + pred.reshape(-1)).numpy()
            conf += np.bincount(idxs, minlength=9).reshape(3, 3)
            for cls in (1, 2):
                m = gt == cls
                if m.any():
                    prob_at_gt[cls].append(probs.permute(0, 2, 3, 1)[m][:, cls])
                    win = pred[m]
                    for k in range(3):
                        rank_at_gt[cls][k] += int((win == k).sum())

    print("\ncell confusion (rows=GT, cols=pred):")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in CLS))
    for g in range(3):
        print(f"{CLS[g]:>12}" + "".join(f"{conf[g, k]:>12,}" for k in range(3)))

    print("\nmean logits by GT cell class (what the head outputs there):")
    for g in range(3):
        if logit_n[g]:
            m = (logit_sum[g] / logit_n[g]).tolist()
            print(f"  GT {CLS[g]:>10}: bg={m[0]:+.2f} mannequin={m[1]:+.2f} tent={m[2]:+.2f}")

    print()
    for cls in (1, 2):
        name = CLS[cls]
        if prob_at_gt[cls]:
            p = torch.cat(prob_at_gt[cls])
            q = torch.quantile(p, torch.tensor([0.5, 0.9, 0.99, 1.0]))
            r = rank_at_gt[cls]
            total_pred = int(conf[:, cls].sum())
            print(f"P({name}) at {len(p):,} GT {name} cells: "
                  f"median={q[0]:.4f} p90={q[1]:.4f} p99={q[2]:.4f} max={q[3]:.4f}")
            print(f"  winner at those cells: bg={r[0]:,} mannequin={r[1]:,} tent={r[2]:,}")
            print(f"  verdict: predicts {total_pred:,} {name} cells anywhere -> "
                  f"{'TRUE COLLAPSE (never wins argmax)' if total_pred == 0 else 'not collapsed'}\n")


if __name__ == "__main__":
    main()
