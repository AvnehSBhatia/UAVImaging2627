"""Linear probe: is mannequin signal present in the window embeddings at all?

Extracts per-window features from a checkpoint (own embedding + 3 Path-A
context vectors — everything window-specific the head sees) and trains a
small logistic regression on window labels. Localizes the failure:

  probe recall HIGH  -> encoder carries the signal; head/global-mix loses it
  probe recall ~ 0   -> Stage 1 never encodes mannequin-vs-clutter (capacity/design)

  python scripts/probe.py --ckpt runs/anet/last.pt [--n 300]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.data.dataset import SUASCells  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import pick_device  # noqa: E402


@torch.no_grad()
def window_features(model, img):
    """(B,3,540,960) -> feats (B, W, 4*d): own embedding + 3 Path-A vectors."""
    m16 = model._map_dense(model._features(img))
    m = torch.cat([m16, model.xy_map.expand(img.shape[0], -1, -1, -1)], 1)
    maps = [p(m) for p in model.pools]
    per_win = [t.flatten(2).permute(0, 2, 1) for t in (m, *maps)]  # 4 x (B,W,d)
    return torch.cat(per_win, -1)


def window_labels(grid):
    """(B,54,96) cell labels -> (B,53*95) window labels via the 2x2 cells each
    window covers; foreground class wins over background, mannequin over tent."""
    g = grid.unfold(1, 2, 1).unfold(2, 2, 1)  # (B,53,95,2,2)
    flat = g.reshape(*g.shape[:3], 4)
    has_m = (flat == 1).any(-1)
    has_t = (flat == 2).any(-1)
    lab = torch.zeros(g.shape[:3], dtype=torch.long)
    lab[has_t] = 2
    lab[has_m] = 1  # mannequin wins ties — it's the class under investigation
    return lab.reshape(grid.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="required unless --raw (raw probing bypasses the model)")
    ap.add_argument("--n", type=int, default=300, help="synthetic val images")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--per-image-z", action="store_true",
                    help="z-score each image's windows against that image's own "
                         "stats (tests mannequin-as-local-outlier; global z is default)")
    ap.add_argument("--raw", action="store_true",
                    help="probe RAW 20x20x3 window pixels instead of embeddings — "
                         "answers 'is the signal even in the pixels at 540p?'")
    ap.add_argument("--edges", action="store_true",
                    help="probe RAW pixels + 7x7 oriented Sobel edge maps (9ch) — "
                         "tests the edge-stem hypothesis before building/training it")
    args = ap.parse_args()
    if args.raw:
        args.n = min(args.n, 120)  # 1200-d features; cap RAM at ~3GB
    if args.edges:
        args.n = min(args.n, 30)   # 3600-d features; cap RAM
    cfg = anet_cfg()
    device = pick_device()

    model, hidden = None, None
    if not (args.raw or args.edges):
        if not args.ckpt:
            raise SystemExit("--ckpt is required unless --raw/--edges")
        sd = torch.load(args.ckpt, map_location=device)
        model = ANetV1.from_state_dict(sd, use_checkpoint=False).to(device)
        model.eval()
        hidden = f"{model.encoder.hidden} stem={model.stem}"

    ds = SUASCells(cfg.data.root, "val", coverage_thresh=cfg.data.coverage_thresh)
    idx = [i for i in range(len(ds)) if not ds.is_visdrone(i)][: args.n]
    loader = DataLoader(Subset(ds, idx), batch_size=8, num_workers=2)

    ekv = ekh = None
    if args.edges:
        from anet.model.blocks import sobel7
        ekv = sobel7("v").reshape(1, 1, 7, 7).expand(3, 1, 7, 7).to(device)
        ekh = sobel7("h").reshape(1, 1, 7, 7).expand(3, 1, 7, 7).to(device)

    X, Y = [], []
    for batch in loader:
        img = batch["image"].to(device)
        if args.edges:  # [raw, vert-edge, horiz-edge] 9ch -> unfold (B,W,3600)
            ev = F.conv2d(img, ekv, padding=3, groups=3)
            eh = F.conv2d(img, ekh, padding=3, groups=3)
            f = F.unfold(torch.cat([img, ev, eh], 1), 20, stride=10).permute(0, 2, 1)
        elif args.raw:  # (B, 1200, 5035) -> (B, W, 1200), same window order as the model
            f = F.unfold(img, 20, stride=10).permute(0, 2, 1)
        else:
            f = window_features(model, img)  # (B, W, d)
        if args.per_image_z:
            f = (f - f.mean(1, keepdim=True)) / f.std(1, keepdim=True).clamp_min(1e-6)
        X.append(f.cpu().flatten(0, 1))
        Y.append(window_labels(batch["grid"]).flatten())
    X, Y = torch.cat(X), torch.cat(Y)
    if args.edges:
        print("probing RAW pixels + 7x7 oriented Sobel edges (9ch, encoder bypassed)")
    elif args.raw:
        print("probing RAW window pixels (encoder bypassed)")
    if args.per_image_z:
        print("per-image z-scoring: ON (each window relative to its own image)")
    d = X.shape[1]
    counts = torch.bincount(Y, minlength=3)
    src = "RAW+EDGES" if args.edges else "RAW PIXELS" if args.raw else f"ckpt {args.ckpt} | hidden={hidden}"
    print(f"{src} | {len(Y):,} windows "
          f"(bg={counts[0]:,} mannequin={counts[1]:,} tent={counts[2]:,}) | feat dim {d}")

    # split, normalize, train weighted logistic regression on GPU
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(Y), generator=g)
    cut = int(0.8 * len(Y))
    tr, te = perm[:cut], perm[cut:]
    mu, sd_ = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    Xn = ((X - mu) / sd_).to(device)
    Yd = Y.to(device)
    w = (len(Y) / (3.0 * counts.clamp_min(1))).to(device)  # balanced class weights

    lin = torch.nn.Linear(d, 3).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for step in range(args.steps):
        sub = tr[torch.randint(len(tr), (8192,), generator=g)].to(device)
        loss = F.cross_entropy(lin(Xn[sub]), Yd[sub], weight=w)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = lin(Xn[te.to(device)]).argmax(1).cpu()
    yt = Y[te]
    print("\nlinear probe on held-out windows (balanced-weight training):")
    for k, name in ((1, "mannequin"), (2, "tent")):
        tp = int(((pred == k) & (yt == k)).sum())
        fn = int(((pred != k) & (yt == k)).sum())
        fp = int(((pred == k) & (yt != k)).sum())
        r = tp / max(tp + fn, 1)
        p = tp / max(tp + fp, 1)
        flag = (tp + fp) / max(len(yt), 1)  # a random classifier's recall == its flag rate
        lift = r / max(flag, 1e-9)
        print(f"  {name:>9}: recall={r:.3f} @ flag rate {flag:.3f} -> lift x{lift:.2f} "
              f"(random = x1.00) precision={p:.3f} (tp={tp:,} fn={fn:,} fp={fp:,})")
    print("\nread (use a LATE checkpoint; tent is the control — it reaches ~0.8 object "
          "recall, so its lift shows what 'signal present' looks like):\n"
          "  mannequin lift ~ tent lift  -> encoder carries it; head/global-mix loses it\n"
          "  mannequin lift ~ x1, tent high -> encoder never separates mannequin from clutter")


if __name__ == "__main__":
    main()
