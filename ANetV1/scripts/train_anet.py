"""Experiment 2: ANetV1 from scratch.

No yaml, no flags — edit the cfg block below and run:
    python scripts/train_anet.py
Device-aware defaults (MI300X vs Mac) come from anet/train/presets.py.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.data.dataset import SUASCells  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import Trainer  # noqa: E402

# --------------------------------------------------------------------------
# EDIT HERE — anything not listed keeps its preset default.
# To resume/fine-tune from a checkpoint, set init_from below (or export
# ANET_INIT_FROM=runs/anet/last.pt). Resume warm-starts the weights, skips
# warmup, and starts a fresh cosine over `epochs` — lower epochs/lr to fine-tune.
# --------------------------------------------------------------------------
cfg = anet_cfg(
    hidden=24,               # 16 = 17,037-param spec model; 24 = capacity bump (~24k)
    checkpoint_dir="runs/anet",
    # init_from="runs/anet/last.pt",   # uncomment to resume, or use ANET_INIT_FROM
)


def build_datasets(cfg, teacher_dir=None):
    kwargs = dict(
        coverage_thresh=cfg.data.coverage_thresh,
        vd_weight=cfg.data.vd_weight,
        mannequin_weight=cfg.data.mannequin_weight,
        tent_weight=cfg.data.tent_weight,
        uint8=getattr(cfg.data, "uint8", False),  # Trainer normalizes on-GPU
        band_lo=getattr(cfg.data, "band_lo", None),  # boundary ignore band for the loss
        cache=getattr(cfg.data, "cache", False),  # memmap preprocessing cache
    )
    train = SUASCells(cfg.data.root, "train", teacher_dir=teacher_dir, **kwargs)
    val = SUASCells(cfg.data.root, "val", **kwargs)
    return train, val


def main():
    train_ds, val_ds = build_datasets(cfg)
    init = getattr(cfg.train, "init_from", None)
    if init:
        sd = torch.load(init, map_location="cpu")
        model = ANetV1.from_state_dict(sd, use_checkpoint=cfg.train.use_checkpoint)
        # keep a SHORT warmup + lower peak lr: hitting a converged model with the
        # full 4e-3 immediately made it thrash (mannequin 0.08<->0.63, fp 0.3<->28)
        cfg.train.warmup_steps = 100
        cfg.train.lr = min(cfg.train.lr, 2.0e-3)
        print(f"RESUMING from {init} (warm start: lr={cfg.train.lr}, warmup=100)")
    else:
        model = ANetV1(use_checkpoint=cfg.train.use_checkpoint, hidden=cfg.train.hidden,
                       stem=cfg.train.stem,
                       path_a_per_channel=cfg.train.path_a_per_channel)
    n_params = sum(p.numel() for p in model.parameters())
    tw = cfg.train.tversky_weight
    print(f"ANetV1: {n_params:,} params (hidden={model.encoder.hidden}, stem={model.stem}) | "
          f"tversky_w={tw} | train {len(train_ds)} | val {len(val_ds)} | data {cfg.data.root}")
    Trainer(model, train_ds, val_ds, cfg).train()


if __name__ == "__main__":
    main()
