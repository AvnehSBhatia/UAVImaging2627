"""Experiment 2: ANetV1 from scratch.

No yaml, no flags — edit the cfg block below and run:
    python scripts/train_anet.py
Device-aware defaults (MI300X vs Mac) come from anet/train/presets.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.data.dataset import SUASCells  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import Trainer  # noqa: E402

# --------------------------------------------------------------------------
# EDIT HERE — anything not listed keeps its preset default
# --------------------------------------------------------------------------
cfg = anet_cfg(
    hidden=24,               # 16 = 17,037-param spec model; 24 = capacity bump (~24k)
    checkpoint_dir="runs/anet",
)


def build_datasets(cfg, teacher_dir=None):
    kwargs = dict(
        coverage_thresh=cfg.data.coverage_thresh,
        vd_weight=cfg.data.vd_weight,
        mannequin_weight=cfg.data.mannequin_weight,
        tent_weight=cfg.data.tent_weight,
    )
    train = SUASCells(cfg.data.root, "train", teacher_dir=teacher_dir, **kwargs)
    val = SUASCells(cfg.data.root, "val", **kwargs)
    return train, val


def main():
    train_ds, val_ds = build_datasets(cfg)
    model = ANetV1(use_checkpoint=cfg.train.use_checkpoint, hidden=cfg.train.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ANetV1: {n_params:,} params (hidden={cfg.train.hidden}) | "
          f"train {len(train_ds)} | val {len(val_ds)} | data {cfg.data.root}")
    Trainer(model, train_ds, val_ds, cfg).train()


if __name__ == "__main__":
    main()
