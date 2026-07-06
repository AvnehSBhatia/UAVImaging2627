"""Experiment 2: ANetV1 from scratch."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.config import load_config  # noqa: E402
from anet.data.dataset import SUASCells  # noqa: E402
from anet.train.trainer import Trainer  # noqa: E402


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/anet.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    train_ds, val_ds = build_datasets(cfg)
    model = ANetV1(use_checkpoint=getattr(cfg.train, "use_checkpoint", True))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ANetV1: {n_params:,} params | train {len(train_ds)} | val {len(val_ds)}")
    Trainer(model, train_ds, val_ds, cfg).train()


if __name__ == "__main__":
    main()
