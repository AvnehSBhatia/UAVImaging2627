"""Experiment 3: ANetV1 distilled from the YOLO teacher (run after cache_teacher.py)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.config import load_config  # noqa: E402
from anet.train.trainer import Trainer  # noqa: E402
from train_anet import build_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/anet.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.train.checkpoint_dir = cfg.distill.checkpoint_dir

    teacher_dir = Path(cfg.distill.teacher_cache) / "train"
    if not teacher_dir.is_dir():
        raise FileNotFoundError(f"{teacher_dir} — run cache_teacher.py first")

    train_ds, val_ds = build_datasets(cfg, teacher_dir=teacher_dir)
    model = ANetV1(use_checkpoint=getattr(cfg.train, "use_checkpoint", True),
                   hidden=getattr(cfg.train, "hidden", 16))
    Trainer(model, train_ds, val_ds, cfg, distill=True).train()


if __name__ == "__main__":
    main()
