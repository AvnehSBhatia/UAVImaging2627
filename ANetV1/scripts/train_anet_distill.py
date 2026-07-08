"""Experiment 3: ANetV1 distilled from the YOLO teacher (run cache_teacher.py first).

No yaml, no flags — edit the cfg block below and run:
    python scripts/train_anet_distill.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import Trainer  # noqa: E402
from train_anet import build_datasets  # noqa: E402

# --------------------------------------------------------------------------
# EDIT HERE — keep `hidden` in sync with train_anet.py for a fair comparison
# --------------------------------------------------------------------------
cfg = anet_cfg(
    hidden=24,
    checkpoint_dir="runs/anet_distill",
    kl_weight=0.7,           # 0.7 KL(teacher) + 0.3 focal(hard labels), T=2 (D27)
    temperature=2.0,
)


def main():
    teacher_dir = Path(cfg.distill.teacher_cache) / "train"
    if not teacher_dir.is_dir():
        raise FileNotFoundError(f"{teacher_dir} — run cache_teacher.py first")

    train_ds, val_ds = build_datasets(cfg, teacher_dir=teacher_dir)
    model = ANetV1(use_checkpoint=cfg.train.use_checkpoint, hidden=cfg.train.hidden,
                   stem=cfg.train.stem,
                   path_a_per_channel=cfg.train.path_a_per_channel)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ANetV1-distill: {n_params:,} params (hidden={cfg.train.hidden}, stem={cfg.train.stem}) | "
          f"train {len(train_ds)} | val {len(val_ds)} | teacher {teacher_dir}")
    Trainer(model, train_ds, val_ds, cfg, distill=True).train()


if __name__ == "__main__":
    main()
