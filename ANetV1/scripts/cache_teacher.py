"""Cache YOLO teacher soft grids for distillation (run after train_yolo.py).

    python scripts/cache_teacher.py --weights runs/yolo/baseline/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet.distill.teacher import cache_teacher  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.trainer import yolo_device  # noqa: E402

cfg = anet_cfg()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="trained YOLO .pt (runs/yolo/baseline/weights/best.pt)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    for split in args.splits:
        cache_teacher(
            args.weights, cfg.data.root, split,
            Path(cfg.distill.teacher_cache) / split,
            imgsz=cfg.yolo.imgsz, device=yolo_device(),
        )


if __name__ == "__main__":
    main()
