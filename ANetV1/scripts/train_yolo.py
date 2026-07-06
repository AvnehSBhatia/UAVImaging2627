"""Experiment 1: YOLO26n baseline (falls back to YOLO11n) on the SUAS dataset."""

import argparse
import gc
import os
import sys
from pathlib import Path

# MPS graph-cache leak mitigations (torch 2.12 has no clear_graph_cache API yet).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anet.config import load_config  # noqa: E402
from anet.train.trainer import yolo_device  # noqa: E402

_LOG_EVERY = 50
_batch_count = 0


def _mps_clear():
    """Best-effort MPS memory reclaim; graph cache only clearable on torch >= 2.13."""
    gc.collect()
    torch.mps.empty_cache()
    clear_graph = getattr(torch.mps, "clear_graph_cache", None)
    if clear_graph is not None:
        clear_graph()


def _mps_periodic(trainer):
    """Clear MPS cache every batch — graph cache grows with varying mosaic shapes."""
    global _batch_count
    if trainer.device.type != "mps":
        return
    _batch_count += 1
    _mps_clear()
    if _batch_count % _LOG_EVERY == 0:
        gb = torch.mps.driver_allocated_memory() / 1e9
        print(f"[mps] batch {_batch_count} driver_allocated={gb:.2f} GB")


def _mps_log(trainer):
    """Log MPS driver memory after each fit epoch so drift is visible early."""
    if trainer.device.type != "mps":
        return
    _mps_clear()
    gb = torch.mps.driver_allocated_memory() / 1e9
    print(f"[mps] epoch {trainer.epoch + 1} driver_allocated={gb:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/anet.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    from ultralytics import YOLO

    try:
        model = YOLO(cfg.yolo.weights)
    except Exception as e:  # yolo26n not in this ultralytics version yet
        print(f"{cfg.yolo.weights} unavailable ({e}); falling back to yolo11n.pt")
        model = YOLO("yolo11n.pt")

    device = yolo_device()
    if device == "mps":  # leak mitigations only needed (and only valid) on MPS
        model.add_callback("on_train_batch_end", _mps_periodic)
        model.add_callback("on_fit_epoch_end", _mps_log)

    anet_root = Path(__file__).resolve().parents[1]
    project = anet_root / cfg.yolo.project
    project.mkdir(parents=True, exist_ok=True)

    model.train(
        data=str(Path(cfg.data.root) / "data.yaml"),
        imgsz=cfg.yolo.imgsz,
        epochs=cfg.yolo.epochs,
        patience=getattr(cfg.yolo, "patience", 100),  # ultralytics native early stop
        batch=cfg.yolo.batch,
        device=device,
        project=str(project),
        name="baseline",
        exist_ok=True,  # resume-safe: never silently forks baseline2/
        cache=False,
        workers=cfg.yolo.workers,
        amp=cfg.yolo.amp,
    )


if __name__ == "__main__":
    main()
