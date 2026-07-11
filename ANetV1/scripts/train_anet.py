"""Experiment 2: ANetV1 from scratch — v9 architecture (see ARCHITECTURE.md
section 14 and V9_CHANGES.md).

No yaml, no flags — edit the cfg block below and run:
    python scripts/train_anet.py
Device-aware defaults (MI300X vs Mac) come from anet/train/presets.py.

What v9 changes (summary; full rationale in the docs):
  - DeployNorm (D39): training normalizes with the same running-stat affine
    the deploy graph uses -> the encoder is tile-local and fusable.
  - Fused Triton Stage-1 (D40): the whole per-token encoder runs in one
    kernel per direction; parity-checked at startup against the reference
    path, with automatic demotion (triton bwd -> chunked-autograd bwd ->
    PyTorch dense at a VRAM-safe batch).
  - Sobel-init 4-orientation stem (D41), fc2 after the pool (D42), ConvNeck
    cross-window context (D43), SlimContext (D44, Path-B 256-d expansions
    removed), 24-wide head (D45), aux deep-supervision probe (D46).
  - focal_norm loss (D47): one smooth per-cell term, per-class positive-
    normalized. No Tversky/anchor tug-of-war, no limit cycles.
  - Weight EMA for eval + checkpoints (D48).

Env overrides (all optional): ANET_BATCH, ANET_ACCUM, ANET_LR, ANET_WARMUP,
ANET_EPOCHS, ANET_COMPILE, ANET_FUSED, ANET_FUSED_BWD, ANET_CACHE,
ANET_NUM_WORKERS, ANET_PRIOR_FG, ANET_CONF, ANET_INIT_FROM, ANET_PATIENCE,
ANET_MIN_EP, ANET_LOSS_MODE, DATA_ROOT.
"""

import os
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
# To resume/fine-tune from a v9 checkpoint, set init_from below (or export
# ANET_INIT_FROM=runs/anet/last.pt). v8 checkpoints (e.g. runs/anet/good.pt)
# still load through from_state_dict for evaluation, but cannot warm-start a
# v9 model (different encoder layout).
# --------------------------------------------------------------------------
cfg = anet_cfg(
    arch="v9",
    stem="edge_dq4",         # 4-orientation Sobel-init edge stem (D41)
    hidden=32,               # embedding width (v9 default; ~21k params total)
    epochs=int(os.environ.get("ANET_EPOCHS", 40)),
    # 1.5e-3 peak (was 3e-3): with the v10 loss fix removing the oscillation
    # forcing function, LR is just the step-amplitude knob — and the cosine sits
    # at ~100% of peak for the first ~8 epochs (stretched over 40), so a hot peak
    # meant every early step took a full-amplitude swing at the argmax boundary.
    lr=float(os.environ["ANET_LR"]) if "ANET_LR" in os.environ else 1.5e-3,
    prior_fg=(float(os.environ["ANET_PRIOR_FG"]) or None)
    if "ANET_PRIOR_FG" in os.environ else 0.05,
    checkpoint_dir="runs/anet",
    # init_from="runs/anet/last.pt",   # uncomment to resume, or ANET_INIT_FROM
)


def build_datasets(cfg, teacher_dir=None):
    kwargs = dict(
        coverage_thresh=cfg.data.coverage_thresh,
        vd_weight=cfg.data.vd_weight,
        mannequin_weight=cfg.data.mannequin_weight,
        tent_weight=cfg.data.tent_weight,
        uint8=getattr(cfg.data, "uint8", False),  # Trainer normalizes on-GPU
        band_lo=getattr(cfg.data, "band_lo", None),  # boundary ignore band
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
        if model.arch != "v9":
            raise SystemExit(
                f"{init} is a {model.arch} checkpoint — v9 cannot warm-start "
                "from it (different encoder). Remove init_from to train from "
                "scratch, or evaluate it with scripts/evaluate_all.py instead.")
        # gentle warm start unless the caller explicitly asked for more
        if "ANET_WARMUP" not in os.environ:
            cfg.train.warmup_steps = 100
        if "ANET_LR" not in os.environ:
            cfg.train.lr = min(cfg.train.lr, 1.5e-3)
        print(f"RESUMING from {init} (warm start: lr={cfg.train.lr}, "
              f"warmup={cfg.train.warmup_steps})")
    else:
        model = ANetV1(
            arch="v9",
            use_checkpoint=cfg.train.use_checkpoint,
            dense=True,
            hidden=cfg.train.hidden,
            h1=cfg.train.h1,
            stem=cfg.train.stem,
            neck_rounds=cfg.train.neck_rounds,
            head_width=cfg.train.head_width,
            aux_head=cfg.train.aux_head,
            path_a_per_channel=cfg.train.path_a_per_channel,
            prior_fg=getattr(cfg.train, "prior_fg", None),
        )
    n_params = sum(p.numel() for p in model.parameters())
    n_aux = model.aux.weight.numel() + model.aux.bias.numel() \
        if getattr(model, "aux", None) is not None else 0
    print(f"ANetV1 {model.arch}: {n_params:,} params "
          f"({n_params - n_aux:,} deployed + {n_aux} aux) | "
          f"hidden={model.encoder.hidden} h1={model.encoder.h1} "
          f"stem={model.stem} | "
          f"loss={cfg.train.loss_mode} lr={cfg.train.lr} "
          f"epochs={cfg.train.epochs} | "
          f"train {len(train_ds)} | val {len(val_ds)} | data {cfg.data.root}")
    assert n_params < 40_000, "param budget exceeded (must stay under 40k)"
    Trainer(model, train_ds, val_ds, cfg).train()


if __name__ == "__main__":
    main()
