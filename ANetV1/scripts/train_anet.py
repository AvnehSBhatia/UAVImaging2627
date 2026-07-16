"""Experiment 2: ANetV1 from scratch — currently v13, the conv-backbone
center-heatmap detector (ARCHITECTURE.md section 15 / D58; v12's window-token
variant and v9 region-classification remain available — v9's cfg is commented
below, see ARCHITECTURE.md section 14 and V9_CHANGES.md for its design).

No yaml, no flags — edit the cfg block below and run:
    python scripts/train_anet.py
Device-aware defaults (MI300X vs Mac) come from anet/train/presets.py.

What v13 changes vs v12 (current default): the whole encoder/neck/Path-A/
context/token-head stack is replaced by one plain multi-scale conv pyramid
(anet/model/backbone.py, D58) with the same {"heat","offset"} output contract.

What v12 changed vs v9 (v12 kept the same Stage-1 encoder/neck/Path-A/
context as v9, D39-D45 below applied to it):
  - Single-phase stride-20 Stage-1 (drop v9's 4x overlap) -> a 27x48 grid
    matching one cell per 20x20 tile, not v9's overlap-averaged 54x96.
  - CenterHead: two INDEPENDENT per-class sigmoids (mannequin, tent — no
    softmax competition) over a heatmap + a class-agnostic (dx,dy) sub-cell
    offset, instead of RegionHeadV9's per-cell 3-way softmax. No aux probe,
    no metric-prototype path.
  - center_focal_loss (CenterNet penalty-reduced focal) + offset_weight *
    offset_l1, instead of the v9 focal_norm/fp_tp cell losses — see
    trainer.py's loss_mode=="center" branch.
  - CenterObjectMetrics (peak/object-only; no per-cell confusion table) for
    eval and best.pt selection.

What v9 changes vs v8 (summary; full rationale in the docs):
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
# To resume/fine-tune from a checkpoint matching the arch below (v13 by
# default), set init_from below (or export ANET_INIT_FROM=runs/anet/last.pt).
# Checkpoints from a DIFFERENT arch (e.g. a v9 or v8 run) still load through
# from_state_dict for evaluation, but cannot warm-start a run here — the
# encoder layouts differ.
# --------------------------------------------------------------------------
cfg = anet_cfg(
    # v9 (region-classification) config — kept available, commented, so v9
    # can still be run by swapping the two blocks below:
    # arch="v9",
    # stem="edge_dq4",         # 4-orientation Sobel-init edge stem (D41)
    # hidden=32,               # embedding width (v9 default; ~21k params total)
    # epochs=int(os.environ.get("ANET_EPOCHS", 40)),
    # # 1.5e-3 peak (was 3e-3): with the v10 loss fix removing the oscillation
    # # forcing function, LR is just the step-amplitude knob — and the cosine
    # # sits at ~100% of peak for the first ~8 epochs (stretched over 40), so a
    # # hot peak meant every early step took a full-amplitude swing at the
    # # argmax boundary.
    # lr=float(os.environ["ANET_LR"]) if "ANET_LR" in os.environ else 1.5e-3,
    # prior_fg=(float(os.environ["ANET_PRIOR_FG"]) or None)
    # if "ANET_PRIOR_FG" in os.environ else 0.05,
    # loss_mode=os.environ.get("ANET_LOSS_MODE") or "fp_tp",
    # checkpoint_dir="runs/anet",
    # # init_from="runs/anet/last.pt",   # uncomment to resume, or ANET_INIT_FROM

    # v13 (conv-backbone center-heatmap) config — current default (D58).
    # Plain multi-scale conv pyramid (model/backbone.py) with the SAME
    # center-heatmap contract, targets, loss and metrics as v12. The window-
    # token encoder is gone: it pooled each 20x20 tile into one vector before
    # any fine-stride learned features existed, which capped object-vs-
    # background embedding separation at ~0.05 and made every from-scratch
    # run crawl (v12 on MI300X: soft p plateaued ~0.09 across two runs at two
    # LRs, ~7k steps). v13 overfits 12 real frames to 19/21 centers past 0.5
    # in 400 steps / 13 s on a Mac; the v12 control also gets there in this
    # small harness but needs 867 s — see ARCHITECTURE.md section 15.2.
    # v14 (D59-D63) — current default: v13 + identity-init structured priors
    # (dw7x7 noise filter, 5x dual-quaternion shift, texture-energy gate,
    # max-pool detail skip, zero-gamma 4th block), each tied to a measured
    # v13 failure mode (ARCHITECTURE.md section 16). ANET_INIT_FROM a v13
    # checkpoint warm-starts v14 to EXACTLY that v13 at step 0.
    arch="v14",
    stem="edge_dq4",         # unused by v13/v14 (kept so v9/v12 swaps stay one-line)
    hidden=32,               # unused by v13/v14 (same reason)
    # from-scratch center-heatmap training is SLOW on the rare tiny mannequin:
    # the first real run climbed soft p(center) only ~0.002/epoch and the 40-epoch
    # cosine decayed LR away mid-climb. The speed levers are center_pos_weight +
    # the widened center_sigma (measured ~10x: soft p 0.004 -> 0.09 by epoch 5),
    # NOT a hotter LR — the 3e-3 peak tried alongside them was unstable: the
    # moment warmup ended, soft p stopped climbing and bounced 0.03-0.09 for 10
    # straight epochs with zero net gain, ON THE EMA WEIGHTS (raw weights swing
    # harder), while at 1.5e-3 the climb was strictly monotonic. Keep the 80-epoch
    # budget so the cosine doesn't decay LR away mid-climb; soft-signal selection
    # keeps early-stop from firing before peaks cross 0.5.
    epochs=int(os.environ.get("ANET_EPOCHS", 80)),
    lr=float(os.environ["ANET_LR"]) if "ANET_LR" in os.environ else 1.5e-3,
    # prior_fg 0.01 (RetinaNet §4.1 prior), NOT 0.1/0.05. On the 27x48=1296-cell
    # heatmap only ~1-2 cells per class are objects, so a HIGH init prior makes
    # the ~2590 background cells' penalty-reduced-focal gradient sink the head's
    # SHARED center bias faster than the deep Tanh-bounded head can lift the
    # object cells — measured in a multi-frame overfit: peaks fell BELOW the init
    # (0.093 -> 0.064) and the model stalled at "predict nothing" (same failure
    # class as the v9 cell collapse). At p_init=0.01 the negative term (~p^2 =
    # 1e-4) barely nudges the bias while the positive -log(0.01) ~ 4.6 strongly
    # lifts the true centers, so localization outruns the bias sink. Isolation
    # confirmed the loss itself is correct (bare-logit optimize -> 16/16 peaks,
    # a 3-conv CNN learns it); the sink was purely the init-prior scale.
    prior_fg=(float(os.environ["ANET_PRIOR_FG"]) or None)
    if "ANET_PRIOR_FG" in os.environ else 0.01,
    loss_mode=os.environ.get("ANET_LOSS_MODE") or "center",
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
        center=True,  # v12: also build heat/offset/reg_mask targets (rasterize.py)
        center_sigma=getattr(cfg.train, "center_sigma", 1.5),  # Gaussian splat width
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
        if model.arch == "v13" and cfg.train.arch == "v14":
            # v14 is a superset of v13 by construction (same module names for
            # the shared trunk; every new module identity-init, D63): copy the
            # v13 weights in and the v14 IS that v13 at step 0 — asserted in
            # smoke_test. Selection can only move away from a proven optimum.
            m14 = ANetV1(arch="v14", use_checkpoint=cfg.train.use_checkpoint,
                         head_width=cfg.train.head_width,
                         prior_fg=getattr(cfg.train, "prior_fg", None))
            missing, unexpected = m14.load_state_dict(model.state_dict(),
                                                      strict=False)
            assert not unexpected, f"v13->v14 transfer: unexpected {unexpected}"
            print(f"v13 -> v14 warm start: {len(model.state_dict())} tensors "
                  f"transferred, {len(missing)} new v14 tensors at identity init")
            model = m14
        elif model.arch != cfg.train.arch:
            raise SystemExit(
                f"{init} is a {model.arch} checkpoint but this script is "
                f"configured for arch={cfg.train.arch!r} — a run cannot "
                "warm-start from a different-arch encoder. Remove init_from "
                "to train from scratch, or evaluate it with "
                "scripts/evaluate_all.py instead.")
        # gentle warm start unless the caller explicitly asked for more
        if "ANET_WARMUP" not in os.environ:
            cfg.train.warmup_steps = 100
        if "ANET_LR" not in os.environ:
            cfg.train.lr = min(cfg.train.lr, 1.5e-3)
        print(f"RESUMING from {init} (warm start: lr={cfg.train.lr}, "
              f"warmup={cfg.train.warmup_steps})")
    else:
        # v14 (current default, D59-D63; v13 = arch swap): the conv backbones
        # ignore hidden/h1/stem/neck/Path-A knobs entirely — the channel plan
        # is fixed in model/backbone.py; only head_width and prior_fg apply.
        model = ANetV1(
            arch=cfg.train.arch,
            use_checkpoint=cfg.train.use_checkpoint,
            head_width=cfg.train.head_width,
            prior_fg=getattr(cfg.train, "prior_fg", None),
        )
        # v12 model construction — swap back alongside a cfg arch="v12":
        # model = ANetV1(
        #     arch="v12",
        #     use_checkpoint=cfg.train.use_checkpoint,
        #     dense=True,
        #     hidden=cfg.train.hidden,
        #     h1=cfg.train.h1,
        #     stem=cfg.train.stem,
        #     neck_rounds=cfg.train.neck_rounds,
        #     head_width=cfg.train.head_width,
        #     aux_head=cfg.train.aux_head,
        #     path_a_per_channel=cfg.train.path_a_per_channel,
        #     prior_fg=getattr(cfg.train, "prior_fg", None),
        # )
        # v9 model construction — uncomment alongside the v9 cfg block above:
        # model = ANetV1(
        #     arch="v9",
        #     use_checkpoint=cfg.train.use_checkpoint,
        #     dense=True,
        #     hidden=cfg.train.hidden,
        #     h1=cfg.train.h1,
        #     stem=cfg.train.stem,
        #     neck_rounds=cfg.train.neck_rounds,
        #     head_width=cfg.train.head_width,
        #     aux_head=cfg.train.aux_head,
        #     path_a_per_channel=cfg.train.path_a_per_channel,
        #     prior_fg=getattr(cfg.train, "prior_fg", None),
        #     head_proto=getattr(cfg.train, "head_proto", True),
        # )
    n_params = sum(p.numel() for p in model.parameters())
    n_aux = model.aux.weight.numel() + model.aux.bias.numel() \
        if getattr(model, "aux", None) is not None else 0
    enc = getattr(model, "encoder", None)  # v13 has no window encoder
    print(f"ANetV1 {model.arch}: {n_params:,} params "
          f"({n_params - n_aux:,} deployed + {n_aux} aux) | "
          + (f"hidden={enc.hidden} h1={enc.h1} " if enc is not None else "")
          + f"stem={model.stem} | "
          f"loss={cfg.train.loss_mode} lr={cfg.train.lr} "
          f"epochs={cfg.train.epochs} | "
          f"train {len(train_ds)} | val {len(val_ds)} | data {cfg.data.root}")
    assert n_params < 40_000, "param budget exceeded (must stay under 40k)"
    Trainer(model, train_ds, val_ds, cfg).train()


if __name__ == "__main__":
    main()
