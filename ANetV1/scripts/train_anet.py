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
ANET_MIN_EP, ANET_LOSS_MODE, DATA_ROOT. Point DATA_ROOT at
../datasets/suas-hyper-6k after `python -m gen_hyper.run` for the
hyper-accurate single-object / bg-only set. ANET_FREEZE_TRUNK=1 (with a v13
ANET_INIT_FROM): adapter mode — freeze the transferred v13 trunk, train only
the identity-init v14 modules. ANET_ARCH picks the arch (default v13);
ANET_CH="stem,mid,top" + ANET_BLOCKS size a v13/v15 capacity tier;
ANET_PARAM_BUDGET overrides the param assert (v15 defaults to 300k, D65).
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
_ARCH = os.environ.get("ANET_ARCH") or "v13"

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
    # DEFAULT REVERTED to v13 (the proven model) after two full-tune v14 runs
    # degraded val and the first adapter run was invalidated by live trunk
    # norm stats (see ARCHITECTURE.md section 16). v14 (D59-D63) is opt-in:
    # ANET_ARCH=v14, ideally with ANET_INIT_FROM=<v13 ckpt> +
    # ANET_FREEZE_TRUNK=1 — the corrected adapter test is v14's remaining
    # clean shot; if it cannot beat the donor's sel, v14 is falsified.
    # ANET_ARCH=v15 runs the D65 scaling tiers.
    arch=_ARCH,
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
    # v15 tiers get a LOWER peak: the SPD projection concentrates ~70% of all
    # params in one layer, so a step at v13's 1.5e-3 moves the function far
    # more than v13's largest 4k-param layer ever did — measured on both
    # tiers: violent recall/fp oscillation from epoch 1 (fp/img 572 -> 5 ->
    # 632 -> 1766 on tier-S). Same bigger-model/lower-LR law as everywhere.
    # v22 shares the funnel-dominant LR law: spd_proj holds ~68% of its
    # params, so the v15-measured 7.5e-4 peak + 600 warmup + 0.2x slow
    # group apply (same oscillation class if run at v13's 1.5e-3).
    lr=float(os.environ["ANET_LR"]) if "ANET_LR" in os.environ
    else (7.5e-4 if _ARCH in ("v15", "v22") else 1.5e-3),
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
    # v23 (D76) reads its two classes on two grids, so it needs the dual-grid
    # loss/eval branch; every other arch keeps the v12-v22 single-grid one.
    loss_mode=os.environ.get("ANET_LOSS_MODE")
    or ("center_dual" if _ARCH == "v23" else "center"),
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
        # v23 (D76): per-class targets on per-class grids (mannequin s10,
        # tent s20). Rasterized per item at load time -> no cache rebuild.
        center_dual=getattr(cfg.train, "loss_mode", "") == "center_dual",
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
        if model.arch == "v13" and cfg.train.arch == "v20":
            # v20 (D70) REPLACES down4/down20 with embed/unembed pairs, so
            # this is a PARTIAL warm start, not a D63 identity transfer:
            # stem/block4/blocks/head land (~20.5k of the donor's 25.2k
            # params), the donor's transition tensors are expected
            # leftovers, and the new pairs start Kaiming — the model is NOT
            # the donor at step 0. Full fine-tune only: ANET_FREEZE_TRUNK
            # would strand the fresh transitions between frozen stages
            # whose DN stats then chase them (the measured v14 adapter
            # failure mode, in reverse).
            m20 = ANetV1(arch="v20",
                         use_checkpoint=cfg.train.use_checkpoint,
                         head_width=cfg.train.head_width,
                         prior_fg=getattr(cfg.train, "prior_fg", None),
                         channels=model.backbone.channels,
                         n_blocks=len(model.backbone.blocks))
            donor_sd = model.state_dict()
            missing, unexpected = m20.load_state_dict(donor_sd, strict=False)
            leftovers = [k for k in unexpected
                         if not k.startswith(("backbone.down4.",
                                              "backbone.down20."))]
            assert not leftovers, f"v13->v20 transfer: unexpected {leftovers}"
            print(f"v13 -> v20 partial warm start: "
                  f"{len(donor_sd) - len(unexpected)} tensors transferred, "
                  f"{len(unexpected)} donor transition tensors dropped, "
                  f"{len(missing)} new tensors at Kaiming init")
            model = m20
        elif model.arch == "v13" and cfg.train.arch == "v23":
            # v23 (D76): the trunk transfers by name; the TENT HEAD is
            # SLICED out of the donor's 4-output head — v13 emits
            # [mann_heat, tent_heat, dx, dy], v23's tent head emits
            # [tent_heat, dx, dy], i.e. donor rows [1,2,3]. That makes the
            # tent path bit-for-bit the donor's, which (with the freeze
            # below) is what makes tent safety a construction, not a hope.
            m23 = ANetV1(arch="v23",
                         use_checkpoint=cfg.train.use_checkpoint,
                         head_width=cfg.train.head_width,
                         prior_fg=getattr(cfg.train, "prior_fg", None),
                         channels=model.backbone.channels,
                         n_blocks=len(model.backbone.blocks))
            donor = model.state_dict()
            sliced = dict(donor)
            for k in ("head.0.weight", "head.0.bias"):
                sliced[f"backbone.tent_head.{k.split('.', 1)[1]}"] = \
                    donor[f"backbone.{k}"]
            for k in ("head.2.weight", "head.2.bias"):
                src = donor[f"backbone.{k}"]
                sliced[f"backbone.tent_head.{k.split('.', 1)[1]}"] = src[[1, 2, 3]]
            for k in list(sliced):
                if k.startswith("backbone.head."):
                    del sliced[k]
            missing, unexpected = m23.load_state_dict(sliced, strict=False)
            assert not unexpected, f"v13->v23: unconsumed donor tensors: {unexpected}"
            new_ok = ("backbone.aniso.", "backbone.man_proj.",
                      "backbone.man_norm.", "backbone.man_block.",
                      "backbone.man_head.")
            stray = [k for k in missing if not k.startswith(new_ok)]
            assert not stray, f"v13->v23: unexpected new tensors: {stray}"
            n_p, n_n = m23.backbone.freeze_donor()
            trainable = sum(p.numel() for p in m23.parameters() if p.requires_grad)
            print(f"v13 -> v23 dual-grid transfer: trunk by name + tent head "
                  f"sliced from donor rows [1,2,3]; {n_p} donor tensors and "
                  f"{n_n} DeployNorm stat sets FROZEN (weights+stats, the "
                  f"D39/16.1 law); {trainable:,} params train (mannequin "
                  f"branch only)")
            model = m23
        elif model.arch == "v13" and cfg.train.arch == "v22":
            # v22 (D72): the family's first FULL-identity capacity growth —
            # every donor tensor (weights AND DeployNorm stat buffers) lands
            # bit-exact; the new funnel branch, peak gates, bias sites and
            # bg head start at Kaiming/zero behind zero valves, so step 0
            # IS v13_best (smoke-asserted). Full fine-tune only: freezing
            # the trunk would strand the fresh funnel between frozen stages
            # (the measured 16.1 adapter failure mode) — ANET_FREEZE_TRUNK
            # is refused below.
            m22 = ANetV1(arch="v22",
                         use_checkpoint=cfg.train.use_checkpoint,
                         head_width=cfg.train.head_width,
                         prior_fg=getattr(cfg.train, "prior_fg", None),
                         channels=model.backbone.channels,
                         n_blocks=len(model.backbone.blocks))
            donor_sd = model.state_dict()
            missing, unexpected = m22.load_state_dict(donor_sd, strict=False)
            assert not unexpected, \
                f"v13->v22 transfer: donor tensors with nowhere to go: {unexpected}"
            new_ok = ("backbone.spd_proj.", "backbone.peak_proj.",
                      "backbone.spd_norm.", "backbone.spd_gain",
                      "backbone.bg_head.")
            stray = [k for k in missing if not k.startswith(new_ok)]
            assert not stray, f"v13->v22 transfer: unexpected new tensors: {stray}"
            if os.environ.get("ANET_FREEZE_TRUNK") == "1":
                raise SystemExit(
                    "ANET_FREEZE_TRUNK is not supported for v22: the fresh "
                    "funnel branch feeds frozen stages whose stats would "
                    "chase it (the 16.1 failure mode). Full fine-tune only.")
            print(f"v13 -> v22 full-identity growth: {len(donor_sd)} donor "
                  f"tensors land bit-exact, {len(missing)} new tensors at "
                  "Kaiming/zero (step 0 == donor; asserted in smoke_test)")
            model = m22
        elif model.arch == "v13" and cfg.train.arch in ("v14", "v16", "v17", "v18", "v19"):
            # v14/v16 are supersets of v13 by construction (same module names
            # for the shared trunk; every new module identity-init, D63): copy
            # the v13 weights in and the target IS that v13 at step 0 —
            # asserted in smoke_test. Selection can only move away from a
            # proven optimum. v16 inherits the donor's channel plan/depth so
            # scaled-v13 donors warm-start too; v14 pins the spec shapes.
            extra = (dict(channels=model.backbone.channels,
                          n_blocks=len(model.backbone.blocks))
                     if cfg.train.arch in ("v16", "v17", "v18", "v19") else {})
            m14 = ANetV1(arch=cfg.train.arch,
                         use_checkpoint=cfg.train.use_checkpoint,
                         head_width=cfg.train.head_width,
                         prior_fg=getattr(cfg.train, "prior_fg", None), **extra)
            donor_sd = model.state_dict()
            missing, unexpected = m14.load_state_dict(donor_sd, strict=False)
            assert not unexpected, \
                f"v13->{cfg.train.arch} transfer: unexpected {unexpected}"
            print(f"v13 -> {cfg.train.arch} warm start: {len(donor_sd)} tensors "
                  f"transferred, {len(missing)} new tensors at identity init")
            if os.environ.get("ANET_FREEZE_TRUNK") == "1":
                # Adapter mode: only the identity-init v14 modules train; the
                # donor v13 trunk is immutable, so the run can test whether
                # the D59-D63 priors ADD value without being able to damage a
                # proven optimum. (Both full-tune v14 runs degraded val while
                # train loss fell — this isolates whether the new modules'
                # concept generalizes at all.) Trunk DeployNorm stats still
                # EMA-track the data; the donor trained on this distribution,
                # so those updates are ~no-ops.
                frozen = 0
                for name, p in m14.named_parameters():
                    if name in donor_sd:
                        p.requires_grad_(False)
                        frozen += 1
                # freeze the donor DeployNorm STATS too. The first adapter
                # run left them live ("updates are ~no-ops") — wrong: the
                # trainable modules sit UPSTREAM of frozen norms (noise ->
                # stem_norm, qshift_i -> next frozen stage), so the stats
                # chased the adapters' distribution shifts while the frozen
                # weights could not re-adapt. Measured: sel 1.712 -> 0.26
                # with train loss RISING from epoch ~9 — function drift, not
                # overfitting. With stats pinned, the donor function is a
                # fixed point again and only the adapter params move.
                from anet.model.norm import DeployNorm
                n_norms = 0
                for name, mod in m14.named_modules():
                    if isinstance(mod, DeployNorm) \
                            and f"{name}.running_mean" in donor_sd:
                        mod.frozen = True
                        n_norms += 1
                trainable = sum(p.numel() for p in m14.parameters()
                                if p.requires_grad)
                print(f"ANET_FREEZE_TRUNK=1: {frozen} donor tensors + "
                      f"{n_norms} donor DeployNorm stat sets frozen; "
                      f"{trainable:,} params (new v14 modules only) train")
            model = m14
        elif model.arch != cfg.train.arch:
            raise SystemExit(
                f"{init} is a {model.arch} checkpoint but this script is "
                f"configured for arch={cfg.train.arch!r} — a run cannot "
                "warm-start from a different-arch encoder. Remove init_from "
                "to train from scratch, or evaluate it with "
                "scripts/evaluate_all.py instead.")
        # gentle warm start unless the caller explicitly asked for more.
        # v22 keeps the funnel-family 600-step ramp even on warm start: the
        # branch is fresh and holds ~68% of all params (the v15 lesson).
        if "ANET_WARMUP" not in os.environ:
            cfg.train.warmup_steps = 600 if cfg.train.arch == "v22" else 100
        if "ANET_LR" not in os.environ:
            cfg.train.lr = min(cfg.train.lr, 1.5e-3)
        print(f"RESUMING from {init} (warm start: lr={cfg.train.lr}, "
              f"warmup={cfg.train.warmup_steps})")
    else:
        # v14 (current default, D59-D63; v13 = arch swap): the conv backbones
        # ignore hidden/h1/stem/neck/Path-A knobs entirely — the channel plan
        # is fixed in model/backbone.py; only head_width and prior_fg apply.
        # D65 scaling-curve knobs: ANET_CH="stem,mid,top" and ANET_BLOCKS
        # size a v13/v15 tier (unset = the spec defaults). Registered tiers,
        # section 16.3: v13 25k (origin) | v15-S defaults ~74k | v15-M
        # ANET_CH=16,48,96 ANET_BLOCKS=4 ~170k.
        channels = (tuple(int(c) for c in os.environ["ANET_CH"].split(","))
                    if "ANET_CH" in os.environ else None)
        n_blocks = (int(os.environ["ANET_BLOCKS"])
                    if "ANET_BLOCKS" in os.environ else None)
        model = ANetV1(
            arch=cfg.train.arch,
            use_checkpoint=cfg.train.use_checkpoint,
            head_width=cfg.train.head_width,
            prior_fg=getattr(cfg.train, "prior_fg", None),
            channels=channels,
            n_blocks=n_blocks,
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
    # the historic <40k budget stands for the deploy-track archs; v15 is the
    # pre-registered budget relaxation for the D65 capacity curve (section
    # 16.3, sanctioned by 16.2's measured underfit). ANET_PARAM_BUDGET pins it.
    # v22's 100k is the pre-registered D65-curve relaxation (§16.2 sanctions
    # capacity; v22 sits at the tier-S point + peak/bias additions).
    # v23 deliberately stays inside the ORIGINAL <=40k budget (owner's
    # chosen envelope: fix the margin via readout/features, not capacity).
    budget = int(os.environ.get("ANET_PARAM_BUDGET",
                                {"v15": 300_000, "v22": 100_000}.get(
                                    cfg.train.arch, 40_000)))
    assert n_params < budget, \
        f"param budget exceeded: {n_params:,} >= {budget:,} (ANET_PARAM_BUDGET overrides)"
    Trainer(model, train_ds, val_ds, cfg).train()


if __name__ == "__main__":
    main()
