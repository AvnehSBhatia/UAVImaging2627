"""Device-aware training defaults — plain Python, no yaml.

Training scripts import a builder and override inline:

    cfg = anet_cfg(hidden=24, epochs=30)

Anything not overridden gets the right default for the machine it runs on:
CUDA/ROCm (MI300X: big batch, bf16, no checkpointing) vs Apple Silicon
(small batch, fp32, gradient checkpointing, allocator cap).
"""

import os
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
IS_CUDA = torch.cuda.is_available()
IS_ROCM = torch.version.hip is not None  # MI300X presents as cuda but is MIOpen


def _cuda_total_gib():
    if IS_CUDA:
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return None
    return None


def _mem_budget_gib():
    """VRAM budget for batch sizing and the allocator cap. ANET_VRAM_GB pins an
    explicit ceiling — e.g. a shared/partitioned MI300X that reports 192 GB but
    only grants 48 ('use 48 GB max'); otherwise the card's real total."""
    if "ANET_VRAM_GB" in os.environ:
        return float(os.environ["ANET_VRAM_GB"])
    return _cuda_total_gib()


def _auto_batch(arch=None):
    """Memory-safe default batch. For the token-encoder archs (v8/v9/v12) the
    model is ~21k params but Stage-1 convolves the full 540x960 frame across a
    4-phase batch, so ACTIVATIONS — not weights — dominate: ~1.9 GiB/img
    dense-EAGER at hidden=32, and with gradient checkpointing (default on,
    below) ~0.5 GiB/img. Free VRAM is NOT knowable from the device total — a
    192 GB MI300X is routinely shared/partitioned — so the default is
    deliberately CONSERVATIVE and does not scale to the card:

      - ANET_BATCH=<n>     : pin the batch directly.
      - ANET_VRAM_GB=<gb>  : size the batch so the dense-eager worst case fits
                             ~55% of that budget (48 -> 13, 192 -> 55), and cap
                             the allocator to it (below).
      - neither            : batch 16 — trains in ~8-12 GiB with checkpointing,
                             fits alongside other jobs on a shared card.

    v13 (D58) is a plain conv pyramid whose biggest activation is the 16-ch
    stride-2 stem map — ~0.1 GiB/img with autograd copies, ~20x lighter than
    v12 — so the 1.9 GiB/img estimate would strand it at batch 13 x accum 7
    (measured on the first v13 MI300X run: a pure v12 VRAM artifact). v13
    sizes against 0.12 GiB/img, which puts any budgeted card at the full
    physical batch 96 / accum 1.

    accum_steps holds the effective batch near 96 so the LR schedule/step count
    are stable regardless of the physical batch."""
    if "ANET_BATCH" in os.environ:
        return int(os.environ["ANET_BATCH"])
    budget = _mem_budget_gib()
    if budget is None:
        return 4  # Mac / CPU
    if arch in ("v13", "v14"):
        gib_per_img = 0.12
    elif arch == "v15":
        gib_per_img = 0.22  # SPD tier-M (wider s4 stage) worst case
    else:
        gib_per_img = 1.9
    if "ANET_VRAM_GB" in os.environ:  # explicit budget -> size the batch to it
        return max(4, min(96, int(budget * 0.55 / gib_per_img)))
    # conservative defaults; free VRAM on a shared card is unknowable. The
    # conv backbones' batch 96 is still only ~10-14 GiB — modest even beside
    # other jobs (v14 adds a full-res dw7x7 + s4 texture maps over v13).
    return 96 if arch in ("v13", "v14") else 16


def anet_cfg(**overrides):
    bs = _auto_batch(overrides.get("arch"))
    train = dict(
        epochs=30,
        # VRAM-aware batch (see _auto_batch): sized so the dense-EAGER path
        # (~1.8 GiB/img at hidden=32, the worst case) fits the actual card, so a
        # 48 GB box no longer OOMs on the batch-96 MI300X default. accum holds
        # the effective batch near 96 (so step count / LR schedule are stable
        # across cards); on Mac it stays the old 4x4=16. MIOpen autotune can
        # still spike multi-GB workspaces on the first steps — the cuda_memory_frac
        # cap turns any such overrun into a catchable OOM, not a driver SIGTERM.
        batch_size=bs,
        accum_steps=int(os.environ["ANET_ACCUM"]) if "ANET_ACCUM" in os.environ
        else (max(1, round(96 / bs)) if IS_CUDA else 4),
        lr=float(os.environ["ANET_LR"]) if "ANET_LR" in os.environ
        else (4.0e-3 if IS_CUDA else 3.0e-3),
        # higher peak LR needs a longer ramp or it diverges on step 1 — scale
        # warmup with LR (ANET_WARMUP overrides). At 1e-2 this gives ~600 steps.
        # v15 gets a longer ramp: the SPD projection holds ~70% of all params
        # in one layer, and the first v15 runs over-fired violently while
        # still INSIDE the 300-step warmup (fp/img 572 at epoch 1).
        warmup_steps=int(os.environ["ANET_WARMUP"]) if "ANET_WARMUP" in os.environ
        else ((600 if overrides.get("arch") == "v15" else 300) if IS_CUDA else 0),
        # LR schedule after warmup: "cosine" (smooth decay to 0, default),
        # "restarts" (cosine warm restarts — periodic LR spikes to re-escape a
        # plateau; good for a stuck class), "plateau" (ReduceLROnPlateau on the
        # val selection metric). ANET_SCHED overrides. See the plateau caveat in
        # trainer.py: it can CUT LR while a class is stuck at 0 (reads as a
        # plateau) — exactly when you want LR high — so restarts is usually the
        # better "ramp down but re-escape" choice for the mannequin basin.
        sched=os.environ.get("ANET_SCHED") or "cosine",
        plateau_patience=3, plateau_factor=0.5,   # ReduceLROnPlateau params
        restart_epochs=5,                          # T_0 for cosine warm restarts
        # 10.0 for focal_norm (measured grad norms 25-180 in early training;
        # clip 1.0 would bind on every step and distort the loss's relative
        # term scales — Adam absorbs uniform scaling but not a varying one).
        # Legacy v8 fine-tunes tuned against 1.0 should override inline.
        grad_clip=10.0,
        # torch.compile ON for CUDA/ROCm: this net is launch-bound (thousands of
        # tiny kernels at ~1% util), and inductor fusing the elementwise chains
        # into a few Triton kernels is the biggest single lever. Two past crashes
        # are now root-caused and fixed:
        #   - "reduce-overhead" (HIP graphs) aliased the compiled output buffer,
        #     fighting grad-accum + on-GPU loss accumulation -> use mode "default"
        #     (fusion, no cudagraph capture). reduce-overhead stays selectable.
        #   - "default" once OOM'd the host compiling the BACKWARD graph because
        #     inductor forks one full-torch compile worker per thread -> the
        #     trainer now setdefaults TORCHINDUCTOR_COMPILE_THREADS=1 before the
        #     first (lazy) compile, and the run script exports it too.
        # Robust fallback: any compile error (setup OR first-step) degrades to
        # eager instead of dying. ROCm now defaults ON: the epoch-0 hang was
        # root-caused to fork()ed spawn workers + the inductor compile-worker
        # fork storm, both fixed (spawn ctx, TORCHINDUCTOR_COMPILE_THREADS=1).
        # ANET_COMPILE=0 is the quick kill switch if a container misbehaves.
        compile=IS_CUDA,
        compile_mode="default",
        # benchmark=True is a win on NVIDIA (cuDNN picks fast algos per shape,
        # shapes here are static) but forces an exhaustive ~27-min MIOpen search
        # on ROCm for zero gain (see trainer.py) — so: on for CUDA, off for HIP.
        cudnn_benchmark=IS_CUDA and not IS_ROCM,
        # nan_check_every=1 forces a device sync per step (loss.item()) — it
        # existed because runahead at ~80GB/step of activations OOM'd the card.
        # With fused BN + bf16 stream + per-round checkpointing a step is
        # ~5-10GB at batch 32, so 2-3 steps of CPU runahead fit easily and the
        # per-step sync (which serialized the launch-bound tail) goes away.
        nan_check_every=25 if IS_CUDA else 1,
        hidden=16,                    # 16 = spec width; 24 = capacity bump (ARCH §8.2)
        stem="edge_dq",               # v7 default (D33); "highpass" = 3x3 variant (D32)
        # ----------------------------------------------------------- v9 keys
        # (harmless for v8 runs; scripts/train_anet.py selects arch="v9")
        arch="v8",
        h1=48,                        # pre-pool token width (D42)
        neck_rounds=2,                # ConvNeck depth (D43)
        head_width=24,                # classifier width (D45)
        # --------------------------------------------------------- v12 keys
        # (harmless for v8/v9 runs; scripts/train_anet.py selects arch="v12"
        # + loss_mode="center" together — see trainer.py's center branch)
        center_alpha=2.0,             # center_focal_loss positive-term focusing power
        center_beta=4.0,              # center_focal_loss negative-penalty falloff around a peak
        # up-weight the (rare, ~1-cell) positive center term vs the ~2500 bg cells
        # so the center prob climbs faster than the measured ~0.002/epoch from-
        # scratch crawl. ANET_POS_W overrides. 1.0 = plain CenterNet.
        center_pos_weight=float(os.environ["ANET_POS_W"]) if "ANET_POS_W" in os.environ
        else 3.0,
        offset_weight=1.0,            # weight of offset_l1 relative to center_focal_loss
        # v12 DEEP SUPERVISION: weight of a center_focal probe straight off the
        # encoder embedding map (ANetV1.aux_center, train-only). The pinpoint
        # diagnostic showed the encoder gives the deep head only a ~0.05 object
        # signal at init, so training stalled at constant output; this direct
        # gradient forces the encoder to amplify that separation. 1.0 = full
        # weight (the encoder is the bottleneck); 0 disables. ANET_AUX_W overrides.
        center_aux_weight=float(os.environ["ANET_AUX_W"]) if "ANET_AUX_W" in os.environ
        else 1.0,
        peak_thresh=0.3,              # eval-time 3x3-local-max heatmap threshold (CenterObjectMetrics)
        # center-mode selection sanity gate: epochs with object fp/img above
        # this can neither become best.pt nor set the early-stop reference
        # (the degenerate over-predictor guard, center-mode edition — see
        # trainer.py; the v9 guard uses cell precision, which center mode
        # lacks). Real operating points are a few fp/img; over-fire episodes
        # measured 40-1800.
        max_sel_fp=25.0,
        # Gaussian splat sigma (cells) for the heat target (rasterize.py). 1.5
        # (was 0.7 ~ a single cell) gives a ~3x3 soft core so cells adjacent to a
        # true center carry a near-1 target and are barely penalized as negatives
        # — a smoother objective the from-scratch head climbs faster. Wired
        # through SUASCells(center_sigma=...) now (the dataset used to hardcode 0.7).
        center_sigma=1.5,
        # aux deep-supervision probe (D46): DROPPED in v10. Measured (single
        # forward+backward decomposition) it contributes 0.02% of the encoder
        # gradient — the hard-loss path dominates ~4000x even when the head is
        # confidently wrong (focal grad doesn't vanish on wrong-class), so it
        # never achieved its "gradient path a collapsed head can't block" goal;
        # meanwhile its private linear-probe weights fitting themselves were
        # ~19% of the logged loss, decoupled from detection (part of why the
        # loss fell while metrics oscillated). Off by default; env re-enables.
        aux_head=(os.environ.get("ANET_AUX", "0").strip().lower() in ("1", "true", "yes")),
        aux_weight=float(os.environ.get("ANET_AUX_W", 0.3)),
        ema_decay=0.998,              # weight EMA for eval/checkpoints (D48); 0=off
        # focal_norm (v10) class weights (bg, mannequin, tent). Per-class MEAN
        # normalization now, so weights are pure class-importance (no size-bias
        # compensation needed): mannequin 2x the rare hard class.
        focal_norm_weights=(1.0, 2.0, 1.0),
        # fused Triton Stage-1 (D40): parity-checked at startup, demotes to
        # chunked-autograd backward, then to the PyTorch dense path (at
        # fallback_batch to stay inside a 20 GB VRAM budget). ANET_FUSED=0 and
        # ANET_FUSED_BWD=triton|chunked override.
        fused=IS_CUDA,
        fused_bwd="triton",
        # dense-fallback batch (used if the fused path demotes): never larger
        # than the VRAM-sized batch, so the heavier dense-eager path still fits.
        fallback_batch=min(32, bs),
        # DeployNorm seeding (D39): 24 covers the full cumulative-average ramp
        # (momentum locks to its 0.05 floor at step 20) before any real gradient,
        # so training never normalizes against a half-formed running stat.
        seed_stat_batches=24,
        prefetch=True,                # background-thread H2D pipeline
        # per-channel Path A kernels (D37): the pre-registered "mushy tent blob"
        # upgrade (ARCH §8.2 step 2), justified by the 000008 viz (tent recall
        # ~half, low-contrast tent nearly missed, mannequin/car scale confusion).
        # Box-filter init -> starts identical to the shared-scalar spec form.
        # False restores the exact 179-param D13 Path A.
        path_a_per_channel=True,
        # per-round checkpointing (encoder.forward_dense(ckpt=True)): recompute
        # Stage-1 activations in the backward instead of storing them — backward
        # peak drops from ~1.9 to ~0.5 GiB/img (one extra Stage-1 forward). Now
        # ON by default EVERYWHERE (the model must fit a shared/partitioned card,
        # and activations — not the 21k weights — are the whole footprint). It
        # only touches the DENSE encoder path; the fused Triton kernel manages
        # its own memory and ignores it, so there is no fused-graph interaction.
        # ANET_CKPT=0 disables (a dedicated card with VRAM to spare + compile).
        use_checkpoint=(os.environ["ANET_CKPT"].strip().lower() in ("1", "true", "yes"))
        if "ANET_CKPT" in os.environ else True,
        amp="bf16" if IS_CUDA else None,  # fp16 NaNs (measured); bf16 validated on MI300X
        # ANET_SAMPLES caps draws/epoch (was uncapped on CUDA -> full ~13.5k ->
        # ~60 min/epoch dense). The WeightedRandomSampler redraws i.i.d. from a
        # FIXED distribution every epoch, so capping changes only how often you
        # eval/checkpoint, not what any step sees — a pure, safe speed lever.
        # run_anet_mi300x.sh sets ANET_SAMPLES=6000 (~2.3x faster feedback).
        samples_per_epoch=(int(os.environ["ANET_SAMPLES"]) if "ANET_SAMPLES" in os.environ
                           else (None if IS_CUDA else 6000)),
        # generous early-stop so a fine-tune from good.pt has room to settle
        # (starts converged near mann 0.573; best.pt selection keeps the peak).
        # min 25 epochs, patience 12. ANET_MIN_EP / ANET_PATIENCE override.
        early_stop_patience=int(os.environ["ANET_PATIENCE"]) if "ANET_PATIENCE" in os.environ else 12,
        early_stop_min_epochs=int(os.environ["ANET_MIN_EP"]) if "ANET_MIN_EP" in os.environ else 25,
        select_tent_weight=0.5,       # best.pt = argmax(mannequin + 0.5*tent), not mannequin alone
        mps_memory_frac=0.5,          # Mac: error instead of swap-freezing
        # cap the torch allocator below physical VRAM so overallocation raises a
        # catchable torch.OutOfMemoryError (traceback, diagnosable) instead of the
        # driver killing the process with a bare SIGTERM; also leaves MIOpen
        # room to get real autotune workspace instead of "provided ptr: 0".
        # ANET_VRAM_GB pins a HARD ceiling: the fraction becomes budget/total so
        # the PyTorch allocator physically can't exceed it (e.g. 48/192 = 0.25 to
        # hold a partitioned MI300X at 48 GB — pair with the matching batch size
        # from _auto_batch so the activations themselves fit).
        cuda_memory_frac=(min(0.95, float(os.environ["ANET_VRAM_GB"]) / _cuda_total_gib())
                          if ("ANET_VRAM_GB" in os.environ and _cuda_total_gib())
                          else 0.90),
        focal_gamma=2.0,
        # mannequin alpha 12 (was 8): a mannequin is ~3x fewer cells than a tent
        # (60 vs 196 in the traced frame), so per-OBJECT it generated less loss
        # even at alpha 8 (60*8 < 196*4). 12 ~ per-object-balanced (4 * 196/60).
        class_alpha=[1.0, 12.0, 4.0],  # [background, mannequin, tent]
        # default "fp_tp": weighted per-class ratio of soft FP-RATE to soft
        # RECALL (v12/D57 — both terms cell-count-normalized, batch-pooled, ONE
        # bounded term). The v11 RAW-SUM form of this loss collapsed the head to
        # mann_r=0: its FP was a sum over ~5000 bg cells vs TP over ~60 object
        # cells, which (a) punished over-prediction ~100x harder than predicting
        # nothing so the correction overshot into the all-background basin, and
        # (b) let that basin's per-cell bg push out-vote recovery ~80:1. Rate-
        # normalizing both terms makes over-predict and collapse cost the SAME
        # (bounded ratio=1) and balances the recovery gradient. See losses.py.
        # Other modes are kept for v8 fine-tunes from good.pt (loss must match
        # what trained the checkpoint) and for ablation:
        #   "focal_norm" (D47), "balanced" (class-balanced Focal-Tversky over
        #   {bg,mann,tent}), "focal_tversky" (FT + focal anchor), "combo"
        #   (legacy focal + separate Tversky). "metric_only" (D56): detection
        #   loss OFF — only proto_metric_loss trains, to PRETRAIN the embedding
        #   space; save last.pt then ANET_INIT_FROM=... fine-tune with fp_tp.
        #   "center" (v12): object center-heatmap detector — center_focal_loss
        #   (CenterNet penalty-reduced focal) over two INDEPENDENT per-class
        #   sigmoids (mannequin, tent; no softmax competition) + offset_weight
        #   * offset_l1 for the sub-cell (dx,dy) regression. Requires the
        #   dataset built with center=True (adds heat/offset/reg_mask targets)
        #   and arch="v12" (model returns a {"heat","offset"} dict, not a cell
        #   tensor) — see trainer.py's loss_mode=="center" branch.
        #   ANET_LOSS_MODE overrides.
        loss_mode=os.environ.get("ANET_LOSS_MODE") or "fp_tp",
        # fp_tp per-class weights in index order (bg, mann, tent). The requested
        # prior: mannequin 0.8, tent 0.15, background 0.05.
        fp_tp_weights=(0.05, 0.8, 0.15),
        # fp_tp smooth (on the [0,1] rate/recall terms, v12): sets the collapse-
        # point escape-gradient scale (~1/smooth) and the dynamic range
        # (perfect = s/(1+s)). 0.1 -> perfect≈0.09 vs collapse=1.0, a strong
        # escape gradient with wide range. (The v11 raw-sum form used 1.0.)
        fp_tp_smooth=0.1,
        # metric-prototype head (D56): weight of proto_metric_loss added to the
        # detection loss. This is the per-cell discriminative signal (softmax
        # over distance-to-prototype) that makes the TRUE class win the argmax —
        # the thing fp_tp cannot do — and shapes the head's prototypes into
        # separable mannequin/tent/bg clusters. 0 disables (falls back to the
        # pure detection loss + a plain-linear-equivalent head). ANET_METRIC_W
        # overrides. Set head_proto=False to remove the prototype head entirely.
        metric_weight=float(os.environ["ANET_METRIC_W"]) if "ANET_METRIC_W" in os.environ
        else 0.5,
        # metric CE per-class weight (bg, mann, tent) AFTER count-normalization
        # (so these are pure priority, not size compensation). Equal by default.
        metric_class_weights=(1.0, 1.0, 1.0),
        # prototype separation push (mean exp(-‖p_i-p_j‖²)); keeps clusters apart.
        metric_sep_weight=0.1,
        head_proto=(os.environ.get("ANET_HEAD_PROTO", "1").strip().lower()
                    in ("1", "true", "yes")),
        # "balanced" per-class FP (alpha) / miss (beta). bg/tent symmetric 0.5;
        # mannequin gets beta>alpha (a recall push) since it's the hard
        # under-detected class. All classes weigh equally regardless of cell
        # count, so no per-class alpha juggling like the other modes need.
        balanced_alpha=(0.5, 0.5, 0.5),   # (bg, mannequin, tent) FP penalty
        balanced_beta=(0.5, 0.65, 0.5),   # (bg, mannequin, tent) miss penalty
        # fixed class importance weights (bg, mann, tent) for balanced mode —
        # overrides difficulty_temp. ANET_CLASS_W="0.06,0.6,0.34" e.g. None=off.
        balanced_class_weights=(
            tuple(float(x) for x in os.environ["ANET_CLASS_W"].split(","))
            if "ANET_CLASS_W" in os.environ else None),
        # ANTI-OSCILLATION / anti-collapse feature: prior-bias head init
        # (RetinaNet §4.1). Starts each foreground class at this probability so
        # the head sits OFF the saturated all-background point and can't fully
        # collapse to 0 (the "predict nothing" half of the mannequin limit
        # cycle). 0.1 -> fc2.bias=[0,-2.08,-2.08]. Default ON for CUDA now.
        # ANET_PRIOR_FG=0 disables. Fresh models only (resume inherits ckpt bias).
        prior_fg=(float(os.environ["ANET_PRIOR_FG"]) or None) if "ANET_PRIOR_FG" in os.environ
        else (0.1 if IS_CUDA else None),
        # difficulty_temp: up-weight the worst-doing class (detached softmax over
        # per-class losses). None = equal weight (stable default). Small (~0.3)
        # focuses hard; large (~2) ~ equal. ANET_DIFF_TEMP overrides.
        difficulty_temp=float(os.environ["ANET_DIFF_TEMP"])
        if "ANET_DIFF_TEMP" in os.environ else None,
        ft_gamma=0.75,               # (1-TI)**gamma; <1 focuses hard classes, stays stable
        # ORIGINAL focal_tversky (the config that trained working tent + good.pt-
        # level mannequin). The 2026-07-08 "mannequin-collapse" rebalance cranked
        # the anchor (weight 0.75, alpha 6) and RE-CREATED the focal-vs-Tversky
        # tug-of-war -> mannequin oscillated 0<->over-predict. Reverted here; the
        # anti-oscillation now comes from stabilizers (prior_fg below stops the
        # collapse half; moderate LR + cosine avoid weight thrash) rather than
        # from a stronger anchor. Env-overridable to tune without editing.
        ft_anchor_weight=float(os.environ["ANET_ANCHOR_W"])
        if "ANET_ANCHOR_W" in os.environ else 0.5,
        ft_anchor_alpha=[1.0, 2.0, 2.0],  # MILD — balancing is Focal-Tversky's job
        tversky_weight=0.2,          # only used in "combo" mode
        # REVERTED to good.pt-original values (2026-07-09). The "collapse-after-
        # spike" these knobs were tuned against was NOT a loss oscillation — it was
        # ARCHITECTURE DRIFT: D36 path_dq + D37 per-channel Path A on a hidden=16
        # encoder regressed mannequin to soft p(fg)=0 (verified by state_dict diff).
        # good.pt (mann recall 0.573) is hidden=24 / shared Path A / no path_dq.
        # We now FINE-TUNE from good.pt, so the loss MUST match what trained good.pt
        # (alpha 0.8 / beta 0.3 / smooth 1.0) or fine-tuning drags it out of its
        # basin. The speculative smooth=2 / alpha=0.7 damping over-suppressed the
        # foreground gradient and helped keep mannequin dead. Env-overridable.
        tversky_alpha=(float(os.environ["ANET_TV_ALPHA"]) if "ANET_TV_ALPHA" in os.environ
                       else 0.8, 0.6),
        tversky_beta=float(os.environ["ANET_TV_BETA"]) if "ANET_TV_BETA" in os.environ else 0.3,
        # smooth = virtual-TP cells in the Tversky index. 1.0 = good.pt-original.
        # ANET_SMOOTH overrides (raise only to damp a REAL loss oscillation, and
        # only after confirming the architecture matches good.pt).
        ft_smooth=float(os.environ["ANET_SMOOTH"]) if "ANET_SMOOTH" in os.environ else 1.0,
        # eval/deploy FP gate (eval-only; does not affect training). 0.5 = original
        # working value; lower reveals sub-threshold predictions. ANET_CONF sets it.
        conf_thresh=float(os.environ["ANET_CONF"]) if "ANET_CONF" in os.environ else 0.5,
        init_from=os.environ.get("ANET_INIT_FROM"),  # resume/fine-tune from a checkpoint
        # D24 coefficients, rescaled ~30x for focal_norm (D47): the new loss
        # is ~27-32x larger in magnitude than the old per-cell-mean focal, so
        # 1e-4 silently diluted the deployment-critical cosine-frequency bound
        # (int8 LUT accuracy) and Path-A sparsity by the same factor. Legacy
        # loss modes (v8 fine-tunes) should override these back to 1e-4.
        l2_score_reg=3.0e-3,          # cosine-frequency bound (D24)
        l1_kernel_reg=3.0e-3,         # sparse pyramid kernels (D24)
        # ROCm default 0 (IN-PROCESS): spawn workers repeatedly DEADLOCK epoch-0
        # on this MIOpen/HIP container (fork copies locked native mutexes; the
        # first batch never arrives, needs kill -9). With data.cache an item is a
        # ~1.5MB memcpy, not a ~10ms decode, so the in-process loader keeps the
        # launch-bound GPU fed anyway — workers buy nothing here and cost a hang.
        # NVIDIA is fine with workers; opt back in on ROCm via ANET_NUM_WORKERS>0
        # only if you've confirmed spawn works on your box.
        num_workers=int(os.environ["ANET_NUM_WORKERS"]) if "ANET_NUM_WORKERS" in os.environ
        else (0 if IS_ROCM else (min(6, os.cpu_count() or 6) if IS_CUDA else 2)),
        prefetch_factor=2 if IS_CUDA else 2,
        checkpoint_dir="runs/anet",
    )
    data = dict(
        root=os.environ.get("DATA_ROOT", str(REPO_ROOT / "datasets/suas-synth-50k")),
        coverage_thresh=0.3,
        # VisDrone downweighting (no-ops if vd_* files were stashed out)
        vd_weight=0.4, mannequin_weight=4.0, tent_weight=2.0,
        # ship uint8 frames from the loader, normalize on-GPU (trainer): 4x less
        # H2D + pin_memory traffic. Only the Trainer handles this; eval scripts
        # build their own SUASCells without it and still get floats.
        uint8=IS_CUDA,
        # boundary ignore band: cells with class coverage in [band_lo, 0.3) are
        # labeled background by the hard threshold but are half object — a 29%-
        # vs 30%-covered cell is the same pixels with opposite labels. The loss
        # ignores them (anchor skips the cell; Tversky FP skips them for the
        # covering class only). Kills the ring tug-of-war at object boundaries.
        # None = off (plain hard labels, pre-band behavior).
        band_lo=0.05,
        # one-time preprocessing cache (memmapped uint8 frames + grids under
        # <root>/.anet_cache, ~70GB for the train split): items become ~1.5MB
        # memcpys instead of ~10ms PIL decode+resize. ANET_CACHE=0 to disable
        # (e.g. tight disk); Mac default off (6k samples/epoch hides decode).
        cache=(os.environ["ANET_CACHE"].strip().lower() in ("1", "true", "yes"))
        if "ANET_CACHE" in os.environ else IS_CUDA,
    )
    distill = dict(teacher_cache="runs/teacher_cache", kl_weight=0.7, temperature=2.0)
    yolo = dict(weights="yolo26n.pt", imgsz=960)  # shared by teacher cache + eval

    for k, v in overrides.items():
        for d in (train, data, distill, yolo):
            if k in d:
                d[k] = v
                break
        else:
            raise KeyError(f"unknown setting {k!r}")
    return SimpleNamespace(
        train=SimpleNamespace(**train), data=SimpleNamespace(**data),
        distill=SimpleNamespace(**distill), yolo=SimpleNamespace(**yolo),
    )
