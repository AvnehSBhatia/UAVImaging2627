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


def anet_cfg(**overrides):
    train = dict(
        epochs=30,
        # effective batch 64 on CUDA (32x2), 16 on Mac (4x4).
        # MEMORY (measured, 2026-07-08, hidden=24/edge_dq, batch 1 fwd+bwd):
        # saved-for-backward was 2.87 GiB/img eager — 1.35 GiB of it fp32
        # ManualBatchNorm intermediates (now fused F.batch_norm off-MPS) and
        # ~0.5 GiB an fp32 type-promotion from the fp32 uv_tile cat (fixed).
        # With those fixes it is 1.54 GiB/img eager, ~0.65 compiled (inductor
        # rematerializes the pointwise chains), ~0.45+segment with per-round
        # checkpointing. ROCm default batch 16 x accum 4 (effective 64
        # unchanged) => ~10 GB compiled / ~25 GB eager fallback on the 192 GB
        # card. MIOpen autotune can still spike multi-GB workspaces on first
        # steps (MIOPEN_FIND_MODE=NORMAL once builds the find-db if it cliffs).
        batch_size=int(os.environ["ANET_BATCH"]) if "ANET_BATCH" in os.environ
        else (16 if IS_ROCM else (32 if IS_CUDA else 4)),
        accum_steps=int(os.environ["ANET_ACCUM"]) if "ANET_ACCUM" in os.environ
        else (4 if IS_ROCM else (2 if IS_CUDA else 4)),
        lr=float(os.environ["ANET_LR"]) if "ANET_LR" in os.environ
        else (4.0e-3 if IS_CUDA else 3.0e-3),
        # higher peak LR needs a longer ramp or it diverges on step 1 — scale
        # warmup with LR (ANET_WARMUP overrides). At 1e-2 this gives ~600 steps.
        warmup_steps=int(os.environ["ANET_WARMUP"]) if "ANET_WARMUP" in os.environ
        else (300 if IS_CUDA else 0),
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
        grad_clip=1.0,
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
        # per-channel Path A kernels (D37): the pre-registered "mushy tent blob"
        # upgrade (ARCH §8.2 step 2), justified by the 000008 viz (tent recall
        # ~half, low-contrast tent nearly missed, mannequin/car scale confusion).
        # Box-filter init -> starts identical to the shared-scalar spec form.
        # False restores the exact 179-param D13 Path A.
        path_a_per_channel=True,
        # per-round checkpointing (encoder.forward_dense(ckpt=True)): backward
        # peak = forward-held boundaries (~0.45 GiB/img) + the largest
        # rematerialized segment, for one extra Stage-1 forward. Mac:
        # mandatory memory valve. CUDA/ROCm: off — compile's min-cut
        # partitioner already rematerializes, and checkpoint HOPs would break
        # up the fused graph. ANET_CKPT=1 for eager runs that must stay small.
        use_checkpoint=(os.environ["ANET_CKPT"].strip().lower() in ("1", "true", "yes"))
        if "ANET_CKPT" in os.environ else not IS_CUDA,
        amp="bf16" if IS_CUDA else None,  # fp16 NaNs (measured); bf16 validated on MI300X
        samples_per_epoch=None if IS_CUDA else 6000,
        early_stop_patience=6,
        early_stop_min_epochs=10,
        select_tent_weight=0.5,       # best.pt = argmax(mannequin + 0.5*tent), not mannequin alone
        mps_memory_frac=0.5,          # Mac: error instead of swap-freezing
        # cap the torch allocator below physical VRAM so overallocation raises a
        # catchable torch.OutOfMemoryError (traceback, diagnosable) instead of the
        # driver killing the process with a bare SIGTERM; also leaves MIOpen
        # room to get real autotune workspace instead of "provided ptr: 0"
        cuda_memory_frac=0.90,
        focal_gamma=2.0,
        # mannequin alpha 12 (was 8): a mannequin is ~3x fewer cells than a tent
        # (60 vs 196 in the traced frame), so per-OBJECT it generated less loss
        # even at alpha 8 (60*8 < 196*4). 12 ~ per-object-balanced (4 * 196/60).
        class_alpha=[1.0, 12.0, 4.0],  # [background, mannequin, tent]
        # loss_mode "focal_tversky": ONE size-invariant balanced term (Focal-Tversky)
        # + a gentle focal anchor. No focal-vs-Tversky tug-of-war -> no fp 0.3<->28
        # limit cycle. "combo" = legacy focal + separate Tversky (kept for ablation).
        # loss_mode: "balanced" (class-balanced Focal-Tversky over {bg,mann,tent},
        # one term, no anchor — the anti-oscillation form), "focal_tversky"
        # (FT + focal anchor), or "combo" (legacy). ANET_LOSS_MODE overrides.
        loss_mode=os.environ.get("ANET_LOSS_MODE") or "focal_tversky",
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
        tversky_alpha=(0.8, 0.6),    # FP penalty per class (mannequin, tent)
        tversky_beta=0.3,            # FN penalty
        # ANTI-OSCILLATION knob: smooth is virtual-TP cells in the Tversky index;
        # a larger value makes the index LESS reactive to per-batch FP/FN noise,
        # which damps the limit cycle. 1.0 = original; ANET_SMOOTH=3 damps harder.
        ft_smooth=float(os.environ["ANET_SMOOTH"]) if "ANET_SMOOTH" in os.environ else 1.0,
        # eval/deploy FP gate (eval-only; does not affect training). 0.5 = original
        # working value; lower reveals sub-threshold predictions. ANET_CONF sets it.
        conf_thresh=float(os.environ["ANET_CONF"]) if "ANET_CONF" in os.environ else 0.5,
        init_from=os.environ.get("ANET_INIT_FROM"),  # resume/fine-tune from a checkpoint
        l2_score_reg=1.0e-4,          # cosine-frequency bound (D24)
        l1_kernel_reg=1.0e-4,         # sparse pyramid kernels (D24)
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
