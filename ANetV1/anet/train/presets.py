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
        # effective batch 64 on CUDA (32x2), 16 on Mac (4x4). NB: activations are
        # ~2.5GB/image under autograd (edge_dq/hidden24), so single-batch 64 OOMs
        # the 192GB card (~160GB) — 32 is the ceiling. accum 2 keeps effective 64.
        # MEMORY NOTE (D38): the single-batched Stage-1 pass holds all 4 phases'
        # transients at once (~10-15% higher peak than the old 4-pass loop) for
        # ~4x fewer kernel launches. Plus MIOpen autotune probes multi-GB
        # workspaces on the fresh D37 grouped-conv shapes (see the "workspace
        # required: 4.7e9" warnings) — a transient VRAM spike that can cliff a
        # batch-32 step. If it OOMs / gets Terminated: ANET_BATCH=16 ANET_ACCUM=4
        # (same effective 64, half the activation footprint, room for autotune
        # workspace), ideally with MIOPEN_FIND_MODE=NORMAL on the first run to
        # build the find-db so later epochs stop probing.
        batch_size=int(os.environ["ANET_BATCH"]) if "ANET_BATCH" in os.environ
        else (32 if IS_CUDA else 4),
        accum_steps=int(os.environ["ANET_ACCUM"]) if "ANET_ACCUM" in os.environ
        else (2 if IS_CUDA else 4),
        lr=4.0e-3 if IS_CUDA else 3.0e-3,
        warmup_steps=300 if IS_CUDA else 0,
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
        # eager instead of dying. Fast off-switch: ANET_COMPILE=0 (or compile=False).
        compile=IS_CUDA,
        compile_mode="default",
        # benchmark=True is a win on NVIDIA (cuDNN picks fast algos per shape,
        # shapes here are static) but forces an exhaustive ~27-min MIOpen search
        # on ROCm for zero gain (see trainer.py) — so: on for CUDA, off for HIP.
        cudnn_benchmark=IS_CUDA and not IS_ROCM,
        # nan_check_every>1 (runahead: no per-step sync) let the CPU enqueue the
        # NEXT step's forward while the previous step's ~80GB of activations were
        # still live -> ~160GB+ in flight -> VRAM exhausted -> the amdgpu/KFD
        # driver SIGTERMs the process ("Terminated", no traceback; MIOpen logs
        # "provided ptr: 0" because torch had no free VRAM for workspace). This
        # model's activations are too big for ANY lookahead at batch 32 — keep 1.
        # (Safe to raise only if batch is dropped so 2-3 steps fit in VRAM.)
        nan_check_every=1,
        hidden=16,                    # 16 = spec width; 24 = capacity bump (ARCH §8.2)
        stem="edge_dq",               # v7 default (D33); "highpass" = 3x3 variant (D32)
        # per-channel Path A kernels (D37): the pre-registered "mushy tent blob"
        # upgrade (ARCH §8.2 step 2), justified by the 000008 viz (tent recall
        # ~half, low-contrast tent nearly missed, mannequin/car scale confusion).
        # Box-filter init -> starts identical to the shared-scalar spec form.
        # False restores the exact 179-param D13 Path A.
        path_a_per_channel=True,
        use_checkpoint=not IS_CUDA,   # Mac: memory valve; CUDA: pure waste
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
        loss_mode="focal_tversky",
        ft_gamma=0.75,               # (1-TI)**gamma; <1 focuses hard classes, stays stable
        ft_anchor_weight=0.5,        # weight of the dense per-cell focal anchor
        ft_anchor_alpha=[1.0, 2.0, 2.0],  # MILD — balancing is Focal-Tversky's job, not the anchor's
        tversky_weight=0.2,          # only used in "combo" mode
        # FP-reduction step 1: alpha > beta makes Focal-Tversky punish false
        # positives harder than misses (fp/img was too high). Per-class pair
        # (mannequin, tent): the viz showed the global 0.8 shrinking tent blobs
        # to ~half their GT cells while the mannequin channel still hedged —
        # keep full FP pressure on mannequin, relax tent so it fills its box.
        tversky_alpha=(0.8, 0.6),    # FP penalty per class (both modes)
        tversky_beta=0.3,            # FN penalty (both modes)
        # smooth ~1 virtual TP cell in the Tversky index. The old eps=1e-6 made
        # the index saturate whenever a class was ABSENT from a frame: gradient
        # wrt FP measured ~1e-14, i.e. zero FP suppression exactly where FPs
        # live. That, not alpha, was why the mannequin channel became a generic
        # objectness halo (rings around tents, blobs on cars) held down only by
        # the mild anchor + the conf_thresh crutch.
        ft_smooth=1.0,
        # FP-reduction step 2: at eval/deploy, only count a foreground cell if its
        # softmax prob clears this bar — kills marginal (ambiguous) predictions that
        # dominate fp/img. 0 = plain argmax. Raise to cut FP, lower if recall drops.
        conf_thresh=0.5,
        init_from=os.environ.get("ANET_INIT_FROM"),  # resume/fine-tune from a checkpoint
        l2_score_reg=1.0e-4,          # cosine-frequency bound (D24)
        l1_kernel_reg=1.0e-4,         # sparse pyramid kernels (D24)
        # SPAWN workers are heavy (each re-imports torch, ~1-2GB); 16 of them + the
        # compile process OOM'd the container ("Terminated" + leaked semaphores at
        # shutdown). torch.compile now adds its own compile-worker RAM tenant, so
        # if the box is tight, cut workers: ANET_NUM_WORKERS=2 ./run_anet_mi300x.sh.
        # The model is launch-bound so the GPU is barely fed anyway — a handful of
        # workers decode well ahead of the ~1-2s/step compute. num_workers=0 makes
        # the loader run in-process (no spawn, no semaphore warning at all).
        num_workers=int(os.environ["ANET_NUM_WORKERS"]) if "ANET_NUM_WORKERS" in os.environ
        else (min(6, os.cpu_count() or 6) if IS_CUDA else 2),
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
