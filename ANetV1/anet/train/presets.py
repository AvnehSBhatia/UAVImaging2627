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
        batch_size=32 if IS_CUDA else 4,
        accum_steps=2 if IS_CUDA else 4,
        lr=4.0e-3 if IS_CUDA else 3.0e-3,
        warmup_steps=300 if IS_CUDA else 0,
        grad_clip=1.0,
        # torch.compile is OFF: it crashed twice on this ROCm build — mode
        # "reduce-overhead" (HIP graphs) is incompatible with grad-accum +
        # on-GPU loss accumulation, and "default" OOM'd the host compiling the
        # BACKWARD graph (died "Terminated" right after the first forward). The
        # runahead (nan_check_every) already gives the launch-bound speedup, so
        # compile is not needed. To retry later: set compile=True AND export
        # TORCHINDUCTOR_COMPILE_THREADS=1 to cap the compiler's host RAM.
        compile=False,
        compile_mode="default",
        # benchmark=True is a win on NVIDIA (cuDNN picks fast algos per shape,
        # shapes here are static) but forces an exhaustive ~27-min MIOpen search
        # on ROCm for zero gain (see trainer.py) — so: on for CUDA, off for HIP.
        cudnn_benchmark=IS_CUDA and not IS_ROCM,
        # every loss.item() is a full GPU sync that stops the CPU from queueing
        # the next steps (the model is launch-bound, so runahead IS the speedup).
        # >1 = accumulate the loss on-GPU and sync/NaN-check every N steps.
        # Tradeoff: a NaN inside a window is only caught at the window edge, after
        # its optimizer steps already ran — divergence still dies fast (streak
        # logic), but transient NaNs are no longer skipped. 1 = old exact behavior.
        nan_check_every=16 if IS_CUDA else 1,
        hidden=16,                    # 16 = spec width; 24 = capacity bump (ARCH §8.2)
        stem="edge_dq",               # v7 default (D33); "highpass" = 3x3 variant (D32)
        use_checkpoint=not IS_CUDA,   # Mac: memory valve; CUDA: pure waste
        amp="bf16" if IS_CUDA else None,  # fp16 NaNs (measured); bf16 validated on MI300X
        samples_per_epoch=None if IS_CUDA else 6000,
        early_stop_patience=6,
        early_stop_min_epochs=10,
        select_tent_weight=0.5,       # best.pt = argmax(mannequin + 0.5*tent), not mannequin alone
        mps_memory_frac=0.5,          # Mac: error instead of swap-freezing
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
        # FP-reduction step 1: alpha 0.7->0.8 makes Focal-Tversky punish false
        # positives harder than misses (fp/img was too high).
        tversky_alpha=0.8,           # FP penalty (both modes)
        tversky_beta=0.3,            # FN penalty (both modes)
        # FP-reduction step 2: at eval/deploy, only count a foreground cell if its
        # softmax prob clears this bar — kills marginal (ambiguous) predictions that
        # dominate fp/img. 0 = plain argmax. Raise to cut FP, lower if recall drops.
        conf_thresh=0.5,
        init_from=os.environ.get("ANET_INIT_FROM"),  # resume/fine-tune from a checkpoint
        l2_score_reg=1.0e-4,          # cosine-frequency bound (D24)
        l1_kernel_reg=1.0e-4,         # sparse pyramid kernels (D24)
        # SPAWN workers are heavy (each re-imports torch, ~1-2GB); 16 of them + the
        # compile process OOM'd the container ("Terminated" + leaked semaphores).
        # The model is launch-bound so the GPU is barely fed anyway — 6 workers
        # decode well ahead of the ~1-2s/step compute. Bump only if a worker
        # profile shows the GPU actually starving.
        num_workers=min(6, os.cpu_count() or 6) if IS_CUDA else 2,
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
