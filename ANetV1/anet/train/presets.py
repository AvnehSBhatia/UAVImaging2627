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
        class_alpha=[1.0, 8.0, 4.0],  # [background, mannequin, tent]
        # Tversky term (FP/FN-aware). focal is ~0.05 here, Tversky is ~0.3-0.6, so
        # weight 0.1 makes it a corrective, not the dominant term. alpha>beta =>
        # punish false positives harder (the fp/img knob). Raise weight or alpha to
        # cut FP; expect a small recall cost. Set weight 0 to disable.
        tversky_weight=0.1,
        tversky_alpha=0.7,            # FP penalty
        tversky_beta=0.3,             # FN penalty
        init_from=os.environ.get("ANET_INIT_FROM"),  # resume/fine-tune from a checkpoint
        l2_score_reg=1.0e-4,          # cosine-frequency bound (D24)
        l1_kernel_reg=1.0e-4,         # sparse pyramid kernels (D24)
        num_workers=8 if IS_CUDA else 2,
        checkpoint_dir="runs/anet",
    )
    data = dict(
        root=os.environ.get("DATA_ROOT", str(REPO_ROOT / "datasets/suas-synth-50k")),
        coverage_thresh=0.3,
        # VisDrone downweighting (no-ops if vd_* files were stashed out)
        vd_weight=0.4, mannequin_weight=4.0, tent_weight=2.0,
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
