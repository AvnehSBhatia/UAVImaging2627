"""On-GPU training-step profiler — run this ON the MI300X to end the guessing.

    python scripts/profile_step.py            # real preset config, random data
    ANET_BATCH=64 python scripts/profile_step.py

Prints: wall ms/step, GPU-busy ms/step (util), kernel LAUNCHES per step, and the
top kernels by device time. The diagnosis is read straight off these:
  - launches/step high (~1000s) + low util  -> launch-bound (HIP graphs / bigger batch)
  - one MIOpen conv solver dominating device time -> bad solver (find-db / layout)
  - wall >> GPU-busy with few launches -> host/loader-bound
"""
import os, sys, time
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anet import ANetV1  # noqa: E402
from anet.train.presets import anet_cfg  # noqa: E402
from anet.train.losses import focal_tversky_loss, focal_loss  # noqa: E402

cfg = anet_cfg(hidden=24)
c = cfg.train
dev = torch.device("cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))
B = c.batch_size
print(f"device={dev} batch={B} compile={bool(c.compile)} ckpt={c.use_checkpoint} "
      f"amp={c.amp} | model hidden=24 edge_dq", flush=True)

if dev.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

model = ANetV1(hidden=24, stem="edge_dq", use_checkpoint=c.use_checkpoint).to(dev).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=(dev.type == "cuda"))
amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(c.amp)

def step(img, grid):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(dev.type, amp_dtype, enabled=amp_dtype is not None):
        cells = model(img)
        loss = focal_tversky_loss(cells, grid, alpha=c.tversky_alpha, beta=c.tversky_beta,
                                  gamma=c.ft_gamma, smooth=c.ft_smooth) \
            + c.ft_anchor_weight * focal_loss(cells, grid, gamma=c.focal_gamma,
                                              alpha=tuple(c.ft_anchor_alpha))
    l2, l1 = model.reg_losses()
    loss = loss + c.l2_score_reg * l2 + c.l1_kernel_reg * l1
    loss.backward()
    opt.step()
    return loss

if c.compile:
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    model = torch.compile(model, mode=c.compile_mode, dynamic=False)

img = torch.rand(B, 3, 540, 960, device=dev)
grid = torch.randint(0, 3, (B, 54, 96), device=dev)

print("warmup (compile + MIOpen autotune — slow)...", flush=True)
for _ in range(8):
    step(img, grid)
if dev.type == "cuda":
    torch.cuda.synchronize()

N = 20
t0 = time.time()
for _ in range(N):
    step(img, grid)
if dev.type == "cuda":
    torch.cuda.synchronize()
wall = (time.time() - t0) / N * 1e3
print(f"\nwall: {wall:.1f} ms/step  ({wall / B:.2f} ms/img, {B} img/step)", flush=True)

if dev.type == "cuda":
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts) as prof:
        for _ in range(5):
            step(img, grid)
        torch.cuda.synchronize()
    evs = prof.key_averages()
    dev_total = sum(e.self_device_time_total for e in evs) / 5 / 1e3  # ms/step
    launches = sum(e.count for e in evs if e.device_type.name == "CUDA") / 5
    print(f"GPU-busy: {dev_total:.1f} ms/step  (util {100*dev_total/wall:.0f}% of wall)")
    print(f"kernel launches/step: ~{launches:.0f}")
    print("\ntop kernels by device time (ms/step, count/step):")
    ranked = sorted(evs, key=lambda e: e.self_device_time_total, reverse=True)[:20]
    for e in ranked:
        dt = e.self_device_time_total / 5 / 1e3
        if dt < 0.05:
            break
        print(f"  {dt:7.2f} ms  x{e.count/5:5.0f}  {e.key[:70]}")
    print("\nverdict: " + (
        "LAUNCH-BOUND (util<25%, 1000s of launches) — HIP graphs / bigger batch"
        if dev_total < 0.4 * wall else
        "GPU-bound — look at the top kernel(s) above"))
