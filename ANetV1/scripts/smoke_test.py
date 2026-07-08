"""Param-count assert + forward/backward on CPU and MPS + dataset sanity."""

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402


def run_device(model, device):
    label = str(device)
    if device.type == "cuda":
        hip = getattr(torch.version, "hip", None)
        if hip:
            label = f"{device} (ROCm hip {hip})"
        print(f"  {label}: fwd+bwd starting (first call may autotune MIOpen — looks hung, ~30-60s)...",
              flush=True)
    else:
        print(f"  {device}: fwd+bwd...", flush=True)

    model = model.to(device)
    x = torch.rand(2, 3, 540, 960, device=device)
    t0 = time.time()
    cells = model(x)
    loss = cells.square().mean()
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    extra = ""
    if device.type == "cuda":
        mib = torch.cuda.max_memory_allocated(device) / 2**20
        extra = f" | peak vram {mib:.0f} MiB"
    print(f"  {device}: out {tuple(cells.shape)} fwd+bwd {elapsed:.2f}s{extra}", flush=True)
    model.zero_grad(set_to_none=True)


def main():
    # use_checkpoint=False: smoke should be fast; MI300X training config also disables it
    for stem in ("edge_dq", "highpass"):
        for per_ch in (False, True):
            m = ANetV1(use_checkpoint=False, stem=stem, path_a_per_channel=per_ch)
            n = sum(p.numel() for p in m.parameters())
            print(f"ANetV1 params ({stem}, path_a_per_channel={per_ch}): {n:,}")
            # ~17-18k shared Path A (D13); ~21-22k per-channel Path A (D37) at hidden=16
            assert 15_000 < n < 24_000, "param count off spec (ARCHITECTURE.md §5)"
    model = ANetV1(use_checkpoint=False, stem="edge_dq")  # training default (D33 + D37)
    assert model.n_win == 5035 and model.nh == 53 and model.nw == 95

    model.train()
    skip_cpu = os.environ.get("ANET_SMOKE_SKIP_CPU", "").lower() in ("1", "true", "yes")
    if not skip_cpu:
        run_device(model, torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        run_device(model, torch.device("cuda"))
    elif torch.backends.mps.is_available():
        run_device(model, torch.device("mps"))

    root = Path(os.environ.get("ANET_DATA_ROOT")
                or Path(__file__).parents[2] / "datasets/suas-synth-50k")
    if root.is_dir():
        from anet.data.dataset import SUASCells

        ds = SUASCells(root, "val")
        s = ds[0]
        print(f"dataset: {len(ds)} val imgs | image {tuple(s['image'].shape)} "
              f"grid {tuple(s['grid'].shape)} fg cells {(s['grid'] > 0).sum().item()}")
        vd = next((i for i in range(len(ds)) if ds.is_visdrone(i)), None)
        if vd is not None:
            sv = ds[vd]
            print(f"visdrone sample ok: {sv['stem']} fg cells {(sv['grid'] > 0).sum().item()}")
    print("smoke test passed")


if __name__ == "__main__":
    main()
