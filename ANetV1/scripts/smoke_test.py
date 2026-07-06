"""Param-count assert + forward/backward on CPU and MPS + dataset sanity."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402


def run_device(model, device):
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
    print(f"  {device}: out {tuple(cells.shape)} fwd+bwd {time.time() - t0:.2f}s")
    model.zero_grad(set_to_none=True)


def main():
    model = ANetV1()
    n = sum(p.numel() for p in model.parameters())
    print(f"ANetV1 params: {n:,}")
    assert 15_000 < n < 19_000, "param count off spec (~17k, ARCHITECTURE.md §5)"
    assert model.n_win == 5035 and model.nh == 53 and model.nw == 95

    model.train()
    run_device(model, torch.device("cpu"))
    if torch.cuda.is_available():
        run_device(model, torch.device("cuda"))
    elif torch.backends.mps.is_available():
        run_device(model, torch.device("mps"))

    import os
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
