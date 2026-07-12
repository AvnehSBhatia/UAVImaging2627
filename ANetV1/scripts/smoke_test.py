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


def v9_checks():
    """v9: param budget, train/eval forward shapes, dense-vs-windowed parity,
    chunked-mirror parity, state_dict roundtrip (all CPU, fp64 for parity)."""
    import torch.nn.functional as F

    from anet.train.fused import _derive_params, pool_from_params
    from anet.train.losses import focal_norm_loss, weighted_fp_tp_loss

    m = ANetV1(arch="v9", stem="edge_dq4", hidden=32, h1=48,
               use_checkpoint=False, aux_head=True, prior_fg=0.05)
    n = sum(p.numel() for p in m.parameters())
    n_aux = m.aux.weight.numel() + m.aux.bias.numel()
    print(f"ANetV1 v9 params: {n:,} ({n - n_aux:,} deployed + {n_aux} aux)")
    assert n < 40_000, "v9 param budget exceeded"

    x = torch.rand(1, 3, 540, 960)
    m.train()
    cells, aux = m(x)
    assert cells.shape == aux.shape == (1, 3, 54, 96)
    grid = torch.zeros(1, 54, 96, dtype=torch.long)
    grid[0, 10, 10] = 1
    loss = focal_norm_loss(cells, grid) + 0.3 * focal_norm_loss(aux, grid)
    l2, l1 = m.reg_losses()
    (loss + 1e-4 * (l2 + l1)).backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing}"
    m.zero_grad(set_to_none=True)

    # v11 default loss (weighted FP/TP): full-grad + anti-collapse property —
    # a predict-nothing head must feel a nonzero recall pull on the true cell.
    cells2, aux2 = m(x)
    ftp = weighted_fp_tp_loss(cells2, grid) + 0.3 * weighted_fp_tp_loss(aux2, grid)
    ftp.backward()
    assert not [k for k, p in m.named_parameters() if p.grad is None], \
        "fp_tp: params without grad"
    m.zero_grad(set_to_none=True)
    collapsed = torch.full((1, 3, 54, 96), 0.0)
    collapsed[:, 0] = 12.0  # softmax ~ all background -> the collapse point
    collapsed.requires_grad_(True)
    weighted_fp_tp_loss(collapsed, grid).backward()
    # gradient on the mannequin logit of the one true cell must push it UP
    assert collapsed.grad[0, 1, 10, 10] < 0, \
        "fp_tp lost its anti-collapse recall gradient at predict-nothing"

    m.eval()
    m64 = m.double()
    with torch.no_grad():
        feat = m64._features(x.double())
        xp = m64._phase_batch(feat, "constant")
        p_dense = m64._interleave(m64.encoder.pool_features_dense(xp), 1)
        # windowed token reference
        win = F.unfold(feat, 20, stride=10)
        win = win.reshape(1, m64.feat, 400, m64.n_win).permute(0, 3, 2, 1)
        toks = torch.cat([win, m64.uv.expand(1, m64.n_win, -1, -1)], -1)
        t = toks.reshape(-1, 400, m64.in_dim)
        for r in m64.encoder.rounds:
            t = r.forward_tokens(t)
        h = F.silu(m64.encoder.fc1(t))
        sc, sh = m64.encoder.pool_norm.fold()
        pooled = (m64.encoder.gate(h * sc + sh).unsqueeze(-1) * h).mean(1)
        p_win = pooled.reshape(1, m64.n_win, -1).permute(0, 2, 1).reshape(
            1, -1, m64.nh, m64.nw)
        d1 = (p_dense - p_win).abs().max().item()
        # chunked-backward mirror (the fused kernels implement this same math)
        p = _derive_params(m64.encoder)
        p = {k: v.double() for k, v in p.items()}
        d2 = (m64._interleave(pool_from_params(xp, p), 1) - p_dense).abs().max().item()
    print(f"  v9 parity: dense-vs-windowed {d1:.2e}, mirror-vs-dense {d2:.2e}")
    assert d1 < 1e-9 and d2 < 1e-8, "v9 path parity broken"
    m2 = ANetV1.from_state_dict({k: v.float() for k, v in m64.state_dict().items()})
    assert m2.arch == "v9" and m2.encoder.hidden == 32
    print("  v9 checks passed")


def main():
    # use_checkpoint=False: smoke should be fast; MI300X training config also disables it
    for stem in ("edge_dq", "highpass"):
        for per_ch in (False, True):
            m = ANetV1(use_checkpoint=False, stem=stem, path_a_per_channel=per_ch)
            n = sum(p.numel() for p in m.parameters())
            print(f"ANetV1 params ({stem}, path_a_per_channel={per_ch}): {n:,}")
            # ~17-18k shared Path A (D13); ~21-22k per-channel Path A (D37) at hidden=16
            assert 15_000 < n < 24_000, "param count off spec (ARCHITECTURE.md §5)"
    v9_checks()
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
