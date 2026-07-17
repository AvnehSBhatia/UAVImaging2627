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
    out = m(x)  # v9 training: dict {cells, aux, z}
    cells, aux, zmap = out["cells"], out["aux"], out["z"]
    assert cells.shape == aux.shape == (1, 3, 54, 96)
    assert zmap.shape == (1, m.head.width, 53, 95), "metric z-map shape"
    grid = torch.zeros(1, 54, 96, dtype=torch.long)
    grid[0, 10, 10] = 1
    loss = focal_norm_loss(cells, grid) + 0.3 * focal_norm_loss(aux, grid)
    l2, l1 = m.reg_losses()
    (loss + 1e-4 * (l2 + l1)).backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing}"
    m.zero_grad(set_to_none=True)

    # metric-prototype path (D56): proto_metric_loss on z + window labels must
    # reach the prototypes and the encoder.
    from anet.train.losses import proto_metric_loss
    out_m = m(x)
    wl = proto_metric_loss(out_m["z"], grid, m.head.prototypes, m.head.metric_scale())
    wl.backward()
    for name in ("head.prototypes", "head.proto_log_scale"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no metric grad to {name}"
    m.zero_grad(set_to_none=True)

    # v11 default loss (weighted FP/TP): full-grad + anti-collapse property —
    # a predict-nothing head must feel a nonzero recall pull on the true cell.
    out2 = m(x)
    cells2, aux2 = out2["cells"], out2["aux"]
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
        g64 = m64.encoder.gate(h * sc + sh)  # (N, 400) dynamic-box gate
        pooled = ((g64.unsqueeze(-1) * h).sum(1)
                  / (g64.sum(1, keepdim=True) + m64.encoder.BOX_EPS))
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


def v12_checks():
    """v12: param budget + forward shapes on the single-phase 27x48 grid."""
    m = ANetV1(arch="v12", dense=True, hidden=32, stem="edge_dq4",
               head_width=24, prior_fg=0.1, use_checkpoint=False)
    n = sum(p.numel() for p in m.parameters())
    print(f"ANetV1 v12 params: {n:,}")
    assert n < 40_000, "v12 param budget exceeded"
    assert m.nh == 27 and m.nw == 48 and m.n_win == 27 * 48

    x = torch.rand(2, 3, 540, 960)
    m.train()
    out = m(x)
    assert out["heat"].shape == (2, 2, 27, 48), out["heat"].shape
    assert out["offset"].shape == (2, 2, 27, 48), out["offset"].shape

    m.eval()
    with torch.no_grad():
        out_eval = m(x)
    assert out_eval["heat"].shape == (2, 2, 27, 48)
    assert out_eval["offset"].shape == (2, 2, 27, 48)

    from anet.train.losses import center_focal_loss, offset_l1

    m.train()
    out = m(x)
    heat_t = torch.zeros(2, 2, 27, 48)
    heat_t[:, 0, 5, 5] = 1.0
    offset_t = torch.zeros(2, 2, 27, 48)
    reg_mask = torch.zeros(2, 1, 27, 48)
    reg_mask[:, 0, 5, 5] = 1.0
    loss = center_focal_loss(out["heat"], heat_t) + offset_l1(out["offset"], offset_t, reg_mask)
    # train-only deep-supervision probe (exercises aux_center's gradient path)
    if "aux_heat" in out:
        loss = loss + center_focal_loss(out["aux_heat"], heat_t)
    l2, l1 = m.reg_losses()
    (loss + 1e-4 * (l2 + l1)).backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, f"v12 params without grad: {missing}"
    m.zero_grad(set_to_none=True)

    sd = m.state_dict()
    m2 = ANetV1.from_state_dict(sd)
    assert m2.arch == "v12" and m2.encoder.hidden == 32
    print("  v12 checks passed")


def v13_checks():
    """v13 (D58): conv backbone — param budget, grid, v12-contract shapes,
    full gradient flow, cold-start sanity (Kaiming init keeps activations
    unit-scale so DeployNorm seeding converges), state_dict roundtrip."""
    m = ANetV1(arch="v13", use_checkpoint=False, prior_fg=0.01)
    n = sum(p.numel() for p in m.parameters())
    print(f"ANetV1 v13 params: {n:,}")
    assert n < 40_000, "v13 param budget exceeded"
    assert m.nh == 27 and m.nw == 48 and m.n_win == 27 * 48

    x = torch.rand(2, 3, 540, 960)
    m.train()
    out = m(x)
    assert out["heat"].shape == (2, 2, 27, 48), out["heat"].shape
    assert out["offset"].shape == (2, 2, 27, 48), out["offset"].shape
    assert "aux_heat" not in out, "v13 has no deep-supervision probe"
    # cold start: first TRAIN forward runs on init stats (identity affine);
    # Kaiming init must keep it finite and near the prior, not 1e23 (the
    # failure mode that motivated the explicit init — see backbone.py)
    assert out["heat"].abs().max() < 50, "v13 cold-start activations blew up"

    from anet.train.losses import center_focal_loss, offset_l1

    heat_t = torch.zeros(2, 2, 27, 48)
    heat_t[:, 0, 5, 5] = 1.0
    offset_t = torch.zeros(2, 2, 27, 48)
    reg_mask = torch.zeros(2, 1, 27, 48)
    reg_mask[:, 0, 5, 5] = 1.0
    loss = center_focal_loss(out["heat"], heat_t) + \
        offset_l1(out["offset"], offset_t, reg_mask)
    l2, l1 = m.reg_losses()
    (loss + 1e-4 * (l2 + l1)).backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, f"v13 params without grad: {missing}"
    m.zero_grad(set_to_none=True)

    m.eval()
    with torch.no_grad():
        out_eval = m(x)
    assert out_eval["heat"].shape == (2, 2, 27, 48)
    # eval forward at init sits at the fg prior (bias -4.6), not saturated
    p = torch.sigmoid(out_eval["heat"])
    assert 0.001 < p.mean() < 0.1, f"init prior off: {p.mean():.4f}"

    m2 = ANetV1.from_state_dict(m.state_dict())
    assert m2.arch == "v13"
    assert sum(p.numel() for p in m2.parameters()) == n
    print("  v13 checks passed")


def v14_checks():
    """v14 (D59-D63): param budget, shapes, grad flow, and THE contract —
    a v14 warm-started from a v13 state_dict computes exactly that v13's
    function at step 0 (every new module identity/zero-gamma init)."""
    m13 = ANetV1(arch="v13", use_checkpoint=False, prior_fg=0.01)
    m14 = ANetV1(arch="v14", use_checkpoint=False, prior_fg=0.01)
    n = sum(p.numel() for p in m14.parameters())
    print(f"ANetV1 v14 params: {n:,}")
    assert n < 40_000, "v14 param budget exceeded"

    # identity-at-init contract (D63): warm-start == the donor v13, exactly
    missing, unexpected = m14.load_state_dict(m13.state_dict(), strict=False)
    assert not unexpected, f"v13 keys not consumed by v14: {unexpected}"
    assert all(k.startswith("backbone.") for k in missing), missing
    m13.eval(); m14.eval()
    x = torch.rand(2, 3, 540, 960)
    with torch.no_grad():
        o13, o14 = m13(x), m14(x)
    d = max((o13["heat"] - o14["heat"]).abs().max().item(),
            (o13["offset"] - o14["offset"]).abs().max().item())
    print(f"  v14 identity-at-init vs donor v13: max delta {d:.2e}")
    assert d < 1e-5, "D63 identity contract broken — a new module is not a no-op at init"

    # from-scratch: shapes, cold-start sanity, full loss backward, valve grads
    m = ANetV1(arch="v14", use_checkpoint=False, prior_fg=0.01)
    m.train()
    out = m(x)
    assert out["heat"].shape == (2, 2, 27, 48) and out["offset"].shape == (2, 2, 27, 48)
    assert out["heat"].abs().max() < 50, "v14 cold-start activations blew up"
    from anet.train.losses import center_focal_loss, offset_l1
    heat_t = torch.zeros(2, 2, 27, 48); heat_t[:, 0, 5, 5] = 1.0
    reg_mask = torch.zeros(2, 1, 27, 48); reg_mask[:, 0, 5, 5] = 1.0
    loss = center_focal_loss(out["heat"], heat_t) + \
        offset_l1(out["offset"], torch.zeros(2, 2, 27, 48), reg_mask)
    l2, l1 = m.reg_losses()
    (loss + 1e-4 * (l2 + l1)).backward()
    assert not [k for k, p in m.named_parameters() if p.grad is None], "v14 no-grad params"
    # each identity valve must feel gradient at the identity point, or its
    # branch can never open (zero-gamma pattern: branch params wake up one
    # optimizer step after their valve moves — that part is expected)
    params = dict(m.named_parameters())
    for valve in ("backbone.skip_gain", "backbone.block_extra.gain",
                  "backbone.tex.w_gate", "backbone.noise.weight"):
        assert params[valve].grad.abs().max() > 0, f"dead valve: {valve}"
    # the texture modulation must be BOUNDED: no parameter setting may zero
    # the trunk (the unbounded first draft collapsed the from-scratch MI300X
    # run — see TextureGate docstring). Factor = 1 + tanh(w)*g > 1 - g > 0.
    with torch.no_grad():
        m.backbone.tex.w_gate.fill_(-1e6)  # worst case: maximum suppression
        out_sup = m(x)
    assert out_sup["heat"].isfinite().all(), "texture gate not collapse-safe"
    with torch.no_grad():
        m.backbone.tex.w_gate.zero_()

    m2 = ANetV1.from_state_dict(m.state_dict())
    assert m2.arch == "v14" and sum(p.numel() for p in m2.parameters()) == n
    print("  v14 checks passed")


def v15_checks():
    """v15 (D64/D65): SPD projection + capacity tiers — shapes, grads,
    lossless rearrangement property, tier construction, scaled roundtrips."""
    from anet.train.losses import center_focal_loss, offset_l1

    m = ANetV1(arch="v15", use_checkpoint=False, prior_fg=0.01)  # tier S
    n_s = sum(p.numel() for p in m.parameters())
    m_m = ANetV1(arch="v15", use_checkpoint=False, prior_fg=0.01,
                 channels=(16, 48, 96), n_blocks=4)              # tier M
    n_m = sum(p.numel() for p in m_m.parameters())
    print(f"ANetV1 v15 params: tier-S {n_s:,} | tier-M {n_m:,}")
    assert n_s < 100_000 and n_m < 300_000, "v15 tiers off their budgets"

    x = torch.rand(2, 3, 540, 960)
    m.train()
    out = m(x)
    assert out["heat"].shape == (2, 2, 27, 48) and out["offset"].shape == (2, 2, 27, 48)
    assert out["heat"].abs().max() < 50, "v15 cold-start activations blew up"
    heat_t = torch.zeros(2, 2, 27, 48); heat_t[:, 0, 5, 5] = 1.0
    reg_mask = torch.zeros(2, 1, 27, 48); reg_mask[:, 0, 5, 5] = 1.0
    loss = center_focal_loss(out["heat"], heat_t) + \
        offset_l1(out["offset"], torch.zeros(2, 2, 27, 48), reg_mask)
    l2, l1 = m.reg_losses()
    (loss + 1e-4 * (l2 + l1)).backward()
    assert not [k for k, p in m.named_parameters() if p.grad is None], "v15 no-grad params"

    # the D64 point: pixel_unshuffle(5) is a lossless rearrangement — every
    # s4 feature value appears exactly once in the (25*ch, 27, 48) tensor
    import torch.nn.functional as F
    t = torch.arange(32 * 135 * 240, dtype=torch.float32).reshape(1, 32, 135, 240)
    u = F.pixel_unshuffle(t, 5)
    assert u.shape == (1, 32 * 25, 27, 48)
    assert torch.equal(torch.sort(u.flatten()).values, torch.sort(t.flatten()).values)

    # roundtrips: tier-M v15 and a scaled v13 both sniff back exactly
    r = ANetV1.from_state_dict(m_m.state_dict())
    assert r.arch == "v15" and r.backbone.channels == (16, 48, 96)
    assert sum(p.numel() for p in r.parameters()) == n_m
    v13w = ANetV1(arch="v13", use_checkpoint=False, channels=(24, 48, 96),
                  n_blocks=4, prior_fg=0.01)
    r13 = ANetV1.from_state_dict(v13w.state_dict())
    assert r13.arch == "v13" and r13.backbone.channels == (24, 48, 96)
    print("  v15 checks passed")


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
    v12_checks()
    v13_checks()
    v14_checks()
    v15_checks()
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
