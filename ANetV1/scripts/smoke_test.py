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


def v16_checks():
    """v16 (D66): cosine-weave texture channel on the v13 trunk — identity
    contract, bounded gate, D24 frequency regularization, budget."""
    m13 = ANetV1(arch="v13", use_checkpoint=False, prior_fg=0.01)
    m16 = ANetV1(arch="v16", use_checkpoint=False, prior_fg=0.01)
    n = sum(p.numel() for p in m16.parameters())
    print(f"ANetV1 v16 params: {n:,}")
    assert n < 40_000, "v16 must stay inside the ORIGINAL budget"

    missing, unexpected = m16.load_state_dict(m13.state_dict(), strict=False)
    assert not unexpected and all("weave" in k for k in missing)
    m13.eval(); m16.eval()
    x = torch.rand(2, 3, 540, 960)
    with torch.no_grad():
        d = max((m13(x)["heat"] - m16(x)["heat"]).abs().max().item(),
                (m13(x)["offset"] - m16(x)["offset"]).abs().max().item())
    print(f"  v16 identity-at-init vs donor v13: max delta {d:.2e}")
    assert d < 1e-5, "D63 identity contract broken in v16"

    with torch.no_grad():  # bounded modulation: no trunk kill-switch
        m16.backbone.weave.w_gate.fill_(-1e6)
        assert m16(x)["heat"].isfinite().all(), "v16 gate not collapse-safe"
        m16.backbone.weave.w_gate.zero_()

    m16.train()
    out = m16(x)
    l2, l1 = m16.reg_losses()  # v16: weave frequency bound (D24)
    (out["heat"].square().mean() + 3e-3 * (l2 + l1)).backward()
    p = dict(m16.named_parameters())
    assert p["backbone.weave.w_gate"].grad.abs().max() > 0, "dead valve"
    assert p["backbone.weave.freq"].grad.abs().max() > 0, "D24 reg not reaching freqs"
    assert not [k for k, q in m16.named_parameters() if q.grad is None]

    r = ANetV1.from_state_dict(m16.state_dict())
    assert r.arch == "v16" and sum(q.numel() for q in r.parameters()) == n
    print("  v16 checks passed")


def v17_checks():
    """v17 (D67): PowerBlend (A^v) injectors on the v13 trunk — identity
    contract at any channel plan, init math, exp clamp, valve wake-up."""
    import torch.nn.functional as F  # noqa: F401
    from anet.model.backbone import PowerBlend

    donor = ANetV1(arch="v13", use_checkpoint=False, prior_fg=0.01,
                   channels=(24, 48, 96), n_blocks=4)
    m = ANetV1(arch="v17", use_checkpoint=False, prior_fg=0.01,
               channels=(24, 48, 96), n_blocks=4)
    n = sum(p.numel() for p in m.parameters())
    print(f"ANetV1 v17 params (big tier): {n:,}")
    missing, unexpected = m.load_state_dict(donor.state_dict(), strict=False)
    assert not unexpected and all(".pb" in k for k in missing)
    donor.eval(); m.eval()
    x = torch.rand(2, 3, 540, 960)
    with torch.no_grad():
        d = (donor(x)["heat"] - m(x)["heat"]).abs().max().item()
    print(f"  v17 identity-at-init vs scaled-v13 donor: max delta {d:.2e}")
    assert d < 1e-5, "D63 identity contract broken in v17"

    pb = PowerBlend()  # W=0 -> A^v = 1 -> out = 3*relu(1 - tau) = 1.5
    out = pb(torch.rand(2, 3, 8, 8))
    assert torch.allclose(out, torch.full_like(out, 1.5))
    with torch.no_grad():
        pb.w.fill_(100.0)  # exp argument must saturate, not overflow
    assert pb(torch.rand(2, 3, 8, 8)).isfinite().all()

    # valve pattern: pb params silent at exact identity (w=0 is also the
    # reg minimum), alive as soon as the gains crack open
    m.train()
    with torch.no_grad():
        for site in (m.backbone.pb1, m.backbone.pb2, m.backbone.pb3,
                     m.backbone.pb4):
            site.gain.fill_(0.01)
    o = m(x)
    l2, l1 = m.reg_losses()
    (o["heat"].square().mean() + 3e-3 * (l2 + l1)).backward()
    p = dict(m.named_parameters())
    for k in ("backbone.pb1.pb.w", "backbone.pb1.pb.tau", "backbone.pb4.gain"):
        assert p[k].grad is not None and p[k].grad.abs().max() > 0, k
    assert not [k for k, q in m.named_parameters() if q.grad is None]

    r = ANetV1.from_state_dict(m.state_dict())
    assert r.arch == "v17" and r.backbone.channels == (24, 48, 96)
    print("  v17 checks passed")


def v18_checks():
    """v18 (D68): exposure-mask + bg-aux heads on the v13 trunk — identity,
    train/eval contract, DN bright-pass isolation, valve/aux gradients."""
    import torch.nn.functional as F
    from anet.model.norm import DeployNorm

    m13 = ANetV1(arch="v13", use_checkpoint=False, prior_fg=0.01)
    m = ANetV1(arch="v18", use_checkpoint=False, prior_fg=0.01)
    n = sum(p.numel() for p in m.parameters())
    print(f"ANetV1 v18 params: {n:,}")
    assert n < 40_000
    missing, unexpected = m.load_state_dict(m13.state_dict(), strict=False)
    assert not unexpected
    m13.eval(); m.eval()
    x = torch.rand(2, 3, 540, 960)
    with torch.no_grad():
        d = (m13(x)["heat"] - m(x)["heat"]).abs().max().item()
    print(f"  v18 identity-at-init vs donor v13: max delta {d:.2e}")
    assert d < 1e-5
    assert "aux_bg" not in m(x), "bg head must be train-only"

    m.train()
    out = m(x)
    assert out["aux_bg"].shape == (2, 1, 27, 48)
    # bright pass must not contaminate front DN stats: pendings must equal a
    # normal-branch-only front pass
    pend = {k: mod._pending[0].clone() for k, mod in m.named_modules()
            if isinstance(mod, DeployNorm) and mod._pending is not None
            and any(t in k for t in ("stem_norm", "down4", "block4"))}
    m.backbone._front(x)
    for k, v in pend.items():
        assert torch.equal(m.get_submodule(k)._pending[0], v), k

    from anet.train.losses import center_focal_loss
    heat_t = torch.zeros(2, 2, 27, 48); heat_t[:, 0, 5, 5] = 1.0
    bg_t = 1.0 - heat_t.max(1, keepdim=True).values
    loss = center_focal_loss(out["heat"], heat_t) + 0.3 * \
        F.binary_cross_entropy_with_logits(out["aux_bg"].float(), bg_t)
    loss.backward()
    p = dict(m.named_parameters())
    assert p["backbone.blend_gain"].grad.abs() > 0, "dead blend valve"
    assert p["backbone.bg_head.weight"].grad.abs().max() > 0, "dead bg head"
    assert not [k for k, q in m.named_parameters() if q.grad is None]
    r = ANetV1.from_state_dict(m.state_dict())
    assert r.arch == "v18"
    print("  v18 checks passed")


def v19_checks():
    """v19 (D69): the attribution build — A bias / B LearnedAct / C
    ExposureBumps / D quat+bg — identity, unit properties, all-live grads."""
    import torch.nn.functional as F
    from anet.model.backbone import ExposureBumps, LearnedAct
    from anet.train.losses import center_focal_loss

    la = LearnedAct()  # parametric identity: beta=1,gamma=0 == SiLU exactly
    t = torch.randn(512)
    assert torch.allclose(la(t), F.silu(t), atol=1e-6)
    eb = ExposureBumps()  # valve=0 == identity; extreme valve stays in [0,1]
    im = torch.rand(1, 3, 540, 960)
    assert torch.equal(eb(im), im.clamp(0, 1))
    with torch.no_grad():
        eb.valve.fill_(10.0); eb.head.bias.fill_(5.0)
    o = eb(im)
    assert o.isfinite().all() and o.max() <= 1.0 and o.min() >= 0.0

    m13 = ANetV1(arch="v13", use_checkpoint=False, prior_fg=0.01)
    m = ANetV1(arch="v19", use_checkpoint=False, prior_fg=0.01)
    n = sum(p.numel() for p in m.parameters())
    print(f"ANetV1 v19 params: {n:,}")
    assert n < 40_000
    missing, unexpected = m.load_state_dict(m13.state_dict(), strict=False)
    assert not unexpected
    m13.eval(); m.eval()
    x = torch.rand(2, 3, 540, 960)
    with torch.no_grad():
        d = (m13(x)["heat"] - m(x)["heat"]).abs().max().item()
    # 5e-5: LearnedAct(beta=1) = x*sigmoid(x) vs the donor's FUSED F.silu
    # kernel — ~1e-7/site rounding amplified through the DN folds. Benign.
    print(f"  v19 identity-at-init vs donor v13: max delta {d:.2e}")
    assert d < 5e-5

    m.train()
    out = m(x)
    assert "aux_bg" in out
    heat_t = torch.zeros(2, 2, 27, 48); heat_t[:, 0, 5, 5] = 1.0
    loss = center_focal_loss(out["heat"], heat_t) + 0.3 * \
        F.binary_cross_entropy_with_logits(
            out["aux_bg"].float(), 1.0 - heat_t.max(1, keepdim=True).values)
    loss.backward()
    p = dict(m.named_parameters())
    for k in ("backbone.bias1", "backbone.bumps.valve", "backbone.act.g",
              "backbone.qshift.qr", "backbone.bg_head.weight"):
        assert p[k].grad is not None and p[k].grad.abs().max() > 0, k
    assert not [k for k, q in m.named_parameters() if q.grad is None]
    r = ANetV1.from_state_dict(m.state_dict())
    assert r.arch == "v19"
    print("  v19 checks passed")


def v20_checks():
    """v20 (D70): re-render cycles — shape contract, partial v13 warm start
    (transitions dropped, everything else lands), sniff order vs v15, budget,
    all-live grads, slow-LR name."""
    from anet.train.losses import center_focal_loss

    m = ANetV1(arch="v20", use_checkpoint=False, prior_fg=0.01)
    n = sum(p.numel() for p in m.parameters())
    print(f"ANetV1 v20 params: {n:,}")
    assert n < 40_000
    # spd_proj must exist under that exact name — the trainer's slow-LR
    # group (the v15 stability fix) matches it by name
    assert any("spd_proj" in k for k, _ in m.named_parameters())

    # partial warm start: only the donor's down4/down20 have nowhere to go
    m13 = ANetV1(arch="v13", use_checkpoint=False, prior_fg=0.01)
    missing, unexpected = m.load_state_dict(m13.state_dict(), strict=False)
    assert all(k.startswith(("backbone.down4.", "backbone.down20."))
               for k in unexpected), unexpected
    transferred = len(m13.state_dict()) - len(unexpected)
    print(f"  v20 partial warm start: {transferred} donor tensors land, "
          f"{len(unexpected)} transition tensors dropped, {len(missing)} new")
    assert transferred > 0 and missing

    m.train()
    x = torch.rand(2, 3, 540, 960)
    out = m(x)
    assert out["heat"].shape == (2, 2, 27, 48)
    assert out["offset"].shape == (2, 2, 27, 48)
    heat_t = torch.zeros(2, 2, 27, 48); heat_t[:, 0, 5, 5] = 1.0
    center_focal_loss(out["heat"], heat_t).backward()
    assert not [k for k, q in m.named_parameters() if q.grad is None]

    # roundtrip must sniff v20, not v15 (both carry spd_proj)
    r = ANetV1.from_state_dict(m.state_dict())
    assert r.arch == "v20"
    print("  v20 checks passed")


def v22_checks():
    """v22 (D72-D75): peak-augmented full-rank funnel growth — budget, the
    FULL identity contract (every donor tensor lands, step 0 == v13_best
    exactly), valve wake-up, bounded-gate safety, sniff order vs v15/v20,
    slow-LR name, train/eval bg-aux contract."""
    import torch.nn.functional as F
    from anet.train.losses import center_focal_loss, offset_l1

    m13 = ANetV1(arch="v13", use_checkpoint=False, prior_fg=0.01)
    m = ANetV1(arch="v22", use_checkpoint=False, prior_fg=0.01)
    n = sum(p.numel() for p in m.parameters())
    n_bg = m.backbone.bg_head.weight.numel() + m.backbone.bg_head.bias.numel()
    print(f"ANetV1 v22 params: {n:,} ({n - n_bg:,} deployed + {n_bg} train-only bg)")
    assert n < 100_000, "v22 param budget exceeded (pre-registered 100k)"
    # slow-LR contract: the funnel must carry the spd_proj name (v15 fix)
    assert any("spd_proj" in k for k, _ in m.named_parameters())

    # THE D72 contract: full-identity growth. Every donor tensor lands
    # (weights AND DeployNorm buffers — no donor module's input changes at
    # step 0), and the warm-started v22 IS the donor, exactly.
    missing, unexpected = m.load_state_dict(m13.state_dict(), strict=False)
    assert not unexpected, f"donor tensors with nowhere to go: {unexpected}"
    new_ok = ("backbone.spd_proj.", "backbone.peak_proj.",
              "backbone.spd_norm.", "backbone.spd_gain", "backbone.bg_head.")
    assert all(k.startswith(new_ok) for k in missing), missing
    m13.eval(); m.eval()
    x = torch.rand(2, 3, 540, 960)
    with torch.no_grad():
        o13, o22 = m13(x), m(x)
    d = max((o13["heat"] - o22["heat"]).abs().max().item(),
            (o13["offset"] - o22["offset"]).abs().max().item())
    print(f"  v22 identity-at-init vs donor v13: max delta {d:.2e}")
    assert d < 1e-6, "D72 identity contract broken — growth is not a no-op at init"
    assert "aux_bg" not in m(x), "bg head must be train-only"

    # bounded funnel valve: 2*tanh saturates at |2| — the third-time law
    # applied to the gain that gates ~68% of all new capacity. Extreme
    # values must stay finite and cannot shut the donor path down (the
    # branch is ADDITIVE; down20's contribution is untouched).
    with torch.no_grad():
        m.backbone.spd_gain.fill_(-1e6)
        assert m(x)["heat"].isfinite().all(), "spd valve not collapse-safe"
        m.backbone.spd_gain.fill_(1e6)
        assert m(x)["heat"].isfinite().all(), "spd valve not collapse-safe"
        m.backbone.spd_gain.zero_()

    # valves alive at the exact identity point (product/add rule); the
    # branch weights behind spd_gain wake one optimizer step later — crack
    # the gain open (v17 idiom) and assert everything is live.
    m.train()
    out, bg = m.backbone(x)
    heat_t = torch.zeros(2, 2, 27, 48); heat_t[:, 0, 5, 5] = 1.0
    reg_mask = torch.zeros(2, 1, 27, 48); reg_mask[:, 0, 5, 5] = 1.0
    loss = center_focal_loss(out[:, 0:2], heat_t) + \
        offset_l1(out[:, 2:4], torch.zeros(2, 2, 27, 48), reg_mask) + 0.3 * \
        F.binary_cross_entropy_with_logits(
            bg.float(), 1.0 - heat_t.max(1, keepdim=True).values)
    loss.backward()
    p = dict(m.named_parameters())
    for k in ("backbone.spd_gain", "backbone.bg_head.weight"):
        assert p[k].grad is not None and p[k].grad.abs().max() > 0, f"dead valve: {k}"
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        m.backbone.spd_gain.fill_(0.01)
    out2 = m(x)
    l2, l1 = m.reg_losses()
    (center_focal_loss(out2["heat"], heat_t) + 0.3 *
     F.binary_cross_entropy_with_logits(
         out2["aux_bg"].float(),
         1.0 - heat_t.max(1, keepdim=True).values) + 1e-4 * (l2 + l1)).backward()
    assert not [k for k, q in m.named_parameters() if q.grad is None], \
        "v22 params without grad after valve crack"
    assert p["backbone.spd_proj.weight"].grad.abs().max() > 0, "funnel not live"
    assert p["backbone.peak_proj.weight"].grad.abs().max() > 0, "peak path not live"
    m.zero_grad(set_to_none=True)

    # from-scratch cold start stays finite/near-prior (Kaiming + DN seeding)
    ms = ANetV1(arch="v22", use_checkpoint=False, prior_fg=0.01)
    ms.train()
    o, _ = ms.backbone(x)
    assert o.abs().max() < 50, "v22 cold-start activations blew up"

    # roundtrip must sniff v22 (not v15 — both carry a spd_proj tensor)
    r = ANetV1.from_state_dict(m.state_dict())
    assert r.arch == "v22" and r.backbone.channels == (16, 32, 64)
    assert sum(q.numel() for q in r.parameters()) == n
    print("  v22 checks passed")


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
    v16_checks()
    v17_checks()
    v18_checks()
    v19_checks()
    v20_checks()
    v22_checks()
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
