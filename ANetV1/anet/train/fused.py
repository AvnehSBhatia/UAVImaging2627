"""Fused Stage-1 training op (v9, D40) — Triton.

The encoder is threadgroup-shaped (D35 proved it with Metal at eval): one
20x20 window's entire pipeline — 3 cosine-gated mixing rounds, fc1, the
cosine-gated pool — fits in registers. Graph runtimes instead materialize
every full-resolution intermediate to HBM (the measured 0.65-2.9 GiB/img and
the 517 s epochs). This op runs the whole per-token Stage 1 in ONE kernel per
direction:

  forward : feat (B, 15, 540, 960) -> pooled (B, 48, 53, 95)
            reads the stem map once (phase offsets computed in-kernel — the
            4-phase crops are never materialized), writes 0.7 MB of pooled
            windows, and accumulates the DeployNorm batch stats via slotted
            atomics on the way through.
  backward: recomputes each tile in registers (nothing is saved except the
            stem map) and emits d_feat + all parameter grads.

DeployNorm (D39) is what makes this legal: normalization is a per-channel
affine folded into the score/gate weights, so no batch coupling crosses tile
boundaries.

Modes (ANET_FUSED_BWD): "triton" (default) — hand-derived backward kernel;
"chunked" — autograd through a pure-torch mirror of the same math in height-
band chunks (slower, but gradients come from autograd; the startup parity
check verifies the triton backward against it on real data and demotes
automatically on mismatch).
"""

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # Mac / CPU boxes — PyTorch dense path only
    HAS_TRITON = False

IMG_H, IMG_W, WIN = 540, 960, 20
NH, NW = 53, 95
PH, PW = 27, 48          # per-phase tile grid
N_TOK = WIN * WIN        # 400
SLOTS = 512              # atomic-contention spreading for stats/param grads


def fused_available():
    return HAS_TRITON and torch.cuda.is_available()


# ============================================================ triton kernels
if HAS_TRITON:

    @triton.jit
    def _silu(x):
        return x * tl.sigmoid(x)

    @triton.jit
    def _blur(t, K):
        # per-tile separable gaussian == K @ T @ K (K symmetric banded, padded
        # to 32x32 with zeros == the zero padding at tile borders)
        return tl.dot(K, tl.dot(t, K, allow_tf32=False), allow_tf32=False)

    @triton.jit
    def _fwd_kernel(
        feat_ptr, out_ptr,
        Wr_ptr, Wf_ptr, bfs_ptr, K_ptr, phi_ptr,
        W1_ptr, b1_ptr, Vgs_ptr, bg_ptr, phig_ptr,
        rsum_ptr, rsq_ptr,        # (SLOTS, 3, 3)  rgb round-input stats
        fsum_ptr, fsq_ptr,        # (SLOTS, 14)    frozen stats
        psum_ptr, psq_ptr,        # (SLOTS, 48)    fc1-output stats
        B,
        sfb, sfc, sfh, sfw,
        sob, soc, soh, sow,
        COLLECT: tl.constexpr,
        NFRO: tl.constexpr,       # 12 edge channels (+2 uv handled inline)
        H1: tl.constexpr,         # 48
    ):
        pid = tl.program_id(0)
        tx = pid % PW
        t = pid // PW
        ty = t % PH
        t = t // PH
        ph = t % 4
        b = t // 4
        oy = (ph // 2) * 10
        ox = (ph % 2) * 10

        R = tl.arange(0, 32)[:, None]
        C = tl.arange(0, 32)[None, :]
        tok = (R < WIN) & (C < WIN)
        # replicate-pad clamp: phase crop covers oy..oy+hp-1
        hp = ((IMG_H - oy) // WIN) * WIN
        wp = ((IMG_W - ox) // WIN) * WIN
        gy = tl.minimum(oy + ty * WIN + R, oy + hp - 1)
        gx = tl.minimum(ox + tx * WIN + C, ox + wp - 1)
        base = feat_ptr + b * sfb
        slot = pid % SLOTS

        # ---- rgb tiles
        r0 = tl.load(base + 0 * sfc + gy * sfh + gx * sfw, mask=tok, other=0.0).to(tl.float32)
        r1 = tl.load(base + 1 * sfc + gy * sfh + gx * sfw, mask=tok, other=0.0).to(tl.float32)
        r2 = tl.load(base + 2 * sfc + gy * sfh + gx * sfw, mask=tok, other=0.0).to(tl.float32)
        rgb = [r0, r1, r2]

        # ---- frozen score contributions for all 3 rounds (9 accumulators),
        #      seeded with the full folded bias
        fs = []
        for r in tl.static_range(3):
            for k in tl.static_range(3):
                fs.append(tl.zeros((32, 32), tl.float32)
                          + tl.load(bfs_ptr + r * 3 + k))
        for ch in tl.static_range(NFRO):
            e = tl.load(base + (ch + 3) * sfc + gy * sfh + gx * sfw,
                        mask=tok, other=0.0).to(tl.float32)
            for r in tl.static_range(3):
                for k in tl.static_range(3):
                    w = tl.load(Wf_ptr + (r * 3 + k) * (NFRO + 2) + ch)
                    fs[r * 3 + k] += w * e
            if COLLECT:
                tl.atomic_add(fsum_ptr + slot * (NFRO + 2) + ch, tl.sum(tl.where(tok, e, 0.0)))
                tl.atomic_add(fsq_ptr + slot * (NFRO + 2) + ch, tl.sum(tl.where(tok, e * e, 0.0)))
        # uv channels (analytic, frozen indices NFRO / NFRO+1)
        u = (C.to(tl.float32) + 0.5) / WIN
        v = (R.to(tl.float32) + 0.5) / WIN
        u = tl.where(tok, u, 0.0)
        v = tl.where(tok, v, 0.0)
        for r in tl.static_range(3):
            for k in tl.static_range(3):
                wu = tl.load(Wf_ptr + (r * 3 + k) * (NFRO + 2) + NFRO)
                wv = tl.load(Wf_ptr + (r * 3 + k) * (NFRO + 2) + NFRO + 1)
                fs[r * 3 + k] += wu * u + wv * v
        if COLLECT:
            tl.atomic_add(fsum_ptr + slot * (NFRO + 2) + NFRO, tl.sum(u))
            tl.atomic_add(fsq_ptr + slot * (NFRO + 2) + NFRO, tl.sum(u * u))
            tl.atomic_add(fsum_ptr + slot * (NFRO + 2) + NFRO + 1, tl.sum(v))
            tl.atomic_add(fsq_ptr + slot * (NFRO + 2) + NFRO + 1, tl.sum(v * v))

        # ---- 3 mixing rounds
        for r in tl.static_range(3):
            if COLLECT:
                for j in tl.static_range(3):
                    tl.atomic_add(rsum_ptr + slot * 9 + r * 3 + j,
                                  tl.sum(tl.where(tok, rgb[j], 0.0)))
                    tl.atomic_add(rsq_ptr + slot * 9 + r * 3 + j,
                                  tl.sum(tl.where(tok, rgb[j] * rgb[j], 0.0)))
            s = []
            for k in tl.static_range(3):
                acc = fs[r * 3 + k]
                for j in tl.static_range(3):
                    acc += tl.load(Wr_ptr + r * 9 + k * 3 + j) * rgb[j]
                s.append(tl.where(tok, acc, 0.0))
            Kr = tl.load(K_ptr + r * 1024 + tl.arange(0, 32)[:, None] * 32
                         + tl.arange(0, 32)[None, :])
            s1b = _blur(s[0], Kr)
            s2b = _blur(s[1], Kr)
            phi = tl.load(phi_ptr + r)
            a = tl.math.tanh(s2b * s[2])
            gate = tl.sigmoid(s1b * tl.cos(math.pi * a + phi))
            gate = tl.where(tok, gate, 0.0)
            for j in tl.static_range(3):
                pooled = tl.sum(gate * rgb[j]) / N_TOK
                z = rgb[j] + pooled
                rgb[j] = tl.where(tok, _silu(z), 0.0)

        # ---- tail: fc1 -> gated pool (two channel passes, h never stored)
        fro = []
        for ch in tl.static_range(NFRO):
            fro.append(tl.load(base + (ch + 3) * sfc + gy * sfh + gx * sfw,
                               mask=tok, other=0.0).to(tl.float32))
        fro.append(u)
        fro.append(v)
        xin = rgb + fro  # 3 + NFRO + 2 = 17 tiles

        sp0 = tl.zeros((32, 32), tl.float32) + tl.load(bg_ptr + 0)
        sp1 = tl.zeros((32, 32), tl.float32) + tl.load(bg_ptr + 1)
        sp2 = tl.zeros((32, 32), tl.float32) + tl.load(bg_ptr + 2)
        for j in tl.static_range(H1):
            pre = tl.zeros((32, 32), tl.float32) + tl.load(b1_ptr + j)
            for ch in tl.static_range(NFRO + 5):
                pre += tl.load(W1_ptr + j * (NFRO + 5) + ch) * xin[ch]
            h = _silu(pre)
            h = tl.where(tok, h, 0.0)
            sp0 += tl.load(Vgs_ptr + 0 * H1 + j) * h
            sp1 += tl.load(Vgs_ptr + 1 * H1 + j) * h
            sp2 += tl.load(Vgs_ptr + 2 * H1 + j) * h
            if COLLECT:
                tl.atomic_add(psum_ptr + slot * H1 + j, tl.sum(h))
                tl.atomic_add(psq_ptr + slot * H1 + j, tl.sum(h * h))
        phig = tl.load(phig_ptr)
        ag = tl.math.tanh(sp1 * sp2)
        gate2 = tl.sigmoid(sp0 * tl.cos(math.pi * ag + phig))
        gate2 = tl.where(tok, gate2, 0.0)

        gy_out = oy // 10 + 2 * ty
        gx_out = ox // 10 + 2 * tx
        if (gy_out < NH) & (gx_out < NW):
            optr = out_ptr + b * sob + gy_out * soh + gx_out * sow
            for j in tl.static_range(H1):
                pre = tl.zeros((32, 32), tl.float32) + tl.load(b1_ptr + j)
                for ch in tl.static_range(NFRO + 5):
                    pre += tl.load(W1_ptr + j * (NFRO + 5) + ch) * xin[ch]
                h = tl.where(tok, _silu(pre), 0.0)
                tl.store(optr + j * soc, tl.sum(gate2 * h) / N_TOK)

    @triton.jit
    def _bwd_kernel(
        feat_ptr, dout_ptr, dfeat_ptr,
        Wr_ptr, Wf_ptr, bfs_ptr, K_ptr, dKds_ptr, phi_ptr,
        W1_ptr, b1_ptr, Vgs_ptr, bg_ptr, phig_ptr,
        dWr_ptr, dWf_ptr, dbfs_ptr, dsig_ptr, dphi_ptr,
        dW1_ptr, db1_ptr, dVgs_ptr, dbg_ptr, dphig_ptr,
        B,
        sfb, sfc, sfh, sfw,
        sob, soc, soh, sow,
        NFRO: tl.constexpr,
        H1: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tx = pid % PW
        t = pid // PW
        ty = t % PH
        t = t // PH
        ph = t % 4
        b = t // 4
        oy = (ph // 2) * 10
        ox = (ph % 2) * 10
        gy_out = oy // 10 + 2 * ty
        gx_out = ox // 10 + 2 * tx
        if (gy_out >= NH) | (gx_out >= NW):
            return  # padded tile: cropped from the output, no gradient flows

        R = tl.arange(0, 32)[:, None]
        C = tl.arange(0, 32)[None, :]
        tok = (R < WIN) & (C < WIN)
        hp = ((IMG_H - oy) // WIN) * WIN
        wp = ((IMG_W - ox) // WIN) * WIN
        gy = tl.minimum(oy + ty * WIN + R, oy + hp - 1)
        gx = tl.minimum(ox + tx * WIN + C, ox + wp - 1)
        base = feat_ptr + b * sfb
        slot = pid % SLOTS

        u = tl.where(tok, (C.to(tl.float32) + 0.5) / WIN, 0.0)
        v = tl.where(tok, (R.to(tl.float32) + 0.5) / WIN, 0.0)

        # ---------------- recompute rounds, keeping each round's rgb INPUT
        rgb = [tl.load(base + j * sfc + gy * sfh + gx * sfw, mask=tok,
                       other=0.0).to(tl.float32) for j in tl.static_range(3)]
        fro = [tl.load(base + (ch + 3) * sfc + gy * sfh + gx * sfw, mask=tok,
                       other=0.0).to(tl.float32) for ch in tl.static_range(NFRO)]
        fro.append(u)
        fro.append(v)
        fs = []
        for r in tl.static_range(3):
            for k in tl.static_range(3):
                acc = tl.zeros((32, 32), tl.float32) + tl.load(bfs_ptr + r * 3 + k)
                for ch in tl.static_range(NFRO + 2):
                    acc += tl.load(Wf_ptr + (r * 3 + k) * (NFRO + 2) + ch) * fro[ch]
                fs.append(acc)
        rin = []  # rgb input of each round (9 tiles)
        for r in tl.static_range(3):
            for j in tl.static_range(3):
                rin.append(rgb[j])
            s = []
            for k in tl.static_range(3):
                acc = fs[r * 3 + k]
                for j in tl.static_range(3):
                    acc += tl.load(Wr_ptr + r * 9 + k * 3 + j) * rgb[j]
                s.append(tl.where(tok, acc, 0.0))
            Kr = tl.load(K_ptr + r * 1024 + tl.arange(0, 32)[:, None] * 32
                         + tl.arange(0, 32)[None, :])
            s1b = _blur(s[0], Kr)
            s2b = _blur(s[1], Kr)
            phi = tl.load(phi_ptr + r)
            a = tl.math.tanh(s2b * s[2])
            gate = tl.where(tok, tl.sigmoid(s1b * tl.cos(math.pi * a + phi)), 0.0)
            for j in tl.static_range(3):
                pooled = tl.sum(gate * rgb[j]) / N_TOK
                rgb[j] = tl.where(tok, _silu(rgb[j] + pooled), 0.0)
        xin = rgb + fro

        # ---------------- tail forward (pass 1) for gate2
        sp = []
        for k in tl.static_range(3):
            sp.append(tl.zeros((32, 32), tl.float32) + tl.load(bg_ptr + k))
        for j in tl.static_range(H1):
            pre = tl.zeros((32, 32), tl.float32) + tl.load(b1_ptr + j)
            for ch in tl.static_range(NFRO + 5):
                pre += tl.load(W1_ptr + j * (NFRO + 5) + ch) * xin[ch]
            h = tl.where(tok, _silu(pre), 0.0)
            for k in tl.static_range(3):
                sp[k] += tl.load(Vgs_ptr + k * H1 + j) * h
        phig = tl.load(phig_ptr)
        ag = tl.math.tanh(sp[1] * sp[2])
        cosg = tl.cos(math.pi * ag + phig)
        sing = tl.sin(math.pi * ag + phig)
        score2 = sp[0] * cosg
        gate2 = tl.where(tok, tl.sigmoid(score2), 0.0)

        # ---------------- tail backward
        # out_j = mean(gate2 * h_j); d_gate2 = sum_j dout_j * h_j / 400
        d_gate2 = tl.zeros((32, 32), tl.float32)
        optr = dout_ptr + b * sob + gy_out * soh + gx_out * sow
        for j in tl.static_range(H1):
            pre = tl.zeros((32, 32), tl.float32) + tl.load(b1_ptr + j)
            for ch in tl.static_range(NFRO + 5):
                pre += tl.load(W1_ptr + j * (NFRO + 5) + ch) * xin[ch]
            h = tl.where(tok, _silu(pre), 0.0)
            doj = tl.load(optr + j * soc)
            d_gate2 += doj * h / N_TOK
        d_gate2 = tl.where(tok, d_gate2, 0.0)
        d_score2 = d_gate2 * gate2 * (1.0 - gate2)
        d_sp0 = d_score2 * cosg
        d_com = d_score2 * sp[0] * (-sing) * math.pi * (1.0 - ag * ag)
        d_sp1 = d_com * sp[2]
        d_sp2 = d_com * sp[1]
        tl.atomic_add(dphig_ptr + slot, tl.sum(d_score2 * sp[0] * (-sing)))
        tl.atomic_add(dbg_ptr + slot * 3 + 0, tl.sum(d_sp0))
        tl.atomic_add(dbg_ptr + slot * 3 + 1, tl.sum(d_sp1))
        tl.atomic_add(dbg_ptr + slot * 3 + 2, tl.sum(d_sp2))
        d_sp = [d_sp0, d_sp1, d_sp2]

        d_x = [tl.zeros((32, 32), tl.float32) for _ in tl.static_range(NFRO + 5)]
        for j in tl.static_range(H1):
            pre = tl.zeros((32, 32), tl.float32) + tl.load(b1_ptr + j)
            for ch in tl.static_range(NFRO + 5):
                pre += tl.load(W1_ptr + j * (NFRO + 5) + ch) * xin[ch]
            sig = tl.sigmoid(pre)
            h = tl.where(tok, pre * sig, 0.0)
            doj = tl.load(optr + j * soc)
            d_h = doj * gate2 / N_TOK
            for k in tl.static_range(3):
                vg = tl.load(Vgs_ptr + k * H1 + j)
                d_h += vg * d_sp[k]
                tl.atomic_add(dVgs_ptr + slot * 3 * H1 + k * H1 + j,
                              tl.sum(d_sp[k] * h))
            d_h = tl.where(tok, d_h, 0.0)
            d_pre = d_h * sig * (1.0 + pre * (1.0 - sig))  # silu'
            tl.atomic_add(db1_ptr + slot * H1 + j, tl.sum(d_pre))
            for ch in tl.static_range(NFRO + 5):
                tl.atomic_add(dW1_ptr + slot * H1 * (NFRO + 5) + j * (NFRO + 5) + ch,
                              tl.sum(d_pre * xin[ch]))
                d_x[ch] += tl.load(W1_ptr + j * (NFRO + 5) + ch) * d_pre

        # ---------------- rounds backward (reverse order)
        d_rgb = [d_x[0], d_x[1], d_x[2]]
        d_fro = [d_x[3 + ch] for ch in tl.static_range(NFRO + 2)]
        for rr in tl.static_range(3):
            r = 2 - rr
            rgb_in = [rin[r * 3 + j] for j in tl.static_range(3)]
            # recompute this round's forward pieces
            s = []
            for k in tl.static_range(3):
                acc = fs[r * 3 + k]
                for j in tl.static_range(3):
                    acc += tl.load(Wr_ptr + r * 9 + k * 3 + j) * rgb_in[j]
                s.append(tl.where(tok, acc, 0.0))
            Kr = tl.load(K_ptr + r * 1024 + tl.arange(0, 32)[:, None] * 32
                         + tl.arange(0, 32)[None, :])
            s1b = _blur(s[0], Kr)
            s2b = _blur(s[1], Kr)
            phi = tl.load(phi_ptr + r)
            a = tl.math.tanh(s2b * s[2])
            cosv = tl.cos(math.pi * a + phi)
            sinv = tl.sin(math.pi * a + phi)
            gate = tl.where(tok, tl.sigmoid(s1b * cosv), 0.0)

            d_gate = tl.zeros((32, 32), tl.float32)
            d_rgb_in = [tl.zeros((32, 32), tl.float32) for _ in tl.static_range(3)]
            for j in tl.static_range(3):
                pooled = tl.sum(gate * rgb_in[j]) / N_TOK
                z = rgb_in[j] + pooled
                sig = tl.sigmoid(z)
                d_z = tl.where(tok, d_rgb[j] * sig * (1.0 + z * (1.0 - sig)), 0.0)
                d_rgb_in[j] += d_z
                d_pool = tl.sum(d_z)  # broadcast add -> sum
                d_gate += d_pool * rgb_in[j] / N_TOK
                d_rgb_in[j] += d_pool * gate / N_TOK
            d_gate = tl.where(tok, d_gate, 0.0)
            d_score = d_gate * gate * (1.0 - gate)
            d_s1b = d_score * cosv
            d_c = d_score * s1b * (-sinv) * math.pi * (1.0 - a * a)
            d_s2b = d_c * s[2]
            d_s3 = d_c * s2b
            tl.atomic_add(dphi_ptr + slot * 3 + r, tl.sum(d_score * s1b * (-sinv)))
            # blur backward: F = K T K (K symmetric) -> d_T = K d_F K;
            # dK = d_F (T K)^T + (K d_F)^T ... projected straight onto dK/dsigma
            d_s1 = _blur(d_s1b, Kr)
            d_s2 = _blur(d_s2b, Kr)
            dKd = tl.load(dKds_ptr + r * 1024 + tl.arange(0, 32)[:, None] * 32
                          + tl.arange(0, 32)[None, :])
            tk0 = tl.dot(s[0], Kr, allow_tf32=False)
            tk1 = tl.dot(s[1], Kr, allow_tf32=False)
            kd0 = tl.dot(Kr, d_s1b, allow_tf32=False)
            kd1 = tl.dot(Kr, d_s2b, allow_tf32=False)
            dsig = tl.sum(d_s1b * tl.dot(dKd, tk0, allow_tf32=False)) \
                + tl.sum(kd0 * tl.dot(s[0], dKd, allow_tf32=False)) \
                + tl.sum(d_s2b * tl.dot(dKd, tk1, allow_tf32=False)) \
                + tl.sum(kd1 * tl.dot(s[1], dKd, allow_tf32=False))
            tl.atomic_add(dsig_ptr + slot * 3 + r, dsig)
            d_s = [tl.where(tok, d_s1, 0.0), tl.where(tok, d_s2, 0.0),
                   tl.where(tok, d_s3, 0.0)]
            for k in tl.static_range(3):
                tl.atomic_add(dbfs_ptr + slot * 9 + r * 3 + k, tl.sum(d_s[k]))
                for j in tl.static_range(3):
                    tl.atomic_add(dWr_ptr + slot * 27 + r * 9 + k * 3 + j,
                                  tl.sum(d_s[k] * rgb_in[j]))
                    d_rgb_in[j] += tl.load(Wr_ptr + r * 9 + k * 3 + j) * d_s[k]
                for ch in tl.static_range(NFRO + 2):
                    tl.atomic_add(
                        dWf_ptr + slot * 9 * (NFRO + 2) + (r * 3 + k) * (NFRO + 2) + ch,
                        tl.sum(d_s[k] * fro[ch]))
                    d_fro[ch] += tl.load(Wf_ptr + (r * 3 + k) * (NFRO + 2) + ch) * d_s[k]
            d_rgb = d_rgb_in

        # ---------------- d_feat (atomic: phases overlap, clamps collide)
        for j in tl.static_range(3):
            tl.atomic_add(dfeat_ptr + b * sfb + j * sfc + gy * sfh + gx * sfw,
                          tl.where(tok, d_rgb[j], 0.0), mask=tok)
        for ch in tl.static_range(NFRO):
            tl.atomic_add(dfeat_ptr + b * sfb + (ch + 3) * sfc + gy * sfh + gx * sfw,
                          tl.where(tok, d_fro[ch], 0.0), mask=tok)
        # d_fro[NFRO], d_fro[NFRO+1] are the uv constants: no gradient


# ================================================== pure-torch param mirror
def pool_from_params(x, p):
    """Pure function mirror of TileEncoder.pool_features_dense in terms of the
    FOLDED parameter tensors — used by the chunked autograd backward and by
    the parity check. x: (N, 17, H, W) phase-padded tiles (H, W multiples of
    20). p: dict of derived tensors (see _derive_params)."""
    rgb, fro = x[:, :3], x[:, 3:]
    grid = WIN
    fs = F.conv2d(fro, p["Wf"].reshape(9, -1, 1, 1), p["bfs"].reshape(9))

    def blur(s, K):  # (N, C, H, W) per-tile via banded matrix
        t = (s.reshape(-1, grid) @ K).reshape(s.shape)
        t = t.transpose(-1, -2)
        t = (t.reshape(-1, grid) @ K).reshape(t.shape)
        return t.transpose(-1, -2)

    for r in range(3):
        s = F.conv2d(rgb, p["Wr"][r].reshape(3, 3, 1, 1)) + fs[:, 3 * r: 3 * r + 3]
        s12 = blur(s[:, :2], p["K"][r])
        arg = torch.tanh(s12[:, 1:2] * s[:, 2:3])
        gate = torch.sigmoid(s12[:, 0:1] * torch.cos(math.pi * arg + p["phi"][r]))
        pooled = F.avg_pool2d(gate * rgb, grid)
        rgb = F.silu(rgb + F.interpolate(pooled, scale_factor=grid, mode="nearest"))
    x = torch.cat([rgb, fro], 1)
    h = F.silu(F.conv2d(x, p["W1"].reshape(-1, x.shape[1], 1, 1), p["b1"]))
    sg = F.conv2d(h, p["Vgs"].reshape(3, -1, 1, 1), p["bg"])
    arg = torch.tanh(sg[:, 1:2] * sg[:, 2:3])
    gate2 = torch.sigmoid(sg[:, 0:1] * torch.cos(math.pi * arg + p["phig"]))
    return F.avg_pool2d(gate2 * h, grid)


PARAM_KEYS = ("Wr", "Wf", "bfs", "K", "phi", "W1", "b1", "Vgs", "bg", "phig")


def _derive_params(encoder):
    """Small differentiable tensors the kernels/mirror consume. Autograd
    chains their grads back to the raw modules (V, phi, sigma, norm affines,
    fc1, gate)."""
    Wr, Wf, bfs, K, phi = [], [], [], [], []
    for r in encoder.rounds:
        wr, wf, b = r.score_convs()
        Wr.append(wr)
        Wf.append(wf)
        bfs.append(b)
        K.append(r.blur_matrix(wr.device, torch.float32))
        phi.append(r.phi.reshape(1))
    scale, shift = encoder.pool_norm.fold()
    p = {
        "Wr": torch.stack(Wr), "Wf": torch.stack(Wf), "bfs": torch.stack(bfs),
        "K": torch.stack(K), "phi": torch.cat(phi),
        "W1": encoder.fc1.weight, "b1": encoder.fc1.bias,
        "Vgs": encoder.gate.V * scale, "bg": encoder.gate.V @ shift,
        "phig": encoder.gate.phi.reshape(1),
    }
    return p


def _dK_dsigma(encoder, device):
    """(3, 20, 20) jacobian of each round's blur matrix wrt raw_sigma (forward
    mode: sigma is a scalar). Built ON DEVICE — a .cpu() here would be a
    blocking host sync on the hot path of a kernel whose whole point is
    removing per-step overhead."""
    outs = []
    for r in encoder.rounds:
        def f(sig):
            rr = torch.arange(-4, 5, dtype=torch.float32, device=device)
            g = torch.exp(-(rr * rr) / (2 * (F.softplus(sig) + 0.5) ** 2))
            g = g / g.sum()
            eye = torch.eye(WIN, dtype=torch.float32, device=device)
            return F.conv1d(eye.unsqueeze(1), g.reshape(1, 1, 9),
                            padding=4).squeeze(1)
        _, jvp = torch.func.jvp(
            f, (r.raw_sigma.detach().to(device),),
            (torch.ones((), dtype=torch.float32, device=device),))
        outs.append(jvp)
    return torch.stack(outs)


def _pad32(k):  # (..., 20, 20) -> (..., 32, 32) zero-padded, contiguous fp32
    return F.pad(k.float(), (0, 12, 0, 12)).contiguous()


def _slot_zeros(device, *shape):
    return torch.zeros((SLOTS,) + shape, device=device, dtype=torch.float32)


class _FusedStage1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, Wr, Wf, bfs, K, phi, W1, b1, Vgs, bg, phig,
                raw_sig, dKds, collect, stats_out, bwd_mode, chunk_rows):
        B = feat.shape[0]
        dev = feat.device
        out = torch.empty(B, W1.shape[0], NH, NW, device=dev, dtype=torch.float32)
        nfro = feat.shape[1] - 3
        n_prog = B * 4 * PH * PW
        bufs = None
        if collect:
            bufs = dict(
                rsum=_slot_zeros(dev, 3, 3), rsq=_slot_zeros(dev, 3, 3),
                fsum=_slot_zeros(dev, nfro + 2), fsq=_slot_zeros(dev, nfro + 2),
                psum=_slot_zeros(dev, W1.shape[0]), psq=_slot_zeros(dev, W1.shape[0]),
            )
        z = torch.zeros(1, device=dev)
        _fwd_kernel[(n_prog,)](
            feat, out,
            Wr.detach().contiguous(), Wf.detach().contiguous(),
            bfs.detach().contiguous(), _pad32(K.detach()),
            phi.detach().contiguous(),
            W1.detach().contiguous(), b1.detach().contiguous(),
            Vgs.detach().contiguous(), bg.detach().contiguous(),
            phig.detach().contiguous(),
            *( (bufs["rsum"], bufs["rsq"], bufs["fsum"], bufs["fsq"],
                bufs["psum"], bufs["psq"]) if collect else (z, z, z, z, z, z)),
            B,
            *feat.stride(), *out.stride(),
            COLLECT=collect, NFRO=nfro, H1=W1.shape[0],
            num_warps=4,
        )
        if collect:
            n = float(n_prog * N_TOK)
            for key, (s_k, q_k) in (("rounds", ("rsum", "rsq")),
                                    ("frozen", ("fsum", "fsq")),
                                    ("pool", ("psum", "psq"))):
                stats_out[key] = (bufs[s_k].sum(0), bufs[q_k].sum(0), n)
        ctx.save_for_backward(feat, Wr, Wf, bfs, K, phi, W1, b1, Vgs, bg,
                              phig, dKds)
        ctx.bwd_mode = bwd_mode
        ctx.chunk_rows = chunk_rows
        return out

    @staticmethod
    def backward(ctx, d_out):
        feat, Wr, Wf, bfs, K, phi, W1, b1, Vgs, bg, phig, dKds = ctx.saved_tensors
        if ctx.bwd_mode == "triton":
            grads = _backward_triton(feat, d_out, Wr, Wf, bfs, K, phi,
                                     W1, b1, Vgs, bg, phig, dKds)
        else:
            grads = _backward_chunked(feat, d_out, Wr, Wf, bfs, K, phi,
                                      W1, b1, Vgs, bg, phig, ctx.chunk_rows)
        d_feat, dWr, dWf, dbfs, dK, dphi, dW1, db1, dVgs, dbg, dphig, dsig = grads
        return (d_feat, dWr, dWf, dbfs, dK, dphi, dW1, db1, dVgs, dbg, dphig,
                dsig, None, None, None, None, None)


def _backward_triton(feat, d_out, Wr, Wf, bfs, K, phi, W1, b1, Vgs, bg,
                     phig, dKds):
    B = feat.shape[0]
    dev = feat.device
    nfro = feat.shape[1] - 3
    h1 = W1.shape[0]
    d_feat = torch.zeros(feat.shape, device=dev, dtype=torch.float32)
    d_out = d_out.contiguous().float()
    bufs = dict(
        dWr=_slot_zeros(dev, 27), dWf=_slot_zeros(dev, 9 * (nfro + 2)),
        dbfs=_slot_zeros(dev, 9), dsig=_slot_zeros(dev, 3),
        dphi=_slot_zeros(dev, 3), dW1=_slot_zeros(dev, h1 * (nfro + 5)),
        db1=_slot_zeros(dev, h1), dVgs=_slot_zeros(dev, 3 * h1),
        dbg=_slot_zeros(dev, 3), dphig=_slot_zeros(dev),
    )
    n_prog = B * 4 * PH * PW
    _bwd_kernel[(n_prog,)](
        feat, d_out, d_feat,
        Wr.detach().contiguous(), Wf.detach().contiguous(),
        bfs.detach().contiguous(), _pad32(K.detach()), _pad32(dKds),
        phi.detach().contiguous(),
        W1.detach().contiguous(), b1.detach().contiguous(),
        Vgs.detach().contiguous(), bg.detach().contiguous(),
        phig.detach().contiguous(),
        bufs["dWr"], bufs["dWf"], bufs["dbfs"], bufs["dsig"], bufs["dphi"],
        bufs["dW1"], bufs["db1"], bufs["dVgs"], bufs["dbg"], bufs["dphig"],
        B,
        *feat.stride(), *d_out.stride(),
        NFRO=nfro, H1=h1,
        num_warps=4,
    )
    return (
        d_feat.to(feat.dtype),
        bufs["dWr"].sum(0).reshape(3, 3, 3),
        bufs["dWf"].sum(0).reshape(3, 3, nfro + 2),
        bufs["dbfs"].sum(0).reshape(3, 3),
        None,  # dK: sigma grads are projected in-kernel (dsig)
        bufs["dphi"].sum(0),
        bufs["dW1"].sum(0).reshape(h1, nfro + 5),
        bufs["db1"].sum(0),
        bufs["dVgs"].sum(0).reshape(3, h1),
        bufs["dbg"].sum(0),
        bufs["dphig"].sum().reshape(1),
        bufs["dsig"].sum(0),
    )


def _backward_chunked(feat, d_out, Wr, Wf, bfs, K, phi, W1, b1, Vgs, bg,
                      phig, chunk_rows, chunk_batch=16):
    """Autograd through the pure-torch mirror, one (batch-slab x phase x
    height-band) at a time. Slower than the triton backward but the gradients
    are autograd's. Memory is bounded by one slab x band regardless of B
    (measured ~1.2 KB of saved-for-backward per batch*pixel: 16 x 180 x 960
    fp32 ~ 3.4 GB transient + the fp32 feat/d_feat copies)."""
    B = feat.shape[0]
    leaves = {
        "Wr": Wr, "Wf": Wf, "bfs": bfs, "K": K, "phi": phi,
        "W1": W1, "b1": b1, "Vgs": Vgs, "bg": bg, "phig": phig,
    }
    leaf = {k: v.detach().float().requires_grad_(True) for k, v in leaves.items()}
    feat_leaf = feat.detach().float().requires_grad_(True)
    d_out = d_out.float()
    phases = ((0, 0), (0, 10), (10, 0), (10, 10))
    uv0 = _uv_tile(feat.device)
    with torch.enable_grad():
        for b0 in range(0, B, chunk_batch):
            b1_ = min(b0 + chunk_batch, B)
            nb = b1_ - b0
            for oy, ox in phases:
                hp = ((IMG_H - oy) // WIN) * WIN
                wp = ((IMG_W - ox) // WIN) * WIN
                # d slice for this phase, zero-padded to the full tile grid
                d_ph = torch.zeros(nb, W1.shape[0], PH, PW, device=feat.device)
                dsl = d_out[b0: b1_, :, oy // 10:: 2, ox // 10:: 2]
                d_ph[:, :, : dsl.shape[2], : dsl.shape[3]] = dsl
                for r0 in range(0, IMG_H, chunk_rows):
                    if r0 >= hp:
                        # band lies entirely in the replicate-padding zone:
                        # its tiles are cropped from the output (zero grad)
                        # AND an empty source slice would crash F.pad
                        continue
                    r1 = min(r0 + chunk_rows, IMG_H)
                    # build each band's crop straight from the leaf so every
                    # band owns its whole graph (a shared full-frame pad node
                    # would be freed by the first band's backward). Bottom pad
                    # rows exist only in the band containing hp; they
                    # replicate row hp-1, exactly like padding the full crop.
                    src_r1 = min(oy + r1, oy + hp)
                    band = feat_leaf[b0: b1_, :, oy + r0: src_r1, ox: ox + wp]
                    band = F.pad(band,
                                 (0, IMG_W - wp, 0, r1 - (src_r1 - oy)),
                                 mode="replicate")
                    x = torch.cat(
                        [band, uv0[:, :, r0: r1].expand(nb, -1, -1, -1)], 1)
                    pooled = pool_from_params(x, {k: leaf[k] for k in PARAM_KEYS})
                    pooled.backward(d_ph[:, :, r0 // WIN: r1 // WIN])
    grads = {k: v.grad for k, v in leaf.items()}
    return (
        feat_leaf.grad.to(feat.dtype),
        grads["Wr"], grads["Wf"], grads["bfs"], grads["K"], grads["phi"],
        grads["W1"], grads["b1"], grads["Vgs"], grads["bg"],
        grads["phig"].reshape(1),
        None,  # dsig: flows through dK in this mode
    )


_UV_CACHE = {}


def _uv_tile(device):
    if device not in _UV_CACHE:
        r = torch.arange(WIN, dtype=torch.float32, device=device)
        v, u = torch.meshgrid(r, r, indexing="ij")
        uv = torch.stack([(u + 0.5) / WIN, (v + 0.5) / WIN])
        _UV_CACHE[device] = uv.repeat(1, IMG_H // WIN, IMG_W // WIN).unsqueeze(0)
    return _UV_CACHE[device]


class FusedStage1:
    """Installable model.fused_pool: (B, feat, 540, 960) -> (B, h1, 53, 95).

    Derives the folded parameter tensors each call (tiny, differentiable),
    runs the fused kernels, and feeds the accumulated batch stats back into
    the DeployNorm buffers."""

    def __init__(self, model, bwd_mode="triton", chunk_rows=180):
        self.model = model
        self.encoder = model.encoder
        self.bwd_mode = bwd_mode
        self.chunk_rows = chunk_rows

    def __call__(self, feat):
        enc = self.encoder
        # derive the folded params with autocast DISABLED: under bf16 autocast
        # the matmuls in score_convs()/gate folding (V @ shift) silently
        # downcast bfs/bg to bf16 while every elementwise-derived tensor stays
        # fp32 — a mixed-precision parameter set the (autocast-free) parity
        # checks would never see. All derived tensors must be fp32.
        with torch.autocast(feat.device.type, enabled=False):
            p = _derive_params(enc)
            need_dsig = enc.training and self.bwd_mode == "triton" \
                and torch.is_grad_enabled()
            dKds = _dK_dsigma(enc, feat.device) if need_dsig else \
                torch.zeros(3, WIN, WIN, device=feat.device)
        collect = enc.training
        stats = {}
        raw_sig = torch.stack([r.raw_sigma for r in enc.rounds])
        # sigma grads: triton projects dK onto dK/dsigma in-kernel and returns
        # them through the raw_sig input; chunked returns dK and autograd
        # chains through blur_matrix inside _derive_params (K is derived).
        pooled = _FusedStage1Fn.apply(
            feat.contiguous(), p["Wr"], p["Wf"], p["bfs"], p["K"], p["phi"],
            p["W1"], p["b1"], p["Vgs"], p["bg"], p["phig"],
            raw_sig, dKds, collect, stats, self.bwd_mode, self.chunk_rows)
        if collect and stats:
            with torch.no_grad():
                # every program contributes 400 tokens per channel, so the
                # per-channel count is identical for all three stat groups
                def mv(s, q, n):
                    mean = s / n
                    return mean, (q / n - mean * mean).clamp_min(0.0)

                rs, rq, n = stats["rounds"]           # (3, 3) [round, rgb ch]
                fmean, fvar = mv(*stats["frozen"])    # (nfro + 2,)
                for i, r in enumerate(enc.rounds):
                    rmean, rvar = mv(rs[i], rq[i], n)
                    m = torch.cat([rmean, fmean])
                    v = torch.cat([rvar, fvar])
                    r.norm._update(m, v)
                pmean, pvar = mv(*stats["pool"])
                enc.pool_norm._update(pmean, pvar)
        return pooled


# ------------------------------------------------------------------- parity
@torch.no_grad()
def parity_forward(model, img, atol=2e-3):
    """Compare the fused forward against the PyTorch dense path on one batch
    (eval mode, fp32 — no stat updates, no autocast). Returns (ok, max_delta)."""
    was_training = model.training
    model.eval()
    feat = model._features(img.float())
    fused = FusedStage1(model)(feat)
    x = model._phase_batch(feat, "constant")
    ref = model._interleave(model.encoder.pool_features_dense(x), feat.shape[0])
    if was_training:
        model.train()
    delta = (fused - ref).abs().max().item()
    return delta <= atol, delta


def parity_backward(model, img, rtol=5e-2):
    """Compare triton-backward grads against chunked-autograd grads on one
    batch. Returns (ok, worst_rel)."""
    was_training = model.training
    model.eval()  # frozen stats; grads still flow to params
    feat_base = model._features(img.float()).detach()
    worst = 0.0
    grads = {}
    for mode in ("triton", "chunked"):
        model.zero_grad(set_to_none=True)
        feat = feat_base.clone().requires_grad_(True)
        pooled = FusedStage1(model, bwd_mode=mode)(feat)
        pooled.square().mean().backward()
        grads[mode] = {
            "feat": feat.grad.detach().clone(),
            **{n: (p.grad.detach().clone() if p.grad is not None else None)
               for n, p in model.encoder.named_parameters()},
        }
    for k, g_t in grads["triton"].items():
        g_c = grads["chunked"][k]
        if g_t is None or g_c is None:
            if (g_t is None) != (g_c is None):
                worst = float("inf")
            continue
        denom = g_c.abs().max().clamp_min(1e-6)
        rel = ((g_t - g_c).abs().max() / denom).item()
        worst = max(worst, rel)
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return worst <= rtol, worst
