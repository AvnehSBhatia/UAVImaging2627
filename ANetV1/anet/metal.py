"""Fused Metal kernel for the ANetV1 encoder (eval only) — the fast local path.

Why: the encoder is memory-bandwidth-bound in any graph runtime (ONNX/CoreML
floor ~21 ms/img at hidden=24): the 3 mixing rounds and the per-token MLP make
repeated full-resolution passes over ~100-200 MB activation maps. But every
window is exactly one 20x20 threadgroup, so the WHOLE encoder — 3 rounds
(score dots, in-tile separable blur, cosine gate, gated tile mean, SiLU
residual) plus the 11->H->H MLP and the final cosine-gated pool — runs in
threadgroup memory and registers, reading the 9-channel stem map once and
writing the (hidden, 53, 95) embedding grid directly (phase-interleaved, no
pixel_shuffle). DRAM traffic per frame drops from ~2.5 GB to ~90 MB.

Same math as ANetV1 eval (D34 folds: eval-BN affines baked into the score
weights); parity is fp-noise only. Training is untouched — this wraps a
trained checkpoint for inference.

    from anet.metal import MetalANet
    model = MetalANet.from_checkpoint("runs/anet/best.pt")
    cells = model(images)   # (B,3,540,960) [0,1] on cpu/mps -> (B,3,54,96)
"""

import torch
import torch.nn.functional as F

from .model.anet import ANetV1

KERNEL_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

constant uint HID      = {hid};
constant uint R_STRIDE = 46;              // per round: W 3x11 + b 3 + phi + g 9
constant uint MLP_OFF  = 138;             // 3 rounds x 46
constant uint W1_OFF   = MLP_OFF;                     // HID*11
constant uint B1_OFF   = W1_OFF + HID * 11;           // HID
constant uint W2_OFF   = B1_OFF + HID;                // HID*HID
constant uint B2_OFF   = W2_OFF + HID * HID;          // HID
constant uint VG_OFF   = B2_OFF + HID;                // 3*HID
constant uint BG_OFF   = VG_OFF + 3 * HID;            // 3
constant uint PHIG_OFF = BG_OFF + 3;

static float silu(float x) {{ return x / (1.0f + exp(-x)); }}

// 4-lane sum over the 400-thread tile; safe back-to-back (leading barrier)
static float4 tg_sum4(float4 v, threadgroup float4* red,
                      uint simd_id, uint lane, uint tid) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 p = simd_sum(v);
    if (lane == 0) red[simd_id] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float4 total = 0.0f;
        for (uint k = 0; k < 13; ++k) total += red[k];
        red[15] = total;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[15];
}}

kernel void anet_encoder(
    device const float* feat [[buffer(0)]],   // (9, 540, 960) stem map
    constant float* C    [[buffer(1)]],   // folded consts (see host)
    device float*       out  [[buffer(2)]],   // (HID, 53, 95) embedding grid
    uint3 tg [[threadgroup_position_in_grid]],      // (j, i, img*4 + phase)
    uint3 tp [[thread_position_in_threadgroup]],    // (dx, dy, 0)
    uint  simd_id [[simdgroup_index_in_threadgroup]],
    uint  lane    [[thread_index_in_simdgroup]])
{{
    const uint dx = tp.x, dy = tp.y;
    const uint tid = dy * 20 + dx;
    const uint img = tg.z >> 2, phase = tg.z & 3;
    const uint ox = 10 * (phase & 1), oy = 10 * (phase >> 1);
    if (ox + 20 * tg.x + 20 > 960 || oy + 20 * tg.y + 20 > 540) return;
    const uint px = ox + 20 * tg.x + dx, py = oy + 20 * tg.y + dy;

    // token: [r, g, b, e0..e5, u, v]
    device const float* fimg = feat + img * 9 * 540 * 960;
    float tok[11];
    for (uint c = 0; c < 9; ++c) tok[c] = fimg[(c * 540 + py) * 960 + px];
    tok[9]  = (float(dx) + 0.5f) / 20.0f;
    tok[10] = (float(dy) + 0.5f) / 20.0f;

    threadgroup float shA[400];
    threadgroup float shB[400];
    threadgroup float4 red[16];

    // ---- 3 mixing rounds, entirely in-tile ----
    for (uint r = 0; r < 3; ++r) {{
        constant float* W  = C + r * R_STRIDE;   // (3, 11) folded V*BN
        constant float* bb = W + 33;             // (3,)   V @ BN-shift
        const float phi = bb[3];
        constant float* g = bb + 4;              // 9 gaussian taps

        float s0 = bb[0], s1 = bb[1], s2 = bb[2];
        for (uint c = 0; c < 11; ++c) {{
            const float t = tok[c];
            s0 += W[c] * t; s1 += W[11 + c] * t; s2 += W[22 + c] * t;
        }}
        // separable blur of s0, s1 within the tile (zero pad at tile border)
        threadgroup_barrier(mem_flags::mem_threadgroup);  // shA/shB reuse
        shA[tid] = s0; shB[tid] = s1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float r0 = 0.0f, r1 = 0.0f;
        for (int t = -4; t <= 4; ++t) {{
            const int xx = int(dx) + t;
            if (xx >= 0 && xx < 20) {{
                const float w = g[t + 4];
                r0 += w * shA[dy * 20 + xx];
                r1 += w * shB[dy * 20 + xx];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shA[tid] = r0; shB[tid] = r1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float c0 = 0.0f, c1 = 0.0f;
        for (int t = -4; t <= 4; ++t) {{
            const int yy = int(dy) + t;
            if (yy >= 0 && yy < 20) {{
                const float w = g[t + 4];
                c0 += w * shA[yy * 20 + dx];
                c1 += w * shB[yy * 20 + dx];
            }}
        }}
        const float score = c0 * cos(M_PI_F * tanh(c1 * s2) + phi);
        const float gate = 1.0f / (1.0f + exp(-score));
        // gated tile mean added to RGB, SiLU (frozen channels untouched)
        const float4 pooled = tg_sum4(
            float4(gate * tok[0], gate * tok[1], gate * tok[2], 0.0f),
            red, simd_id, lane, tid) / 400.0f;
        for (uint c = 0; c < 3; ++c) tok[c] = silu(tok[c] + pooled[c]);
    }}

    // ---- per-token MLP 11 -> HID -> HID (SiLU) ----
    float h1[HID];
    #pragma unroll
    for (uint o = 0; o < HID; ++o) {{
        float acc = C[B1_OFF + o];
        constant float* w = C + W1_OFF + o * 11;
        for (uint c = 0; c < 11; ++c) acc += w[c] * tok[c];
        h1[o] = silu(acc);
    }}
    float h2[HID];
    #pragma unroll
    for (uint o = 0; o < HID; ++o) {{
        float acc = C[B2_OFF + o];
        constant float* w = C + W2_OFF + o * HID;
        #pragma unroll
        for (uint c = 0; c < HID; ++c) acc += w[c] * h1[c];
        h2[o] = silu(acc);
    }}

    // ---- cosine-gated pool (BN folded into the gate dots) ----
    float gs[3];
    for (uint k = 0; k < 3; ++k) {{
        float acc = C[BG_OFF + k];
        constant float* w = C + VG_OFF + k * HID;
        for (uint c = 0; c < HID; ++c) acc += w[c] * h2[c];
        gs[k] = acc;
    }}
    const float gscore = gs[0] * cos(M_PI_F * tanh(gs[1] * gs[2]) + C[PHIG_OFF]);
    const float gate2 = 1.0f / (1.0f + exp(-gscore));

    const uint row = 2 * tg.y + oy / 10;
    const uint col = 2 * tg.x + ox / 10;
    device float* oimg = out + img * HID * 53 * 95;
    for (uint c = 0; c < HID; c += 4) {{  // HID % 4 == 0
        const float4 e = tg_sum4(
            float4(gate2 * h2[c], gate2 * h2[c + 1],
                   gate2 * h2[c + 2], gate2 * h2[c + 3]),
            red, simd_id, lane, tid) / 400.0f;
        if (tid == 0)
            for (uint k = 0; k < 4; ++k)
                oimg[((c + k) * 53 + row) * 95 + col] = e[k];
    }}
}}
"""


def _fuse_stem(stem):
    """EdgeDQStem as ONE 7x7 conv over [img, ones]: W (9,4,7,7), b (9,).

    Each edge branch is dq_out ∘ depthwise7x7 ∘ dq_in — a linear chain, so it
    composes into a dense 7x7 kernel. The dq_in translation t1 interacts with
    the branch's zero padding (the ORIGINAL pads the dq-transformed map, so at
    borders t1 only contributes over valid taps): a constant ones channel,
    zero-padded like the image, with taps D[o]*t1[o] reproduces that
    valid-tap sum exactly. Raw branch = dq_out[0] as a center-tap 1x1."""
    dev = next(stem.parameters()).device
    W = torch.zeros(9, 4, 7, 7, device=dev)
    b = torch.zeros(9, device=dev)
    R0, t0 = stem.dq_out[0].matrix()
    W[0:3, 0:3, 3, 3] = R0
    b[0:3] = t0
    branches = ((stem.dq_v, stem.edge_v, stem.dq_out[1]),
                (stem.dq_h, stem.edge_h, stem.dq_out[2]))
    for k, (dq_in, edge, dq_o) in enumerate(branches):
        R1, t1 = dq_in.matrix()
        R2, t2 = dq_o.matrix()
        D = edge.weight[:, 0]  # (3, 7, 7) depthwise taps
        sl = slice(3 + 3 * k, 6 + 3 * k)
        W[sl, 0:3] = torch.einsum("po,oyx,oi->piyx", R2, D, R1)
        W[sl, 3] = torch.einsum("po,oyx,o->pyx", R2, D, t1)
        b[sl] = t2
    return W, b


def _build_consts(model):
    """Fold eval-BN affines into the score weights (same algebra as D34) and
    pack everything the kernel reads into one flat fp32 buffer."""
    enc = model.encoder
    assert enc.in_dim == 11, "kernel is specialized for 11-d tokens (edge_dq stem)"
    dev = next(enc.parameters()).device
    dt = torch.float32
    parts = []
    for r in enc.rounds:
        scale, shift = r._fold()
        parts += [(r.V * scale).reshape(-1), r.V @ shift, r.phi.reshape(1),
                  r._kernel1d(dev, dt)]
    fc1, fc2 = enc.mlp[0], enc.mlp[2]
    scale, shift = (
        enc.bn.weight * torch.rsqrt(enc.bn.running_var + enc.bn.eps),
        None,
    )
    shift = enc.bn.bias - enc.bn.running_mean * scale
    parts += [fc1.weight.reshape(-1), fc1.bias, fc2.weight.reshape(-1), fc2.bias,
              (enc.gate.V * scale).reshape(-1), enc.gate.V @ shift,
              enc.gate.phi.reshape(1)]
    return torch.cat([p.detach().to(dev, dt).reshape(-1) for p in parts]).to("mps")


class MetalANet:
    """Callable inference wrapper: stem + tail run as (small) eager MPS ops,
    the entire encoder runs as one fused Metal kernel dispatch."""

    def __init__(self, model):
        assert torch.backends.mps.is_available(), "MetalANet needs an MPS device"
        self.model = model.eval().to("mps")
        self.hidden = model.encoder.hidden
        with torch.no_grad():
            self.consts = _build_consts(self.model)
            self.stem_w, self.stem_b = _fuse_stem(self.model.stem_mod)
        self.ones = torch.ones(1, 1, 540, 960, device="mps")
        self.lib = torch.mps.compile_shader(KERNEL_TEMPLATE.format(hid=self.hidden))

    @classmethod
    def from_checkpoint(cls, ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        return cls(ANetV1.from_state_dict(sd, use_checkpoint=False))

    @torch.no_grad()
    def __call__(self, images):  # (3,540,960) or (B,3,540,960) in [0,1]
        x = images.to("mps", torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        b = x.shape[0]
        ones = self.ones.expand(b, -1, -1, -1)
        feat = F.conv2d(torch.cat([x, ones], 1),
                        self.stem_w, self.stem_b, padding=3).contiguous()
        m16 = torch.zeros(b, self.hidden, 53, 95, device="mps")
        self.lib.anet_encoder(feat, self.consts, m16,
                              threads=(960, 540, 4 * b), group_size=(20, 20, 1))
        return self.model._tail(m16)
