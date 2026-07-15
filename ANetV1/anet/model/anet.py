import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .blocks import DualQuaternionRGB, EdgeDQStem, EdgeDQStem4
from .context import SlimContext
from .encoder import WindowEncoder
from .globalmix import GlobalCosineMix
from .head import CenterHead, RegionHead, RegionHeadV9, prior_bias_
from .neck import ConvNeck
from .pyramid import GatedGlobalPool, ScalarKernelPool
from .tile_encoder import TileEncoder


class ANetV1(nn.Module):
    """ANetV1 v6 — see ANetV1/ARCHITECTURE.md for the full spec and decision log.

    Input:  (B, 3, 540, 960) in [0,1]
    Output: (B, 3, 54, 96) cell logits — 10x10-px cells, overlap-averaged (D21)

    Stage 1 runs in one of two mathematically equivalent modes:
      dense=True  (default) — 4-phase dense formulation (D25): plain convs/pools
                  on image tensors; ~10x faster on GPU and matches the Hailo graph
      dense=False — unfold-based per-window tokens; reference implementation
    """

    IMG_H, IMG_W = 540, 960
    WIN, STRIDE = 20, 10
    PHASES = ((0, 0), (0, 10), (10, 0), (10, 10))

    def __init__(self, use_checkpoint=True, dense=True, hidden=16, stem="highpass",
                 path_a_per_channel=True, prior_fg=None, arch="v8", h1=48,
                 neck_rounds=2, head_width=24, aux_head=False, head_proto=True):
        super().__init__()
        self.prior_fg = prior_fg
        if arch == "v12":
            # v12 stage 1 is single-phase, non-overlapping stride-20 windows
            # (no 4x phase overlap): one cell per 20x20 tile, exactly matching
            # the center-heatmap grid rasterize.py builds (V12_H=27, V12_W=48).
            self.nh = self.IMG_H // self.WIN  # 27
            self.nw = self.IMG_W // self.WIN  # 48
        else:
            self.nh = (self.IMG_H - self.WIN) // self.STRIDE + 1  # 53
            self.nw = (self.IMG_W - self.WIN) // self.STRIDE + 1  # 95
        self.n_win = self.nh * self.nw  # 5035 (v8/v9) / 1296 (v12)
        self.use_checkpoint = use_checkpoint
        self.dense = dense
        self.arch = arch
        self.stem = stem
        self.path_a_per_channel = path_a_per_channel
        # trainer-installed fused Stage-1 (Triton, D40): callable
        # (B,feat,540,960) -> (B,h1,53,95) pooled window features, replacing
        # encoder.pool_features_dense + phase plumbing. None = PyTorch path.
        self.fused_pool = None

        # hidden=16 is the 17k spec model; hidden=24 is the pre-registered
        # capacity mitigation (ARCHITECTURE §8 risk 2, ~24k params with the
        # Path-B ripple). d = embedding + global (x,y) coords, 18 at spec.
        d = hidden + 2
        # Pixel-token feature stem: gives Stage 1 the edge/texture evidence colour
        # alone can't provide (probe: mannequin signal is in the pixels, not the
        # embeddings). Runs on the full frame BEFORE windowing so windowed/dense
        # paths stay bit-identical and the export graph gains only plain convs.
        #   highpass (D32): quat-RGB + isotropic 3x3 high-pass -> 6 feat channels
        #   edge_dq  (D33): oriented dual-quaternion edge front-end -> 9 feat channels
        # Both keep the 3 colour channels first, so the encoder's colour-only
        # residual update (MixRound `:3`) and frozen-evidence channels are unchanged.
        if stem == "edge_dq4":
            self.stem_mod = EdgeDQStem4()
            self.feat = EdgeDQStem4.out_channels  # 15
        elif stem == "edge_dq":
            self.stem_mod = EdgeDQStem()
            self.feat = EdgeDQStem.out_channels  # 9
        elif stem == "highpass":
            self.quat = DualQuaternionRGB()
            self.grad = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
            with torch.no_grad():
                self.grad.weight.fill_(-1.0 / 9.0)
                self.grad.weight[:, :, 1, 1] += 1.0
            self.feat = 6
        else:
            raise ValueError(f"unknown stem {stem!r} (highpass|edge_dq|edge_dq4)")
        self.in_dim = self.feat + 2  # + (u,v) window-relative coords

        if arch == "v9":
            self.encoder = TileEncoder(hidden=hidden, h1=h1, in_dim=self.in_dim)
            self.neck = ConvNeck(d, rounds=neck_rounds)
            self.pools = nn.ModuleList(
                [ScalarKernelPool(d, k, per_channel=path_a_per_channel)
                 for k in (3, 7, 11)]
            )
            self.path_dq = nn.ModuleList([nn.Conv2d(d, d, 1) for _ in range(3)])
            with torch.no_grad():
                for conv in self.path_dq:
                    conv.weight.copy_(torch.eye(d).reshape(d, d, 1, 1))
                    conv.bias.zero_()
            self.context = SlimContext(d)
            self.head = RegionHeadV9(dim=d, width=head_width, prior_fg=prior_fg,
                                     proto=head_proto)
            # train-only deep supervision (D46): a linear probe on the raw
            # embedding map gives the encoder a direct gradient path that
            # cannot be blocked by a collapsed head. Dropped at export/eval.
            self.aux = nn.Conv2d(d, 3, 1) if aux_head else None
            if aux_head and prior_fg:
                prior_bias_(self.aux, prior_fg)
        elif arch == "v8":
            self.encoder = WindowEncoder(hidden, in_dim=self.in_dim)
            self.pools = nn.ModuleList(
                [ScalarKernelPool(d, k, per_channel=path_a_per_channel) for k in (3, 7, 11)]
            )
            # learned per-path transform after Path A k3/k7/k11 (D36): one generalized
            # DQ per scale. A quaternion rotates a 3-vector, so on the d-dim map it's a
            # learned 1x1 conv (identity-init -> starts as a no-op, bakes to a constant
            # 1x1 conv at export). Lets each scale recombine its channels before both
            # Path B and the head consume it.
            self.path_dq = nn.ModuleList([nn.Conv2d(d, d, 1) for _ in range(3)])
            with torch.no_grad():
                for conv in self.path_dq:
                    conv.weight.copy_(torch.eye(d).reshape(d, d, 1, 1))
                    conv.bias.zero_()
            self.globals_ = nn.ModuleList([GatedGlobalPool(dim=d) for _ in range(3)])  # unshared (D28)
            self.mix = GlobalCosineMix(pad_to=d)
            self.head = RegionHead(dim=d, prior_fg=prior_fg)
        elif arch == "v12":
            # object center-heatmap detector (see workflow spec / planned
            # ARCHITECTURE.md delta): same Stage-1 encoder/neck/Path-A/context
            # as v9, but single-phase stride-20 (no 4x overlap, D-something
            # TBD) and a CenterHead readout instead of RegionHeadV9's per-cell
            # classifier — no aux probe, no metric-prototype path.
            self.encoder = TileEncoder(hidden=hidden, h1=h1, in_dim=self.in_dim)
            self.neck = ConvNeck(d, rounds=neck_rounds)
            self.pools = nn.ModuleList(
                [ScalarKernelPool(d, k, per_channel=path_a_per_channel)
                 for k in (3, 7, 11)]
            )
            self.path_dq = nn.ModuleList([nn.Conv2d(d, d, 1) for _ in range(3)])
            with torch.no_grad():
                for conv in self.path_dq:
                    conv.weight.copy_(torch.eye(d).reshape(d, d, 1, 1))
                    conv.bias.zero_()
            self.context = SlimContext(d)
            self.head = CenterHead(dim=d, width=head_width, prior_fg=prior_fg)
            # train-only DEEP SUPERVISION (v12): a 1x1 conv center-heatmap probe
            # straight off the encoder embedding map, BEFORE the deep neck/Path-A/
            # context/Tanh head. Pinpoint diagnostic: at init the encoder ranks the
            # true object windows most-distinct but the separation is tiny (~0.05
            # in the normalized embedding), so the deep head gets almost no signal
            # to amplify and training crawls into a constant-output basin. This
            # short probe gives the ENCODER a direct center_focal gradient to
            # AMPLIFY that object-vs-background separation, so the main head then
            # has a strong signal. ~66 params, dropped at eval/export (only the
            # training forward reads it), so the deploy graph is unchanged.
            self.aux_center = nn.Conv2d(hidden, 2, 1)
            if prior_fg:
                with torch.no_grad():
                    self.aux_center.bias.fill_(
                        math.log(prior_fg / max(1.0 - prior_fg, 1e-6)))
        else:
            raise ValueError(f"unknown arch {arch!r} (v8|v9|v12)")

        # window-relative pixel coords, row-major to match F.unfold token order
        r = torch.arange(self.WIN, dtype=torch.float32)
        v, u = torch.meshgrid(r, r, indexing="ij")
        uv = torch.stack([(u + 0.5) / self.WIN, (v + 0.5) / self.WIN], -1)
        self.register_buffer("uv", uv.reshape(1, 1, self.WIN * self.WIN, 2))
        # same pattern tiled over the frame for the dense path (window-aligned
        # per phase because every phase crop starts at a fresh window origin)
        tile = uv.permute(2, 0, 1)  # (2, 20, 20)
        reps_h = self.IMG_H // self.WIN
        reps_w = self.IMG_W // self.WIN
        self.register_buffer("uv_tile", tile.repeat(1, reps_h, reps_w).unsqueeze(0))
        # global window centers (== mean of the center-4 pixels, D4). v12 uses
        # non-overlapping stride-20 windows (self.nh/nw already 27/48 above),
        # so its centers are (c*20+10)/960, (r*20+10)/540 — same formula with
        # STRIDE replaced by WIN (the effective stride). v8/v9 keep the
        # overlapping stride-10 53x95 grid; the two never coexist on one
        # instance (arch is fixed at construction), so "xy_map" is safe to
        # reuse for either shape.
        jj = torch.arange(self.nh, dtype=torch.float32)
        ii = torch.arange(self.nw, dtype=torch.float32)
        y, x = torch.meshgrid(jj, ii, indexing="ij")
        eff_stride = self.WIN if arch == "v12" else self.STRIDE
        xy = torch.stack(
            [
                (x * eff_stride + self.WIN / 2) / self.IMG_W,
                (y * eff_stride + self.WIN / 2) / self.IMG_H,
            ],
            -1,
        )
        self.register_buffer("xy_map", xy.permute(2, 0, 1).unsqueeze(0))  # (1,2,53,95) v8/v9, (1,2,27,48) v12
        # overlap counts for exact cell averaging (1/2/4 corners/edges/interior)
        ones = torch.ones(1, 1, self.nh, self.nw)
        self.register_buffer("cell_counts", F.conv_transpose2d(ones, torch.ones(1, 1, 2, 2)))
        self.register_buffer("cell_kernel", torch.ones(3, 1, 2, 2))

    @classmethod
    def from_state_dict(cls, sd, **kwargs):
        """Rebuild a model with arch/hidden/stem/Path-A inferred from checkpoint
        shapes (v8 pre/post-D37 checkpoints and v9 checkpoints all load)."""
        if any(k.startswith("neck.") for k in sd):  # v9 or v12 signature
            stem = "edge_dq4" if any(k.startswith("stem_mod.dq_in.") for k in sd) \
                else ("edge_dq" if any(k.startswith("stem_mod.") for k in sd)
                      else "highpass")
            # v12's CenterHead has fc1/fc2 like RegionHeadV9(proto=False) but
            # NO prototypes and fc2 projects to exactly 4 outputs (center_mann,
            # center_tent, dx, dy) instead of 3 class logits — that's the only
            # shape signature distinguishing the two once "neck." is present.
            is_v12 = "head.fc2.weight" in sd \
                and sd["head.fc2.weight"].shape[0] == 4 \
                and "head.prototypes" not in sd
            model = cls(
                arch="v12" if is_v12 else "v9", stem=stem,
                hidden=sd["encoder.fc2.weight"].shape[0],
                h1=sd["encoder.fc1.weight"].shape[0],
                neck_rounds=sum(1 for k in sd if k.startswith("neck.dw.")
                                and k.endswith(".weight")),
                head_width=sd["head.fc1.weight"].shape[0],
                path_a_per_channel=sd["pools.0.weight"].shape[0] > 1,
                aux_head="aux.weight" in sd,
                head_proto="head.prototypes" in sd,
                **kwargs)
            # migrate pre-v10 stems: 4 separate edge convs -> one fused groups=12
            # (stem_mod.edges.{0..3}.weight -> stem_mod.edge.weight). Same params,
            # just concatenated on the out-channel axis.
            old_edges = sorted(k for k in sd if k.startswith("stem_mod.edges."))
            if old_edges and "stem_mod.edge.weight" not in sd:
                sd = dict(sd)
                sd["stem_mod.edge.weight"] = torch.cat(
                    [sd.pop(k) for k in old_edges], 0)
            model.load_state_dict(sd, strict=True)
            return model
        hidden = sd["encoder.mlp.0.weight"].shape[0]
        stem = "edge_dq" if any(k.startswith("stem_mod.") for k in sd) else "highpass"
        per_channel = sd["pools.0.weight"].shape[0] > 1
        model = cls(hidden=hidden, stem=stem, path_a_per_channel=per_channel, **kwargs)
        # pre-D36 checkpoints (e.g. runs/anet/good.pt) predate path_dq; those
        # convs are identity-init, so leaving them at init IS the old model
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith("path_dq.")] + list(unexpected)
        if bad:
            raise RuntimeError(f"checkpoint/model mismatch beyond path_dq: {bad}")
        return model

    def _ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _features(self, img):  # (B,3,540,960) -> (B,feat,540,960)
        if self.stem in ("edge_dq", "edge_dq4"):
            return self.stem_mod(img)
        rgb = self.quat(img)
        return torch.cat([rgb, self.grad(rgb)], 1)

    def _map_dense_all(self, feat, pad_mode):  # (B,feat,540,960) -> (B,hidden,53,95)
        # ALL FOUR stride-2 phases ride the batch dim through ONE encoder pass
        # (the phase loop used to launch the whole 96%-FLOP encoder 4x — pure
        # dispatch overhead on a launch-bound tiny model). Legal because every
        # encoder op is tile-local (1x1 convs, per-tile blur, non-overlapping
        # 20x20 pool) and the stem's cross-tile receptive field already ran on
        # the full frame in _features. Each phase crop is padded back to
        # 540x960 so the four share the same tile grid; the padded row/col land
        # exactly on the cropped index 53/95 (pixel_shuffle interleave below).
        #   pad_mode "constant" (eval/export): zero pad — those garbage tiles are
        #     discarded and eval BN uses frozen running stats, so their values
        #     never touch a real output. Kept for the ONNX/Hailo graph (a
        #     ConstantPad is what the CoreML partitioner and DFC expect).
        #   pad_mode "replicate" (train): edge-replicate so the ~3% padded tiles
        #     are IN-DISTRIBUTION and don't skew the shared BN *batch* statistics
        #     the valid tiles are normalized against (a joint 4-phase BN batch,
        #     vs the old per-phase stats — larger and cleaner, never garbage).
        b = feat.shape[0]
        ph, pw = self.nh // 2 + 1, self.nw // 2 + 1  # 27x48 per-phase grid
        crops = []
        for oy, ox in self.PHASES:
            hp = ((self.IMG_H - oy) // self.WIN) * self.WIN
            wp = ((self.IMG_W - ox) // self.WIN) * self.WIN
            crops.append(F.pad(feat[:, :, oy : oy + hp, ox : ox + wp],
                               (0, self.IMG_W - wp, 0, self.IMG_H - hp), mode=pad_mode))
        x = torch.cat(crops, 0)  # (4B, feat, 540, 960), phase-major
        # match uv to the stream dtype: the buffer is fp32, and cat's type
        # promotion silently upcast the ENTIRE Stage-1 residual stream to fp32
        # under bf16 autocast (autocast only re-casts conv inputs; every saved
        # elementwise intermediate stayed fp32 — measured ~0.5 GiB/img)
        uv = self.uv_tile.to(x.dtype).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], 1)
        e = self.encoder.forward_dense(
            x, ckpt=self.use_checkpoint and self.training)  # (4B, C, 27, 48)
        e = e.reshape(4, b, -1, ph, pw).permute(1, 2, 0, 3, 4)  # (B, C, 4, 27, 48)
        # interleave the four stride-2 phase grids with pixel_shuffle instead of
        # strided scatter: identical placement (phase (oy,ox) -> out[oy/10::2,
        # ox/10::2]), but ScatterND forced CPU partitions in the CoreML EP; the
        # padded row/col land at index 53/95 and are cropped off
        m = F.pixel_shuffle(e.flatten(1, 2), 2)
        return m[:, :, : self.nh, : self.nw]

    def _map_dense(self, feat):  # train path: replicate-pad (BN-batch-safe)
        return self._map_dense_all(feat, "replicate")

    def _map_dense_batched(self, feat):  # eval/export path: zero-pad (graph-stable)
        return self._map_dense_all(feat, "constant")

    def _map_windowed(self, feat):  # reference path
        b = feat.shape[0]
        win = F.unfold(feat, self.WIN, stride=self.STRIDE)  # (B, feat*400, 5035)
        win = win.reshape(b, self.feat, self.WIN * self.WIN, self.n_win).permute(0, 3, 2, 1)
        # uv cast to the stream dtype: cat's type promotion would silently
        # upcast the whole token stream to fp32 under bf16 autocast (same bug
        # class as _map_dense_all's uv_tile, fixed there earlier)
        x = torch.cat([win, self.uv.to(win.dtype).expand(b, self.n_win, -1, -1)], -1)
        emb = self._ckpt(self.encoder, x.reshape(-1, self.WIN * self.WIN, self.in_dim))
        return emb.reshape(b, self.n_win, -1).permute(0, 2, 1).reshape(b, -1, self.nh, self.nw)

    def _tail(self, m16):  # (B,hidden,53,95) embedding map -> (B,3,54,96) cells
        b = m16.shape[0]
        # xy cast to the stream dtype (bf16 fp32-promotion guard, as in v9)
        m = torch.cat([m16, self.xy_map.to(m16.dtype).expand(b, -1, -1, -1)], 1)

        maps = [dq(p(m)) for p, dq in zip(self.pools, self.path_dq)]  # Path A + per-path DQ (D14, D36)
        states = torch.stack([gp(mp) for gp, mp in zip(self.globals_, maps)], 1)  # (B,3,256)
        gtoks = self.mix(states)  # (B, 16, 18)

        emb = m.flatten(2).permute(0, 2, 1)  # (B, W, 18)
        own = emb.unsqueeze(2)  # (B, W, 1, 18)
        local = torch.stack([mp.flatten(2).permute(0, 2, 1) for mp in maps], 2)  # (B,W,3,18)
        ltoks = torch.cat([own, local], 2)  # (B, W, 4, 18) — per-window stream (D31)

        wlogits = self.head(ltoks, gtoks).permute(0, 2, 1).reshape(b, 3, self.nh, self.nw)
        cells = F.conv_transpose2d(wlogits, self.cell_kernel.to(wlogits.dtype), groups=3)
        return cells / self.cell_counts.to(wlogits.dtype)  # (B, 3, 54, 96)

    # ------------------------------------------------------------------- v9
    def _phase_batch(self, feat, pad_mode):  # (B,feat,540,960) -> (4B,in_dim,540,960)
        crops = []
        for oy, ox in self.PHASES:
            hp = ((self.IMG_H - oy) // self.WIN) * self.WIN
            wp = ((self.IMG_W - ox) // self.WIN) * self.WIN
            crops.append(F.pad(feat[:, :, oy: oy + hp, ox: ox + wp],
                               (0, self.IMG_W - wp, 0, self.IMG_H - hp), mode=pad_mode))
        x = torch.cat(crops, 0)  # (4B, feat, 540, 960), phase-major
        uv = self.uv_tile.to(x.dtype).expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, uv], 1)

    def _interleave(self, e, b):  # (4B, C, 27, 48) -> (B, C, 53, 95)
        ph, pw = self.nh // 2 + 1, self.nw // 2 + 1
        e = e.reshape(4, b, -1, ph, pw).permute(1, 2, 0, 3, 4)
        m = F.pixel_shuffle(e.flatten(1, 2), 2)
        return m[:, :, : self.nh, : self.nw]

    def _embed_map_v9(self, feat):  # (B,feat,540,960) -> (B,hidden,53,95)
        if self.fused_pool is not None:
            pooled = self.fused_pool(feat)  # fused Triton Stage 1 (D40)
        else:
            x = self._phase_batch(feat, "replicate" if self.training else "constant")
            p = self.encoder.pool_features_dense(
                x, ckpt=self.use_checkpoint and self.training)  # (4B, h1, 27, 48)
            pooled = self._interleave(p, feat.shape[0])
        return self.encoder.embed(pooled)

    def _tail_v9(self, m_emb):  # (B,hidden,53,95) -> cells (+ aux cells in train)
        b = m_emb.shape[0]
        m0 = torch.cat([m_emb, self.xy_map.expand(b, -1, -1, -1).to(m_emb.dtype)], 1)
        m = self.neck(m0)
        maps = [dq(p(m)) for p, dq in zip(self.pools, self.path_dq)]
        ctx = self.context(maps)  # (B, d)
        emb = m.flatten(2).permute(0, 2, 1)  # (B, W, d)
        own = emb.unsqueeze(2)
        local = torch.stack([mp.flatten(2).permute(0, 2, 1) for mp in maps], 2)
        ltoks = torch.cat([own, local], 2)  # (B, W, 4, d)
        if self.training:
            # dict return in training: cells + optional aux probe + the per-window
            # metric embedding z (B, width, 53, 95) for proto_metric_loss. Eval /
            # export stays a bare tensor (ONNX-safe, unchanged graph).
            wl, z = self.head.logits_z(ltoks, ctx)
            cells = self._cells(wl.permute(0, 2, 1).reshape(b, 3, self.nh, self.nw))
            aux_cells = self._cells(self.aux(m0)) if self.aux is not None else None
            zmap = z.permute(0, 2, 1).reshape(b, -1, self.nh, self.nw)
            return {"cells": cells, "aux": aux_cells, "z": zmap}
        wlogits = self.head(ltoks, ctx).permute(0, 2, 1).reshape(b, 3, self.nh, self.nw)
        return self._cells(wlogits)

    def _cells(self, wlogits):  # (B,3,53,95) window logits -> (B,3,54,96) cells
        cells = F.conv_transpose2d(wlogits, self.cell_kernel.to(wlogits.dtype), groups=3)
        return cells / self.cell_counts.to(wlogits.dtype)

    # ------------------------------------------------------------------ v12
    def _embed_map_v12(self, feat):  # (B,feat,540,960) -> (B,hidden,27,48)
        # single-phase (no 4x overlap, no _phase_batch/_interleave): stage 1's
        # pool_features_dense already tiles in non-overlapping 20x20 windows
        # via its internal avg_pool2d(..., GRID=20), so the full 540x960 frame
        # maps straight to the 27x48 window grid in one encoder pass.
        b = feat.shape[0]
        uv = self.uv_tile.to(feat.dtype).expand(b, -1, -1, -1)
        x = torch.cat([feat, uv], 1)
        pooled = self.encoder.pool_features_dense(
            x, ckpt=self.use_checkpoint and self.training)  # (B, h1, 27, 48)
        return self.encoder.embed(pooled)  # (B, hidden, 27, 48)

    def _tail_v12(self, m_emb):  # (B,hidden,27,48) -> {"heat","offset"} dict
        b = m_emb.shape[0]
        m0 = torch.cat([m_emb, self.xy_map.to(m_emb.dtype).expand(b, -1, -1, -1)], 1)
        m = self.neck(m0)
        maps = [dq(p(m)) for p, dq in zip(self.pools, self.path_dq)]
        ctx = self.context(maps)  # (B, d)
        emb = m.flatten(2).permute(0, 2, 1)  # (B, W, d)
        own = emb.unsqueeze(2)
        local = torch.stack([mp.flatten(2).permute(0, 2, 1) for mp in maps], 2)
        ltoks = torch.cat([own, local], 2)  # (B, W, 4, d), W = 27*48 = 1296
        out = self.head(ltoks, ctx)  # (B, W, 4): [center_mann, center_tent, dx, dy]
        out = out.permute(0, 2, 1).reshape(b, 4, self.nh, self.nw)
        # both train and eval return the same heat/offset: the loss/metrics side
        # (center_focal_loss/offset_l1, CenterObjectMetrics) always wants raw
        # heat + offset logits, no cell-average step (v12 predicts at the
        # native 27x48 window grid, not an overlap-averaged pixel-cell grid)
        result = {"heat": out[:, 0:2], "offset": out[:, 2:4]}
        # train-only deep-supervision probe on the raw embedding map (see __init__)
        if self.training and getattr(self, "aux_center", None) is not None:
            result["aux_heat"] = self.aux_center(m_emb)  # (B, 2, 27, 48)
        return result

    def forward(self, img):
        feat = self._features(img)
        if self.arch == "v12":
            return self._tail_v12(self._embed_map_v12(feat))
        if self.arch == "v9":
            return self._tail_v9(self._embed_map_v9(feat))
        if self.dense:
            # same single-pass batched encoder in both modes; only the phase-pad
            # mode differs (replicate keeps BN batch stats clean in training,
            # constant keeps the exported graph Hailo/CoreML-friendly)
            m16 = self._map_dense_all(feat, "replicate" if self.training else "constant")
        else:
            m16 = self._map_windowed(feat)
        return self._tail(m16)

    def reg_losses(self):
        if self.arch in ("v9", "v12"):
            l2 = self.encoder.reg_l2() + self.context.reg_l2() + self.head.reg_l2()
        else:
            l2 = self.encoder.reg_l2() + self.mix.reg_l2() + self.head.reg_l2()
        l1 = sum(p.reg_l1() for p in self.pools)
        return l2, l1

    @torch.no_grad()
    def export_onnx(self, path, opset=18, batch=1):
        """Deploy-form export: Hailo DFC compile spike + fast local inference via
        ONNX Runtime (see scripts/export_onnx.py, anet/onnxrt.py). Quaternion and
        blur sigmas evaluate to constants when traced; BN folding is the DFC's job.
        opset 18 is the torch-2.12 dynamo exporter's native target (17 forces a
        version-converter pass that fails on Resize)."""
        self.eval()
        prev = self.use_checkpoint
        self.use_checkpoint = False
        dummy = torch.zeros(batch, 3, self.IMG_H, self.IMG_W)
        torch.onnx.export(
            self, dummy, path, opset_version=opset,
            input_names=["frame"], output_names=["cells"],
        )
        self.use_checkpoint = prev
