import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .blocks import DualQuaternionRGB, EdgeDQStem
from .encoder import WindowEncoder
from .globalmix import GlobalCosineMix
from .head import RegionHead
from .pyramid import GatedGlobalPool, ScalarKernelPool


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

    def __init__(self, use_checkpoint=True, dense=True, hidden=16, stem="highpass"):
        super().__init__()
        self.nh = (self.IMG_H - self.WIN) // self.STRIDE + 1  # 53
        self.nw = (self.IMG_W - self.WIN) // self.STRIDE + 1  # 95
        self.n_win = self.nh * self.nw  # 5035
        self.use_checkpoint = use_checkpoint
        self.dense = dense
        self.stem = stem

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
        if stem == "edge_dq":
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
            raise ValueError(f"unknown stem {stem!r} (highpass|edge_dq)")
        self.in_dim = self.feat + 2  # + (u,v) window-relative coords
        self.encoder = WindowEncoder(hidden, in_dim=self.in_dim)
        self.pools = nn.ModuleList([ScalarKernelPool(d, k) for k in (3, 7, 11)])
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
        self.head = RegionHead(dim=d)

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
        # global window centers (== mean of the center-4 pixels, D4)
        jj = torch.arange(self.nh, dtype=torch.float32)
        ii = torch.arange(self.nw, dtype=torch.float32)
        y, x = torch.meshgrid(jj, ii, indexing="ij")
        xy = torch.stack(
            [
                (x * self.STRIDE + self.WIN / 2) / self.IMG_W,
                (y * self.STRIDE + self.WIN / 2) / self.IMG_H,
            ],
            -1,
        )
        self.register_buffer("xy_map", xy.permute(2, 0, 1).unsqueeze(0))  # (1,2,53,95)
        # overlap counts for exact cell averaging (1/2/4 corners/edges/interior)
        ones = torch.ones(1, 1, self.nh, self.nw)
        self.register_buffer("cell_counts", F.conv_transpose2d(ones, torch.ones(1, 1, 2, 2)))
        self.register_buffer("cell_kernel", torch.ones(3, 1, 2, 2))

    @classmethod
    def from_state_dict(cls, sd, **kwargs):
        """Rebuild a model with hidden/stem inferred from checkpoint shapes."""
        hidden = sd["encoder.mlp.0.weight"].shape[0]
        stem = "edge_dq" if any(k.startswith("stem_mod.") for k in sd) else "highpass"
        model = cls(hidden=hidden, stem=stem, **kwargs)
        model.load_state_dict(sd)
        return model

    def _ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _features(self, img):  # (B,3,540,960) -> (B,feat,540,960)
        if self.stem == "edge_dq":
            return self.stem_mod(img)
        rgb = self.quat(img)
        return torch.cat([rgb, self.grad(rgb)], 1)

    def _map_dense(self, feat):  # (B,feat,540,960) -> (B,hidden,53,95)
        b = feat.shape[0]
        ph, pw = self.nh // 2 + 1, self.nw // 2 + 1  # padded phase grid 27x48
        es = []
        for oy, ox in self.PHASES:
            hp = ((self.IMG_H - oy) // self.WIN) * self.WIN
            wp = ((self.IMG_W - ox) // self.WIN) * self.WIN
            x = torch.cat(
                [
                    feat[:, :, oy : oy + hp, ox : ox + wp],
                    self.uv_tile[:, :, :hp, :wp].expand(b, -1, -1, -1),
                ],
                1,
            )
            e = self._ckpt(self.encoder.forward_dense, x)  # (B,16,hp/20,wp/20)
            es.append(F.pad(e, (0, pw - e.shape[-1], 0, ph - e.shape[-2])))
        # interleave the four stride-2 phase grids with pixel_shuffle instead of
        # strided scatter: identical placement (phase (oy,ox) -> out[oy/10::2,
        # ox/10::2]), but ScatterND forced CPU partitions in the CoreML EP; the
        # padded row/col land at index 53/95 and are cropped off
        m = F.pixel_shuffle(torch.stack(es, 2).flatten(1, 2), 2)
        return m[:, :, : self.nh, : self.nw]

    def _map_dense_batched(self, feat):  # eval/export: 4 phases in ONE encoder pass
        # Every encoder op is tile-local and eval BN uses running stats, so the
        # 4 phase crops (padded back to 540x960 to share the tile grid) can ride
        # the batch dim: one encoder trace instead of four -> ~4x fewer kernel
        # dispatches (the CoreML/MPS bottleneck). Padding creates garbage tiles
        # whose outputs land exactly on the cropped row 53 / col 95. Not used in
        # training: batch-stat BNs would mix garbage tiles into the statistics.
        b = feat.shape[0]
        ph, pw = self.nh // 2 + 1, self.nw // 2 + 1  # 27x48 per-phase grid
        crops = []
        for oy, ox in self.PHASES:
            hp = ((self.IMG_H - oy) // self.WIN) * self.WIN
            wp = ((self.IMG_W - ox) // self.WIN) * self.WIN
            crops.append(F.pad(feat[:, :, oy : oy + hp, ox : ox + wp],
                               (0, self.IMG_W - wp, 0, self.IMG_H - hp)))
        x = torch.cat(crops, 0)  # (4B, feat, 540, 960), phase-major
        uv = torch.zeros_like(x[:, :2]) + self.uv_tile  # expand w/o shape math
        x = torch.cat([x, uv], 1)
        e = self.encoder.forward_dense(x)  # (4B, C, 27, 48)
        e = e.reshape(4, b, -1, ph, pw).permute(1, 2, 0, 3, 4)  # (B, C, 4, 27, 48)
        m = F.pixel_shuffle(e.flatten(1, 2), 2)
        return m[:, :, : self.nh, : self.nw]

    def _map_windowed(self, feat):  # reference path
        b = feat.shape[0]
        win = F.unfold(feat, self.WIN, stride=self.STRIDE)  # (B, feat*400, 5035)
        win = win.reshape(b, self.feat, self.WIN * self.WIN, self.n_win).permute(0, 3, 2, 1)
        x = torch.cat([win, self.uv.expand(b, self.n_win, -1, -1)], -1)
        emb = self._ckpt(self.encoder, x.reshape(-1, self.WIN * self.WIN, self.in_dim))
        return emb.reshape(b, self.n_win, -1).permute(0, 2, 1).reshape(b, -1, self.nh, self.nw)

    def _tail(self, m16):  # (B,hidden,53,95) embedding map -> (B,3,54,96) cells
        b = m16.shape[0]
        m = torch.cat([m16, self.xy_map.expand(b, -1, -1, -1)], 1)  # (B,18,53,95)

        maps = [dq(p(m)) for p, dq in zip(self.pools, self.path_dq)]  # Path A + per-path DQ (D14, D36)
        states = torch.stack([gp(mp) for gp, mp in zip(self.globals_, maps)], 1)  # (B,3,256)
        gtoks = self.mix(states)  # (B, 16, 18)

        emb = m.flatten(2).permute(0, 2, 1)  # (B, W, 18)
        own = emb.unsqueeze(2)  # (B, W, 1, 18)
        local = torch.stack([mp.flatten(2).permute(0, 2, 1) for mp in maps], 2)  # (B,W,3,18)
        ltoks = torch.cat([own, local], 2)  # (B, W, 4, 18) — per-window stream (D31)

        wlogits = self.head(ltoks, gtoks).permute(0, 2, 1).reshape(b, 3, self.nh, self.nw)
        cells = F.conv_transpose2d(wlogits, self.cell_kernel, groups=3)
        return cells / self.cell_counts  # (B, 3, 54, 96)

    def forward(self, img):
        feat = self._features(img)
        if self.dense:
            m16 = self._map_dense(feat) if self.training else self._map_dense_batched(feat)
        else:
            m16 = self._map_windowed(feat)
        return self._tail(m16)

    def reg_losses(self):
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
