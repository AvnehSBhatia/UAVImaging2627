import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .blocks import DualQuaternionRGB
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

    def __init__(self, use_checkpoint=True, dense=True, hidden=16):
        super().__init__()
        self.nh = (self.IMG_H - self.WIN) // self.STRIDE + 1  # 53
        self.nw = (self.IMG_W - self.WIN) // self.STRIDE + 1  # 95
        self.n_win = self.nh * self.nw  # 5035
        self.use_checkpoint = use_checkpoint
        self.dense = dense

        # hidden=16 is the 17k spec model; hidden=24 is the pre-registered
        # capacity mitigation (ARCHITECTURE §8 risk 2, ~24k params with the
        # Path-B ripple). d = embedding + global (x,y) coords, 18 at spec.
        d = hidden + 2
        self.quat = DualQuaternionRGB()
        self.encoder = WindowEncoder(hidden)
        self.pools = nn.ModuleList([ScalarKernelPool(d, k) for k in (3, 7, 11)])
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

    def _ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _map_dense(self, rgb):  # (B,3,540,960) -> (B,16,53,95)
        b = rgb.shape[0]
        out = None
        for oy, ox in self.PHASES:
            hp = ((self.IMG_H - oy) // self.WIN) * self.WIN
            wp = ((self.IMG_W - ox) // self.WIN) * self.WIN
            x = torch.cat(
                [
                    rgb[:, :, oy : oy + hp, ox : ox + wp],
                    self.uv_tile[:, :, :hp, :wp].expand(b, -1, -1, -1),
                ],
                1,
            )
            e = self._ckpt(self.encoder.forward_dense, x)  # (B,16,hp/20,wp/20)
            if out is None:
                out = e.new_zeros(b, e.shape[1], self.nh, self.nw)
            out[:, :, oy // self.STRIDE :: 2, ox // self.STRIDE :: 2] = e
        return out

    def _map_windowed(self, rgb):  # reference path
        b = rgb.shape[0]
        win = F.unfold(rgb, self.WIN, stride=self.STRIDE)  # (B, 3*400, 5035)
        win = win.reshape(b, 3, self.WIN * self.WIN, self.n_win).permute(0, 3, 2, 1)
        x = torch.cat([win, self.uv.expand(b, self.n_win, -1, -1)], -1)
        emb = self._ckpt(self.encoder, x.reshape(-1, self.WIN * self.WIN, 5))
        return emb.reshape(b, self.n_win, -1).permute(0, 2, 1).reshape(b, -1, self.nh, self.nw)

    def forward(self, img):
        b = img.shape[0]
        rgb = self.quat(img)
        m16 = self._map_dense(rgb) if self.dense else self._map_windowed(rgb)
        m = torch.cat([m16, self.xy_map.expand(b, -1, -1, -1)], 1)  # (B,18,53,95)

        maps = [p(m) for p in self.pools]  # Path A, full res (D14)
        states = torch.stack([gp(mp) for gp, mp in zip(self.globals_, maps)], 1)  # (B,3,256)
        gtoks = self.mix(states)  # (B, 16, 18)

        emb = m.flatten(2).permute(0, 2, 1)  # (B, W, 18)
        own = emb.unsqueeze(2)  # (B, W, 1, 18)
        local = torch.stack([mp.flatten(2).permute(0, 2, 1) for mp in maps], 2)  # (B,W,3,18)
        toks = torch.cat(
            [gtoks.unsqueeze(1).expand(-1, self.n_win, -1, -1), own, local], 2
        )  # (B, W, 20, 18)

        wlogits = self.head(toks).permute(0, 2, 1).reshape(b, 3, self.nh, self.nw)
        cells = F.conv_transpose2d(wlogits, self.cell_kernel, groups=3)
        return cells / self.cell_counts  # (B, 3, 54, 96)

    def reg_losses(self):
        l2 = self.encoder.reg_l2() + self.mix.reg_l2() + self.head.reg_l2()
        l1 = sum(p.reg_l1() for p in self.pools)
        return l2, l1

    @torch.no_grad()
    def export_onnx(self, path, opset=17):
        """Deploy-form export for the Hailo DFC compile spike. Quaternion and
        blur sigmas evaluate to constants when traced; BN folding is the DFC's job."""
        self.eval()
        prev = self.use_checkpoint
        self.use_checkpoint = False
        dummy = torch.zeros(1, 3, self.IMG_H, self.IMG_W)
        torch.onnx.export(
            self, dummy, path, opset_version=opset,
            input_names=["frame"], output_names=["cells"],
        )
        self.use_checkpoint = prev
