"""Train-time input augmentation (D85). The pipeline had NONE — this is new.

WHY. §19.3 measured v13 on 14 real web scenes against synthetic val, same
model, same threshold:

    slice                    p at object   bg>0.30 /frame   bg p99
    synthetic                    0.570         2.64          0.146
    real web scenes              0.482         6.80          0.220

Objects respond 15% weaker AND background fires 2.6x more often — degrading
from both sides, which is what produces the inverted margins that triggered
the whole v23 line. The decisive context is that gen2 composites Blender
objects onto REAL aerial photographs, so the backgrounds are already real;
what differs on a real frame is that nothing has been through Reinhard
harmonization and sensor sim. A model trained on renderer-consistent frames
with zero augmentation is free to key on renderer statistics — absolute
sharpness, noise floor, colour balance — and that is exactly the failure
signature above.

This also re-scopes §16.2. That section measured a zero train/test
generalization gap and concluded "more data cannot help". But both splits
come from the SAME generator, so a zero gap between them says nothing about
transfer to real photographs; it measured consistency within one
distribution, not generalization out of it.

WHAT IS AND IS NOT TOUCHED. Train-only, zero parameters, zero deploy change:
the exported graph, the <=40k budget and Hailo legality are untouched by
construction, and no D63 identity contract is needed because the MODEL does
not change. What does change is the distribution DeployNorm's running stats
observe — deliberately, and the trainer's seeding passes run through the same
path so stats start on the augmented distribution (D39).

WHERE EACH PIECE RUNS.
  flips        in the DATASET, before rasterization, so heat/offset/grid/band
               targets are regenerated from the flipped boxes and stay exact
               (no resampling, no interpolation — pure array reversal).
  photometric  on-GPU in the Trainer, fused after the uint8->float step. ROCm
               runs with num_workers=0 (spawn deadlocks on fork'd MIOpen
               mutexes), so any CPU-side augmentation would serialize
               straight into the training loop.

ORIENTATION. Both flips are valid for nadir imagery: at 150 ft AGL looking
straight down there is no gravity-defined "up", and a flipped sun angle is
just a different time of day or heading. The caveat is VisDrone, whose frames
are OBLIQUE — a vertical flip puts the horizon at the bottom. vd frames are
already downweighted (vd_weight) and are a different task (§19.1), so this is
accepted rather than special-cased; ANET_AUG_VFLIP=0 disables it if a run
ever needs to isolate that.

EXACTNESS NOTE. letterbox_params centre-pads with `(tw-nw)//2`, so when the
pad is odd the flipped canvas shifts content by ONE pixel relative to the
`cx -> 1-cx` box mapping. That is 0.1% of the canvas and ~1/20th of a cell,
below the letterbox resampling error it rides on. Synthetic frames are
1920x1080 -> exact 0.5 scale with no padding at all, so they are unaffected.
"""

import numpy as np
import torch
import torch.nn.functional as F

# fixed 3x3 Gaussian, used for the unsharp/blur axis
_BLUR_K = torch.tensor([[1.0, 2.0, 1.0],
                        [2.0, 4.0, 2.0],
                        [1.0, 2.0, 1.0]]) / 16.0


def flip_sample(image_u8, boxes, grid=None, band=None, do_h=False, do_v=False):
    """Mirror one loaded item in place-safe fashion, BEFORE rasterization.

    image_u8 (3,H,W) uint8 | boxes (N,5) [cls,cx,cy,w,h] canvas-normalized,
    -1-padded | grid (54,96) | band (2,54,96). Returns the same tuple.
    Every operation is an exact array reversal — no interpolation.
    """
    if not (do_h or do_v):
        return image_u8, boxes, grid, band
    boxes = boxes.copy()
    valid = boxes[:, 0] >= 0            # -1 padding rows must stay untouched
    if do_h:
        image_u8 = image_u8[:, :, ::-1]
        boxes[valid, 1] = 1.0 - boxes[valid, 1]
        if grid is not None:
            grid = grid[:, ::-1]
        if band is not None:
            band = band[:, :, ::-1]
    if do_v:
        image_u8 = image_u8[:, ::-1, :]
        boxes[valid, 2] = 1.0 - boxes[valid, 2]
        if grid is not None:
            grid = grid[::-1, :]
        if band is not None:
            band = band[:, ::-1, :]
    return (np.ascontiguousarray(image_u8), boxes,
            None if grid is None else np.ascontiguousarray(grid),
            None if band is None else np.ascontiguousarray(band))


def photometric(x, cfg, gen=None):
    """(B,3,H,W) float in [0,1] -> augmented, same shape/dtype/device.

    Every knob is a half-width: 0 disables that axis exactly, so components
    can be isolated one at a time (D69 interference law). Draws happen on CPU
    and move to the device, which keeps results identical across ROCm/MPS/CPU
    for a given generator and avoids per-backend generator differences.
    """
    B, _, H, W = x.shape
    dev, dt = x.device, x.dtype

    def rnd(lo, hi, c=1):
        t = torch.empty(B, c, 1, 1).uniform_(lo, hi, generator=gen)
        return t.to(dev, dt)

    def coin(p):
        t = (torch.rand(B, 1, 1, 1, generator=gen) < p)
        return t.to(dev)

    # --- blur <-> sharpen: THE axis aimed at the measured tell -------------
    # Measured (aug_premise, mean |Laplacian| of luminance): real web scenes
    # carry 2.54x the high-frequency energy of synthetic frames (median
    # 0.0989 vs 0.0389), and only 43% of real scenes land inside the raw
    # synthetic p05-p95 band. The tell is real and LARGE, so this axis is
    # deliberately ASYMMETRIC: unsharp scales high-frequency content by
    # (1+s), so covering a 2.5x gap needs s ~ 1.5, and a symmetric U(-a,+a)
    # would spend half its range making synthetic frames even smoother than
    # they already are. s<0 still blurs, so the model cannot instead learn
    # "sharp = object" in the other direction.
    if cfg.sharpen_hi > cfg.sharpen_lo:
        k = _BLUR_K.to(dev, dt).expand(3, 1, 3, 3)
        blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k, groups=3)
        s = rnd(cfg.sharpen_lo, cfg.sharpen_hi)
        x = x + s * (x - blur)

    # --- illuminant / exposure --------------------------------------------
    if cfg.channel_gain > 0:                      # white balance
        x = x * rnd(1.0 - cfg.channel_gain, 1.0 + cfg.channel_gain, c=3)
    if cfg.brightness > 0:
        x = x * rnd(1.0 - cfg.brightness, 1.0 + cfg.brightness)
    if cfg.contrast > 0:
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = mu + (x - mu) * rnd(1.0 - cfg.contrast, 1.0 + cfg.contrast)
    if cfg.gamma > 0:
        x = x.clamp_min(1e-4) ** torch.exp(rnd(-cfg.gamma, cfg.gamma))

    # --- sensor noise ------------------------------------------------------
    # Applied AFTER the tone curve, like real sensor noise relative to the
    # ISP, and gated per-image so the model sees clean frames too.
    if cfg.noise > 0:
        sigma = rnd(0.0, cfg.noise) * coin(cfg.noise_p)
        # randn_like, NOT a CPU draw moved across: a (B,3,540,960) CPU tensor
        # is ~100 MB per call and the allocate+H2D dominated everything else
        # in the block (measured 12.5 -> 0.6 ms/img on MPS).
        x = x + torch.randn_like(x) * sigma

    return x.clamp_(0.0, 1.0)
