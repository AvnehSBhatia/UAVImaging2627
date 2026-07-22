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


def _rgb2hsv_np(rgb):
    """(...,3) float 0-1 -> (h, s, v), each (...) — colorsys-order hue in 0-1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    de = d + 1e-12
    s = np.where(mx > 1e-12, d / (mx + 1e-12), 0.0)
    rc, gc, bc = (mx - r) / de, (mx - g) / de, (mx - b) / de
    h = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(d > 1e-12, (h / 6.0) % 1.0, 0.0)
    return h, s, mx


def _hsv2rgb_np(h, s, v):
    """Inverse of _rgb2hsv_np — (h,s,v) each (...) 0-1 -> (...,3) float."""
    i = np.floor(h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


def camouflage_objects(image_u8, boxes, p=0.0, a_lo=0.2, a_hi=0.6,
                       hue=0.09, sat_drop=0.85, lum_pull=0.7, classes=(0,)):
    """Push a fraction of object regions toward earth-tone / low-contrast
    camouflage, IN the composited uint8 frame, BEFORE the model sees it (D92).

    WHY. The gen2 object palette is saturated and high-contrast — bright
    clothing renders that POP off the terrain (measured: 20-sample montage is
    red/blue/purple/white/teal). So v22g_r2 fires median 0.78 on a synthetic
    mannequin but collapses to 0.17 once that same object is desaturated,
    hue-shifted to earth-tone and luminance-matched to its surround — which is
    exactly the 0.21 it fires on a REAL person lying prone in dry brush
    (webscene_check: eval_mann_only_brush 0.221, runway_scrub_mann 0.208). The
    generator NEVER produces the camouflaged case (Reinhard harmonize is
    damped 0.50-0.75 with chroma HALVED, explicitly to keep objects visible),
    so the model has no learned contrast-invariance in that regime — a
    distribution-EDGE covariate shift, not a background/tail one.

    This widens the training object-appearance distribution into that regime
    while keeping object GEOMETRY intact — only the COLOUR of pixels inside the
    box changes, the silhouette/limb structure is untouched — so the model
    learns "earth-tone person-SHAPE", not "earth-tone blob". The transform is
    the exact one contrast_probe measured the 0.78->0.17 response curve on;
    a_hi is capped well below invisibility (a=0.6 -> the object still fires
    ~0.25, hard-but-present, so the label stays honest — a careful human finds
    it, and the mission requires finding it).

    image_u8 (3,H,W) uint8 | boxes (N,5) [cls,cx,cy,w,h] canvas-normalized,
    -1-padded. Draws use torch's global RNG (per-worker reseeded like flips,
    NOT numpy's global RNG which DataLoader never reseeds). p<=0 returns the
    input untouched with NO draw, so a disabled run is bit-exact the old path.
    """
    if p <= 0.0:
        return image_u8
    C, H, W = image_u8.shape
    hw = np.ascontiguousarray(
        image_u8.transpose(1, 2, 0).astype(np.float32) / 255.0)   # (H,W,3)
    changed = False
    for j in range(len(boxes)):
        cls = boxes[j, 0]
        if cls < 0 or int(cls) not in classes:
            continue
        if float(torch.rand(1)) >= p:
            continue
        a = float(torch.empty(1).uniform_(a_lo, a_hi))
        _, cx, cy, bw, bh = boxes[j]
        x0, x1 = int((cx - bw / 2) * W), int(np.ceil((cx + bw / 2) * W))
        y0, y1 = int((cy - bh / 2) * H), int(np.ceil((cy + bh / 2) * H))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        # luminance target from a context ring (1.75x box); object is a small
        # fraction of it, so the mean reads the local terrain it must blend to.
        pxw, pxh = int((x1 - x0) * 0.75), int((y1 - y0) * 0.75)
        cx0, cy0 = max(0, x0 - pxw), max(0, y0 - pxh)
        cx1, cy1 = min(W, x1 + pxw), min(H, y1 + pxh)
        bg_v = float(hw[cy0:cy1, cx0:cx1].mean())
        obj = hw[y0:y1, x0:x1]
        h_, s_, v_ = _rgb2hsv_np(obj)
        s2 = s_ * (1.0 - sat_drop * a)
        h2 = h_ * (1.0 - a) + hue * a
        v2 = v_ * (1.0 - lum_pull * a) + bg_v * (lum_pull * a)
        hw[y0:y1, x0:x1] = _hsv2rgb_np(h2, s2, v2)
        changed = True
    if not changed:
        return image_u8
    out = np.clip(hw, 0.0, 1.0).transpose(2, 0, 1)
    return np.ascontiguousarray((out * 255.0 + 0.5).astype(np.uint8))


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
