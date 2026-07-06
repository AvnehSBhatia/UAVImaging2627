"""Scene composition: one synthetic frame per call.

Physically grounded: pixel sizes derive from background GSD (m/px) and the
render contract (100 px/m). One sun per image drives every shadow. Occlusion
is deliberate (rules: objects 'surrounded/covered by bushes/trees/vehicles').
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from .assets import AssetLibrary, Background, RENDER_PX_PER_M
from .harmonize import composite, pick_blend, reinhard_toward_bg
from .shadows import apply_shadow_pass, synth_shadow


@dataclass
class PlacedObject:
    class_id: int
    alpha: np.ndarray          # full-canvas visible alpha (float 0..1)
    orig_area: float


def _weighted_int(rng: random.Random, weights: dict) -> int:
    ks = [int(k) for k in weights.keys()]
    ws = [float(v) for v in weights.values()]
    return rng.choices(ks, weights=ws, k=1)[0]


def _rotate_expand(imgs: list[np.ndarray | None], deg: float) -> list[np.ndarray | None]:
    """Rotate a set of registered layers by deg CCW, expanding the canvas."""
    ref = next(i for i in imgs if i is not None)
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = []
    for im in imgs:
        if im is None:
            out.append(None)
            continue
        out.append(cv2.warpAffine(im, M, (nw, nh), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return out


def _resize_set(imgs: list[np.ndarray | None], factor: float) -> list[np.ndarray | None]:
    if abs(factor - 1.0) < 1e-3:
        return imgs
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    out = []
    for im in imgs:
        if im is None:
            out.append(None)
            continue
        nw = max(2, int(round(im.shape[1] * factor)))
        nh = max(2, int(round(im.shape[0] * factor)))
        out.append(cv2.resize(im, (nw, nh), interpolation=interp))
    return out


def _prepare_background(cfg, rng: random.Random, bg: Background) -> tuple[np.ndarray, float]:
    """Load, crop-zoom, flip/rot90. Returns (canvas float32, effective gsd m/px)."""
    img = cv2.imread(str(bg.path))
    if img is None:
        raise FileNotFoundError(bg.path)
    W, H = int(cfg.dataset.image_width), int(cfg.dataset.image_height)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    z = rng.uniform(*cfg.augment.crop_zoom)
    gsd = bg.gsd_m
    if z > 1.01:
        cw, ch = int(W / z), int(H / z)
        x0 = rng.randint(0, W - cw)
        y0 = rng.randint(0, H - ch)
        img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_CUBIC)
        gsd = gsd / z

    if cfg.augment.rot90:
        k = rng.randint(0, 3)
        if k:
            img = np.rot90(img, k).copy()
            if k % 2:  # keep output WxH
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if rng.random() < cfg.augment.hflip_prob:
        img = img[:, ::-1].copy()
    if rng.random() < cfg.augment.vflip_prob:
        img = img[::-1].copy()

    return img.astype(np.float32), gsd


def _occlude_previous(placed: list[PlacedObject], paste_alpha: np.ndarray | None) -> None:
    if paste_alpha is None:
        return
    # only opaque cover counts as occlusion: semi-transparent foliage alpha
    # must not silently erase labels of objects that remain clearly visible
    hard = (paste_alpha > 0.55).astype(np.float32)
    for p in placed:
        p.alpha *= (1.0 - hard)


def _paste_occluder(
    cfg, rng, lib: AssetLibrary, canvas, placed, px_per_m_bg: float,
    sun_az: float, sun_el: float, shadow_op: float,
    center_xy: tuple[int, int] | None, otype: str | None = None,
    budget: dict | None = None,
):
    occ = lib.pick_occluder(rng, otype)
    if budget is not None and occ.otype == "vehicle":
        if budget.get("vehicle", 0) >= 1:  # max one vehicle per frame
            occ = lib.pick_occluder(rng, "vegetation" if "vegetation" in lib.occluders else None)
        else:
            budget["vehicle"] = budget.get("vehicle", 0) + 1
    bgr, a = lib.load_cutout(occ)
    factor = px_per_m_bg / occ.px_per_m * rng.uniform(0.85, 1.2)
    # hard realism cap: partial/close-up source crops can blow up the naive
    # length-based px_per_m estimate (seen: half-frame bus roof). Nothing we
    # paste should exceed ~600 px (≈9 m) or ~10% of the frame area.
    H_, W_ = canvas.shape[:2]
    long_px = max(bgr.shape[:2]) * factor
    if factor <= 0 or long_px > 600 or (bgr.shape[0] * factor) * (bgr.shape[1] * factor) > 0.10 * W_ * H_:
        return
    bgr, a = _resize_set([bgr, a], factor)
    # per-instance variety so repeated cutouts don't read as copy-paste
    if rng.random() < 0.5:
        bgr, a = bgr[:, ::-1].copy(), a[:, ::-1].copy()
    if occ.otype == "vegetation":
        hsv = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.randint(-6, 6)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-25, 15), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    bgr, a = _rotate_expand([bgr, a], rng.uniform(0, 360))

    h, w = a.shape
    H, W = canvas.shape[:2]
    if center_xy is None:
        x = rng.randint(-w // 4, W - 3 * w // 4)
        y = rng.randint(-h // 4, H - 3 * h // 4)
    else:
        cx, cy = center_xy
        x = int(cx - w / 2 + rng.uniform(-w, w) * 0.4)
        y = int(cy - h / 2 + rng.uniform(-h, h) * 0.4)

    # shadow first, then body
    if shadow_op > 0.02:
        sh, ox, oy = synth_shadow(a, px_per_m_bg, occ.height_m, sun_az, sun_el)
        apply_shadow_pass(canvas, sh, x + ox, y + oy, shadow_op * 0.45)

    strength = rng.uniform(*cfg.harmonize.reinhard_strength) * 0.7
    patch = _local_patch(canvas, x, y, w, h, cfg.harmonize.local_context_scale)
    bgr = reinhard_toward_bg(bgr, a, patch, strength)

    mode = pick_blend(rng, cfg.harmonize.blend_weights.raw())
    used = composite(canvas, bgr, a, x, y, mode, rng,
                     tuple(cfg.harmonize.feather_px), tuple(cfg.harmonize.feather_wide_px))
    _occlude_previous(placed, used)


def _local_patch(canvas: np.ndarray, x: int, y: int, w: int, h: int, scale: float) -> np.ndarray:
    H, W = canvas.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    hw, hh = w * scale / 2, h * scale / 2
    x1, x2 = int(max(0, cx - hw)), int(min(W, cx + hw))
    y1, y2 = int(max(0, cy - hh)), int(min(H, cy + hh))
    return canvas[y1:y2, x1:x2]


def compose_frame(cfg, lib: AssetLibrary, idx: int):
    """Returns (bgr float32 canvas, labels list[(cls,cx,cy,w,h)], meta dict)."""
    rng = random.Random(int(cfg.project.seed) + idx)
    W, H = int(cfg.dataset.image_width), int(cfg.dataset.image_height)

    buckets = cfg.scene.background_bucket_weights.raw()
    bucket = rng.choices(list(buckets), weights=list(buckets.values()), k=1)[0]
    bg = lib.pick_background(rng, bucket)
    canvas, gsd = _prepare_background(cfg, rng, bg)
    gsd *= rng.uniform(*cfg.scene.altitude_jitter)
    px_per_m_bg = 1.0 / gsd / 1.0  # px per meter on this frame

    overcast = rng.random() < float(cfg.scene.overcast_prob)
    sun_az = rng.uniform(0, 360)
    sun_el = 65 if overcast else rng.choice(list(cfg.scene.sun_elevations))
    shadow_op = rng.uniform(0.03, 0.10) if overcast else rng.uniform(*cfg.scene.shadow_opacity)

    labels: list[tuple[int, float, float, float, float]] = []
    placed: list[PlacedObject] = []
    meta = {"bucket": bucket, "bg": bg.path.name, "gsd": gsd, "sun_az": sun_az,
            "sun_el": sun_el, "overcast": overcast}

    occ_budget: dict = {}

    # scene-level clutter FIRST so objects never look buried under random junk;
    # only deliberate occluders (below) may cover an object
    if lib.occluders:
        for _ in range(rng.randint(int(cfg.objects.clutter_anywhere.min),
                                   int(cfg.objects.clutter_anywhere.max))):
            _paste_occluder(cfg, rng, lib, canvas, placed, px_per_m_bg,
                            sun_az, sun_el, shadow_op, None, budget=occ_budget)

    background_only = rng.random() < float(cfg.dataset.background_only_ratio)
    if not background_only and lib.objects:
        class_plan: list[str] = []
        class_plan += ["mannequin"] * _weighted_int(rng, cfg.scene.mannequin_count_weights.raw())
        class_plan += ["tent"] * _weighted_int(rng, cfg.scene.tent_count_weights.raw())
        if not class_plan:  # non-background frames must contain at least one object
            class_plan.append(rng.choice(["mannequin", "tent"]))
        rng.shuffle(class_plan)

        for cls in class_plan:
            if cls not in lib.objects:
                continue
            v = lib.pick_variant(rng, cls)
            yaw = rng.uniform(0, 360)
            az = v.nearest_az(sun_az - yaw)
            el = v.nearest_el(sun_el)
            bgr, a, shadow = AssetLibrary.load_render(v, az, el)

            factor = px_per_m_bg / float(v.meta.get("px_per_m", RENDER_PX_PER_M))
            if cls == "mannequin":
                factor *= rng.uniform(0.97, 1.03)
            else:
                factor *= rng.uniform(*cfg.objects.tent_scale_jitter)
            bgr, a, shadow = _resize_set([bgr, a, shadow], factor)
            bgr, a, shadow = _rotate_expand([bgr, a, shadow], yaw)

            # place by the OBJECT's alpha bbox; the render canvas is padded for
            # the shadow throw, which is allowed to clip off-frame
            ys_, xs_ = np.where(a > 0.05)
            if len(xs_) == 0:
                continue
            ox1, ox2, oy1, oy2 = xs_.min(), xs_.max() + 1, ys_.min(), ys_.max() + 1
            ow, oh = ox2 - ox1, oy2 - oy1
            if ow >= W or oh >= H:
                continue
            # mostly fully inside; occasional edge clipping of the object itself
            if rng.random() < 0.12:
                bx = rng.randint(-ow // 3, W - 2 * ow // 3)
                by = rng.randint(-oh // 3, H - 2 * oh // 3)
            else:
                bx = rng.randint(0, W - ow)
                by = rng.randint(0, H - oh)
            x, y = bx - ox1, by - oy1  # canvas top-left so bbox lands at (bx, by)
            h, w = a.shape

            if shadow is not None:
                apply_shadow_pass(canvas, shadow, x, y, shadow_op)

            patch = _local_patch(canvas, x, y, w, h, cfg.harmonize.local_context_scale)
            bgr = reinhard_toward_bg(bgr, a, patch, rng.uniform(*cfg.harmonize.reinhard_strength))

            mode = pick_blend(rng, cfg.harmonize.blend_weights.raw())
            used = composite(canvas, bgr, a, x, y, mode, rng,
                             tuple(cfg.harmonize.feather_px), tuple(cfg.harmonize.feather_wide_px))
            if used is None:
                continue
            _occlude_previous(placed, used)
            placed.append(PlacedObject(cfg.classes.index(cls), used.copy(), float(used.sum())))

            # deliberate partial occlusion + nearby clutter
            cx, cy = bx + ow // 2, by + oh // 2
            if lib.occluders and rng.random() < float(cfg.objects.occlusion_prob):
                # vehicles surround objects but never cover them (z-order realism)
                _paste_occluder(cfg, rng, lib, canvas, placed, px_per_m_bg,
                                sun_az, sun_el, shadow_op, (cx, cy),
                                rng.choice(["vegetation", "vegetation", "debris"]),
                                budget=occ_budget)
            n_clutter = rng.randint(int(cfg.objects.clutter_near_object.min),
                                    int(cfg.objects.clutter_near_object.max))
            for _ in range(n_clutter):
                if not lib.occluders:
                    break
                ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(2.0, 7.0) * px_per_m_bg
                _paste_occluder(cfg, rng, lib, canvas, placed, px_per_m_bg,
                                sun_az, sun_el, shadow_op,
                                (int(cx + math.cos(ang) * dist), int(cy + math.sin(ang) * dist)),
                                budget=occ_budget)

    # labels from remaining visible alpha
    for p in placed:
        vis = p.alpha
        visibility = float(vis.sum()) / max(p.orig_area, 1.0)
        if visibility < float(cfg.objects.min_visibility):
            continue
        ys, xs = np.where(vis > 0.35)
        if len(xs) == 0:
            continue
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        bw, bh = x2 - x1, y2 - y1
        if bw < int(cfg.objects.min_bbox_px) or bh < int(cfg.objects.min_bbox_px):
            continue
        labels.append((p.class_id, (x1 + bw / 2) / W, (y1 + bh / 2) / H, bw / W, bh / H))

    meta["n_labels"] = len(labels)
    return canvas, labels, meta
