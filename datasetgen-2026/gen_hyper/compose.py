"""Hyper-accurate frame composition: bg-only or exactly one labeled object.

Scenarios (single-object frames only):
  open_field      — grass/dirt/runway, optional light clutter away from object
  runway_drygrass — prefer runway/grass buckets, object on grass-ish region
  tree_clearing   — vegetation RING around object; center kept clear
  brush_occlusion — vegetation pasted OVER part of a mannequin (or light
                    perimeter veg for tent). Label kept if visibility ok.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np

from gen2.assets import RENDER_PX_PER_M, AssetLibrary, Background
from gen2.harmonize import composite, pick_blend, reinhard_toward_bg
from gen2.scene import (
    PlacedObject,
    _local_patch,
    _prepare_background,
    _resize_set,
    _rotate_expand,
)
from gen2.shadows import apply_shadow_pass, synth_shadow


def attach_web_backgrounds(lib: AssetLibrary, web_dir: str | Path, default_gsd=0.018):
    """Index optional web nadir JPGs/PNGs as bucket 'web'."""
    root = Path(web_dir)
    if not root.is_dir():
        return
    items = []
    for p in sorted(list(root.glob("*.jpg")) + list(root.glob("*.jpeg"))
                    + list(root.glob("*.png"))):
        items.append(Background(p, float(default_gsd), "web"))
    if items:
        lib.backgrounds["web"] = items


def _pick_scenario(cfg, rng: random.Random) -> str:
    w = cfg.scenarios.raw()
    return rng.choices(list(w), weights=list(w.values()), k=1)[0]


def _paste_veg(cfg, rng, lib, canvas, placed, px_per_m_bg, sun_az, sun_el,
               shadow_op, cx, cy, scale_mult=1.0):
    """Paste one vegetation cutout near (cx, cy). Does not update labels."""
    if "vegetation" not in lib.occluders:
        return None
    occ = rng.choice(lib.occluders["vegetation"])
    bgr, a = lib.load_cutout(occ)
    factor = px_per_m_bg / occ.px_per_m * rng.uniform(0.7, 1.25) * scale_mult
    H, W = canvas.shape[:2]
    if max(bgr.shape[:2]) * factor > 500:
        factor *= 500 / (max(bgr.shape[:2]) * factor)
    bgr, a = _resize_set([bgr, a], factor)
    if rng.random() < 0.5:
        bgr, a = bgr[:, ::-1].copy(), a[:, ::-1].copy()
    bgr, a = _rotate_expand([bgr, a], rng.uniform(0, 360))
    h, w = a.shape
    x = int(cx - w / 2 + rng.uniform(-0.15, 0.15) * w)
    y = int(cy - h / 2 + rng.uniform(-0.15, 0.15) * h)
    if shadow_op > 0.02:
        sh, ox, oy = synth_shadow(a, px_per_m_bg, occ.height_m, sun_az, sun_el)
        apply_shadow_pass(canvas, sh, x + ox, y + oy, shadow_op * 0.4)
    patch = _local_patch(canvas, x, y, w, h, cfg.harmonize.local_context_scale)
    bgr = reinhard_toward_bg(
        bgr, a, patch, rng.uniform(*cfg.harmonize.reinhard_strength) * 0.7)
    mode = pick_blend(rng, cfg.harmonize.blend_weights.raw())
    used = composite(canvas, bgr, a, x, y, mode, rng,
                     tuple(cfg.harmonize.feather_px),
                     tuple(cfg.harmonize.feather_wide_px))
    if used is not None and placed:
        hard = (used > 0.55).astype(np.float32)
        for p in placed:
            p.alpha *= (1.0 - hard)
    return used


def _place_object(cfg, rng, lib, canvas, cls, px_per_m_bg, sun_az, sun_el,
                  shadow_op, prefer_center=False):
    """Paste one object. Returns PlacedObject or None."""
    if cls not in lib.objects:
        return None
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

    H, W = canvas.shape[:2]
    ys_, xs_ = np.where(a > 0.05)
    if len(xs_) == 0:
        return None
    ox1, ox2 = int(xs_.min()), int(xs_.max()) + 1
    oy1, oy2 = int(ys_.min()), int(ys_.max()) + 1
    ow, oh = ox2 - ox1, oy2 - oy1
    if ow >= W or oh >= H or ow < 4 or oh < 4:
        return None

    if prefer_center:
        # keep object in central 60% so a vegetation ring has room
        bx = rng.randint(int(0.2 * W), max(int(0.2 * W) + 1, W - ow - int(0.2 * W)))
        by = rng.randint(int(0.2 * H), max(int(0.2 * H) + 1, H - oh - int(0.2 * H)))
        bx = int(np.clip(bx, 0, W - ow))
        by = int(np.clip(by, 0, H - oh))
    else:
        bx = rng.randint(0, W - ow)
        by = rng.randint(0, H - oh)
    x, y = bx - ox1, by - oy1
    h, w = a.shape

    if shadow is not None:
        apply_shadow_pass(canvas, shadow, x, y, shadow_op)
    patch = _local_patch(canvas, x, y, w, h, cfg.harmonize.local_context_scale)
    bgr = reinhard_toward_bg(
        bgr, a, patch, rng.uniform(*cfg.harmonize.reinhard_strength))
    mode = pick_blend(rng, cfg.harmonize.blend_weights.raw())
    used = composite(canvas, bgr, a, x, y, mode, rng,
                     tuple(cfg.harmonize.feather_px),
                     tuple(cfg.harmonize.feather_wide_px))
    if used is None:
        return None
    return (PlacedObject(_class_id(cfg, cls), used.copy(), float(used.sum())),
            (bx + ow // 2, by + oh // 2))


def _class_id(cfg, cls: str) -> int:
    # root Cfg exposes yaml lists as plain lists
    return list(cfg._d["classes"]).index(cls)


def _labels_from_placed(cfg, placed, W, H):
    labels = []
    for p in placed:
        vis = p.alpha
        visibility = float(vis.sum()) / max(p.orig_area, 1.0)
        if visibility < float(cfg.objects.min_visibility):
            continue
        ys, xs = np.where(vis > 0.35)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bw, bh = x2 - x1, y2 - y1
        if bw < int(cfg.objects.min_bbox_px) or bh < int(cfg.objects.min_bbox_px):
            continue
        labels.append((p.class_id, (x1 + bw / 2) / W, (y1 + bh / 2) / H,
                       bw / W, bh / H))
    return labels


def compose_hyper(cfg, lib: AssetLibrary, idx: int, mode: str, cls: str | None):
    """mode: 'bg' | 'single'. cls required for single ('mannequin'|'tent').

    Returns (canvas float32 BGR, labels, meta).
    """
    rng = random.Random(int(cfg.project.seed) + idx * 10007 + (0 if mode == "bg" else 1))
    W, H = int(cfg.dataset.image_width), int(cfg.dataset.image_height)
    scenario = "bg_only" if mode == "bg" else _pick_scenario(cfg, rng)

    # bucket weights must only use existing library buckets
    raw_w = dict(cfg.scene.background_bucket_weights.raw())
    # never sample underscore staging pools (_rejected, _oblique, …)
    raw_w = {k: v for k, v in raw_w.items()
             if k in lib.backgrounds and not k.startswith("_")}
    if not raw_w:
        raise RuntimeError("no backgrounds indexed — check asset_root")
    # scenario bias
    if scenario == "runway_drygrass":
        for k in list(raw_w):
            raw_w[k] *= 1.5 if k in ("runway", "grass") else (0.8 if k == "dirt" else 0.25)
    elif scenario == "tree_clearing":
        for k in list(raw_w):
            raw_w[k] *= 1.7 if k in ("forest", "grass") else 0.4
    elif scenario == "brush_occlusion":
        for k in list(raw_w):
            raw_w[k] *= 1.4 if k in ("grass", "dirt", "runway") else 0.4
    bucket = rng.choices(list(raw_w), weights=list(raw_w.values()), k=1)[0]
    bg = lib.pick_background(rng, bucket)
    canvas, gsd = _prepare_background(cfg, rng, bg)
    gsd *= rng.uniform(*cfg.scene.altitude_jitter)
    px_per_m_bg = 1.0 / gsd

    overcast = rng.random() < float(cfg.scene.overcast_prob)
    sun_az = rng.uniform(0, 360)
    sun_el = 65 if overcast else rng.choice(list(cfg.scene.sun_elevations))
    shadow_op = (rng.uniform(0.03, 0.10) if overcast
                 else rng.uniform(*cfg.scene.shadow_opacity))

    placed: list[PlacedObject] = []
    meta = {
        "mode": mode, "scenario": scenario, "bucket": bucket,
        "bg": bg.path.name, "gsd": gsd, "cls": cls or "",
        "sun_az": sun_az, "sun_el": sun_el, "overcast": overcast,
    }

    if mode == "bg":
        meta["n_labels"] = 0
        return canvas, [], meta

    assert cls in ("mannequin", "tent")
    prefer_center = scenario in ("tree_clearing", "brush_occlusion")
    result = _place_object(cfg, rng, lib, canvas, cls, px_per_m_bg,
                           sun_az, sun_el, shadow_op, prefer_center=prefer_center)
    if result is None:
        # rare: retry once without prefer_center
        result = _place_object(cfg, rng, lib, canvas, cls, px_per_m_bg,
                               sun_az, sun_el, shadow_op, prefer_center=False)
    if result is None:
        meta["n_labels"] = 0
        meta["failed_place"] = True
        return canvas, [], meta

    obj, (cx, cy) = result
    placed.append(obj)

    if scenario == "tree_clearing":
        n_lo, n_hi = [int(x) for x in cfg.objects.clearing_n_veg]
        n = rng.randint(n_lo, n_hi)
        r0, r1 = [float(x) for x in cfg.objects.clearing_ring_m]
        for _ in range(n):
            ang = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(r0, r1) * px_per_m_bg
            _paste_veg(cfg, rng, lib, canvas, placed, px_per_m_bg,
                       sun_az, sun_el, shadow_op,
                       int(cx + math.cos(ang) * dist),
                       int(cy + math.sin(ang) * dist),
                       scale_mult=rng.uniform(0.9, 1.4))

    elif scenario == "brush_occlusion" and cls == "mannequin":
        # cover a fraction of the object with vegetation offset from centre
        cover = rng.uniform(*[float(x) for x in cfg.objects.brush_cover])
        # 1–2 vegetation blobs toward a random side
        for _ in range(rng.randint(1, 2)):
            ang = rng.uniform(0, 2 * math.pi)
            # land on the object, not a distant ring
            dist = rng.uniform(0.15, 0.45) * min(
                canvas.shape[1], canvas.shape[0]) * 0.05 * px_per_m_bg / 50
            # better: offset in object-local px from bbox
            ys, xs = np.where(placed[0].alpha > 0.35)
            if len(xs):
                ox = int(np.mean(xs) + math.cos(ang) * (xs.max() - xs.min()) * 0.35)
                oy = int(np.mean(ys) + math.sin(ang) * (ys.max() - ys.min()) * 0.35)
            else:
                ox, oy = cx, cy
            _paste_veg(cfg, rng, lib, canvas, placed, px_per_m_bg,
                       sun_az, sun_el, shadow_op, ox, oy,
                       scale_mult=rng.uniform(0.6, 1.1) * (0.8 + cover))
        meta["brush_cover_target"] = cover

    elif scenario in ("open_field", "runway_drygrass"):
        # sparse far clutter — never on the object centroid
        if "vegetation" in lib.occluders and rng.random() < 0.45:
            for _ in range(rng.randint(0, 2)):
                ang = rng.uniform(0, 2 * math.pi)
                dist = rng.uniform(8.0, 18.0) * px_per_m_bg
                _paste_veg(cfg, rng, lib, canvas, [], px_per_m_bg,
                           sun_az, sun_el, shadow_op,
                           int(cx + math.cos(ang) * dist),
                           int(cy + math.sin(ang) * dist),
                           scale_mult=rng.uniform(0.7, 1.2))
        # light brush chance on mannequin even in open_field
        if cls == "mannequin" and rng.random() < 0.35:
            ys, xs = np.where(placed[0].alpha > 0.35)
            if len(xs):
                ox = int(np.percentile(xs, rng.uniform(60, 90)))
                oy = int(np.percentile(ys, rng.uniform(60, 90)))
                _paste_veg(cfg, rng, lib, canvas, placed, px_per_m_bg,
                           sun_az, sun_el, shadow_op, ox, oy,
                           scale_mult=rng.uniform(0.5, 0.9))

    elif scenario == "brush_occlusion" and cls == "tent":
        # perimeter veg only — do not bury the tent
        for _ in range(rng.randint(2, 4)):
            ang = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(2.0, 5.0) * px_per_m_bg
            _paste_veg(cfg, rng, lib, canvas, placed, px_per_m_bg,
                       sun_az, sun_el, shadow_op,
                       int(cx + math.cos(ang) * dist),
                       int(cy + math.sin(ang) * dist),
                       scale_mult=rng.uniform(0.8, 1.3))

    labels = _labels_from_placed(cfg, placed, W, H)
    # hyper-accurate contract: exactly 0 or 1 label; if occlusion ate the
    # object, drop to bg-equivalent (empty labels) rather than a wrong box
    if len(labels) > 1:
        labels = labels[:1]
    meta["n_labels"] = len(labels)
    meta["visibility"] = (
        float(placed[0].alpha.sum()) / max(placed[0].orig_area, 1.0) if placed else 0.0)
    return canvas, labels, meta
