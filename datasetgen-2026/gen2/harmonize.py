"""Color harmonization and blend-mode ensemble.

Dwibedi et al. 2017: randomizing blend modes per instance beats any single
mode (Poisson-only actually hurt). We default to feathered alpha with damped
Reinhard LAB transfer toward local background statistics.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def reinhard_toward_bg(
    obj_bgr: np.ndarray,
    obj_alpha: np.ndarray,
    bg_patch: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Shift object LAB stats toward the local background's, damped by strength.

    obj_bgr float32 0..255, obj_alpha float32 0..1, bg_patch uint8/float32 BGR.
    """
    mask = obj_alpha > 0.5
    if mask.sum() < 16 or bg_patch.size == 0:
        return obj_bgr

    obj_lab = cv2.cvtColor(np.clip(obj_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(np.clip(bg_patch, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    o_mean = obj_lab[mask].mean(axis=0)
    o_std = obj_lab[mask].std(axis=0) + 1e-5
    b_mean = bg_lab.reshape(-1, 3).mean(axis=0)
    b_std = bg_lab.reshape(-1, 3).std(axis=0) + 1e-5

    # damped transfer; clamp std ratio so tiny uniform objects don't explode
    ratio = np.clip(b_std / o_std, 0.5, 2.0)
    target = (obj_lab - o_mean) * ratio + b_mean
    out = obj_lab * (1 - strength) + target * strength

    # luminance matters most for sitting "in" the scene; chroma gets half dose
    out[..., 1:] = obj_lab[..., 1:] * (1 - strength * 0.5) + target[..., 1:] * (strength * 0.5)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR).astype(np.float32)


def pick_blend(rng: random.Random, weights: dict[str, float]) -> str:
    modes = list(weights.keys())
    return rng.choices(modes, weights=[weights[m] for m in modes], k=1)[0]


def feather_alpha(alpha: np.ndarray, radius_px: float) -> np.ndarray:
    if radius_px <= 0.05:
        return alpha
    k = max(3, int(radius_px * 2) * 2 + 1)
    return cv2.GaussianBlur(alpha, (k, k), radius_px)


def composite(
    canvas: np.ndarray,
    obj_bgr: np.ndarray,
    alpha: np.ndarray,
    x: int,
    y: int,
    mode: str,
    rng: random.Random,
    feather_px: tuple[float, float],
    feather_wide_px: tuple[float, float],
) -> np.ndarray | None:
    """Paste obj onto canvas (float32) at top-left (x, y). Returns the alpha
    actually used (feathered), clipped to the canvas, or None if fully off-frame.
    Canvas is modified in place.
    """
    H, W = canvas.shape[:2]
    h, w = obj_bgr.shape[:2]

    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(W, x + w), min(H, y + h)
    if dx2 <= dx1 or dy2 <= dy1:
        return None
    sx1, sy1 = dx1 - x, dy1 - y
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)

    fg = obj_bgr[sy1:sy2, sx1:sx2]
    a = alpha[sy1:sy2, sx1:sx2]

    if mode == "feather":
        a = feather_alpha(a, rng.uniform(*feather_px))
    elif mode == "feather_wide":
        a = feather_alpha(a, rng.uniform(*feather_wide_px))
    elif mode == "seamless":
        # cv2.seamlessClone on small patches; fall back to feather on failure
        try:
            mask = (a > 0.35).astype(np.uint8) * 255
            if mask.sum() > 0:
                bh, bw = fg.shape[:2]
                center = (dx1 + bw // 2, dy1 + bh // 2)
                src = np.clip(fg, 0, 255).astype(np.uint8)
                dst = np.clip(canvas, 0, 255).astype(np.uint8)
                out = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
                canvas[:] = out.astype(np.float32)
                full = np.zeros((H, W), np.float32)
                full[dy1:dy2, dx1:dx2] = a
                return full
        except cv2.error:
            a = feather_alpha(a, rng.uniform(*feather_px))
    # mode == "none": hard edge as-is

    roi = canvas[dy1:dy2, dx1:dx2]
    a3 = a[..., None]
    roi[:] = fg * a3 + roi * (1 - a3)

    full = np.zeros((H, W), np.float32)
    full[dy1:dy2, dx1:dx2] = a
    return full
