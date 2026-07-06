"""Camera/ISP simulation.

Order mirrors a real pipeline (Brooks et al. 2019: noise lives in the linear
domain, before tone mapping): motion blur -> defocus -> linearize -> exposure
-> WB -> shot+read noise -> re-gamma -> (bayer round-trip) -> chromatic
aberration -> vignette -> unsharp -> JPEG last (encoder param returned).
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np


def _motion_kernel(length_px: float, angle_deg: float) -> np.ndarray | None:
    L = int(round(length_px))
    if L < 2:
        return None
    k = np.zeros((L, L), np.float32)
    c = (L - 1) / 2
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for t in np.linspace(-c, c, L * 2):
        x, y = int(round(c + dx * t)), int(round(c + dy * t))
        if 0 <= x < L and 0 <= y < L:
            k[y, x] = 1
    s = k.sum()
    return k / s if s > 0 else None


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255.0


def _bayer_roundtrip(img_u8: np.ndarray) -> np.ndarray:
    """Mosaic to RGGB then demosaic — introduces real debayer color fringing."""
    h, w = img_u8.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    img = img_u8[:h2, :w2]
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    mosaic = np.empty((h2, w2), np.uint8)
    mosaic[0::2, 0::2] = r[0::2, 0::2]
    mosaic[0::2, 1::2] = g[0::2, 1::2]
    mosaic[1::2, 0::2] = g[1::2, 0::2]
    mosaic[1::2, 1::2] = b[1::2, 1::2]
    out = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR)
    if (h2, w2) != (h, w):
        out = cv2.copyMakeBorder(out, 0, h - h2, 0, w - w2, cv2.BORDER_REPLICATE)
    return out


def _chromatic_aberration(img: np.ndarray, shift_px: float) -> np.ndarray:
    if shift_px < 0.2:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = (xx - cx) / cx, (yy - cy) / cy
    out = img.copy()
    for ch, s in ((2, shift_px), (0, -shift_px)):  # r out, b in
        mx = xx + rx * s
        my = yy + ry * s
        out[..., ch] = cv2.remap(img[..., ch], mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return out


def _vignette(img: np.ndarray, strength: float) -> np.ndarray:
    if strength < 0.01:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    return img * (1.0 - strength * r2 * 0.5)[..., None]


def apply_sensor(
    img: np.ndarray,
    rng: random.Random,
    tier: dict,
    wb_shift: tuple[float, float],
    exposure_ev: tuple[float, float],
    unsharp_amount: tuple[float, float],
) -> tuple[np.ndarray, int]:
    """img float32 BGR 0..255 -> (uint8 BGR, jpeg_quality)."""
    npr = np.random.default_rng(rng.getrandbits(32))

    # optics first
    mb = rng.uniform(*tier["motion_blur_px"])
    k = _motion_kernel(mb, rng.uniform(0, 180))
    if k is not None:
        img = cv2.filter2D(img, -1, k)
    df = rng.uniform(*tier["defocus_px"])
    if df > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), df)

    # linear domain: exposure, WB, heteroscedastic noise
    lin = srgb_to_linear(img)
    lin *= 2.0 ** rng.uniform(*exposure_ev)
    gains = np.array([rng.uniform(*wb_shift) for _ in range(3)], np.float32)
    lin *= gains[None, None, :]

    sigma = rng.uniform(*tier["noise_lum_sigma"])
    if sigma > 1e-4:
        # shot noise ~ sqrt(signal), plus read-noise floor
        noise_sd = sigma * np.sqrt(np.clip(lin, 0, 1)) + sigma * 0.35
        lin = lin + npr.normal(0, 1, lin.shape).astype(np.float32) * noise_sd

    img = linear_to_srgb(lin)
    u8 = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < tier.get("bayer_roundtrip_prob", 0.0):
        u8 = _bayer_roundtrip(u8)

    ca = rng.uniform(*tier.get("chroma_aberration_px", (0.0, 0.0)))
    u8 = _chromatic_aberration(u8, ca)

    f = _vignette(u8.astype(np.float32), rng.uniform(*tier["vignette"]))

    amt = rng.uniform(*unsharp_amount)
    if amt > 0.05:
        blur = cv2.GaussianBlur(f, (0, 0), 1.2)
        f = f + (f - blur) * amt

    out = np.clip(f, 0, 255).astype(np.uint8)
    q = rng.randint(int(tier["jpeg_q"][0]), int(tier["jpeg_q"][1]))
    return out, q
