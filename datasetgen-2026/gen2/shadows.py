"""Shadow compositing.

Rendered objects come with a registered shadow pass (same canvas, 255=full
shadow). Occluder cutouts have no shadow pass, so we synthesize one: shear
the silhouette along the shadow direction with length height/tan(elevation),
soften, and multiply-darken.

Convention: sun azimuth = direction light comes FROM, degrees CCW from +X in
image space. Shadows extend toward azimuth + 180.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def apply_shadow_pass(
    canvas: np.ndarray,
    shadow: np.ndarray,
    x: int,
    y: int,
    opacity: float,
) -> None:
    """Multiply-darken canvas by a registered shadow pass (float 0..1)."""
    if opacity <= 0.01:
        return
    H, W = canvas.shape[:2]
    h, w = shadow.shape[:2]
    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(W, x + w), min(H, y + h)
    if dx2 <= dx1 or dy2 <= dy1:
        return
    s = shadow[dy1 - y:dy2 - y, dx1 - x:dx2 - x]
    # soften slightly; render passes can be crisp
    s = cv2.GaussianBlur(s, (5, 5), 1.2)
    factor = 1.0 - opacity * s
    canvas[dy1:dy2, dx1:dx2] *= factor[..., None]


def synth_shadow(
    alpha: np.ndarray,
    px_per_m: float,
    height_m: float,
    sun_az_deg: float,
    sun_el_deg: float,
) -> tuple[np.ndarray, int, int]:
    """Synthesize a soft cast shadow from a cutout silhouette.

    Returns (shadow float 0..1, offset_x, offset_y) where offsets position the
    shadow canvas relative to the cutout's top-left.
    """
    length_px = px_per_m * height_m / max(math.tan(math.radians(sun_el_deg)), 0.15)
    length_px = float(min(length_px, 4.0 * max(alpha.shape)))

    ang = math.radians(sun_az_deg + 180.0)
    dx, dy = math.cos(ang), -math.sin(ang)  # image y is down

    h, w = alpha.shape
    steps = max(2, int(length_px / 3))
    pad = int(length_px) + 4
    sh = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)

    # stack progressively squashed, shifted copies of the silhouette
    sil = (alpha > 0.4).astype(np.float32)
    for i in range(steps):
        t = i / max(steps - 1, 1)
        ox = int(pad + dx * length_px * t)
        oy = int(pad + dy * length_px * t)
        # squash silhouette toward its base as the shadow extends
        scale_y = 1.0 - 0.35 * t
        sw, sh_ = w, max(2, int(h * scale_y))
        s = cv2.resize(sil, (sw, sh_), interpolation=cv2.INTER_AREA)
        y0 = oy + (h - sh_) // 2
        region = sh[y0:y0 + sh_, ox:ox + sw]
        if region.shape == s.shape:
            np.maximum(region, s * (1.0 - 0.55 * t), out=region)

    k = max(3, int(length_px * 0.18) * 2 + 1)
    sh = cv2.GaussianBlur(sh, (min(k, 61),) * 2, 0)
    return np.clip(sh, 0, 1), -pad, -pad
