"""
Image utility functions for loading, transforming, and compositing images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging


def load_image_with_alpha(
    image_path: Path,
    mask_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image and its alpha channel.
    
    Priority:
      1. Alpha channel embedded in the image (e.g. PNG with transparency)
      2. External mask file
      3. Full-opaque mask (all 255)
    
    Args:
        image_path: Path to the image file.
        mask_path: Optional path to an external mask file.
    
    Returns:
        Tuple of (image_bgr, alpha_uint8) where alpha is 0-255.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    alpha = None

    if image.ndim == 2:
        # Grayscale → BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # BGRA → split
        alpha = image[:, :, 3].copy()
        image = image[:, :, :3].copy()
    elif image.shape[2] == 2:
        # GA → split
        alpha = image[:, :, 1].copy()
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        image = image[:, :, :3].copy()

    # External mask fallback
    if alpha is None and mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Resize mask to match image if needed
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask, (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            alpha = mask

    # Full-opaque fallback
    if alpha is None:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)

    return image, alpha


def apply_alpha_compositing(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
    x: int,
    y: int
) -> np.ndarray:
    """
    Composite foreground onto background at position (x, y) using alpha.
    
    This function is robust:
    - Clips the paste region to the background boundaries.
    - Crops foreground and alpha to match the clipped region.
    - Validates that all arrays have matching dimensions before blending.
    - If dimensions still don't match, logs a warning and returns the
      background unchanged.
    
    Args:
        background: Background image (H, W, 3) uint8.
        foreground: Foreground image (hf, wf, 3) uint8.
        alpha: Alpha mask (hf, wf) uint8, values 0-255.
        x: X coordinate of top-left corner on background.
        y: Y coordinate of top-left corner on background.
    
    Returns:
        Composited background image.
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    a_h, a_w = alpha.shape[:2]

    # The foreground and alpha must match
    if (fg_h, fg_w) != (a_h, a_w):
        logging.warning(
            f"Shape mismatch: foreground ({fg_h},{fg_w}) vs alpha ({a_h},{a_w}). "
            f"Resizing alpha to match foreground."
        )
        alpha = cv2.resize(alpha, (fg_w, fg_h), interpolation=cv2.INTER_NEAREST)

    # Calculate the valid paste region (clip to background boundaries)
    # Destination region in background (clamped to bg bounds)
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(bg_w, x + fg_w)
    dst_y2 = min(bg_h, y + fg_h)

    # Source region in foreground (derived from destination to guarantee match)
    src_x1 = dst_x1 - x
    src_y1 = dst_y1 - y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    # Check for valid region
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        # Completely outside background
        return background

    # Crop to matching regions
    fg_crop = foreground[src_y1:src_y2, src_x1:src_x2]
    alpha_crop = alpha[src_y1:src_y2, src_x1:src_x2]
    bg_roi = background[dst_y1:dst_y2, dst_x1:dst_x2]

    # Final sanity check
    if fg_crop.shape[:2] != bg_roi.shape[:2]:
        logging.error(
            f"Compositing shape mismatch after clipping: "
            f"fg_crop {fg_crop.shape} vs bg_roi {bg_roi.shape}. Skipping object."
        )
        return background

    if fg_crop.shape[:2] != alpha_crop.shape[:2]:
        logging.error(
            f"Compositing shape mismatch: "
            f"fg_crop {fg_crop.shape} vs alpha_crop {alpha_crop.shape}. Skipping object."
        )
        return background

    # Blend using float arithmetic
    alpha_f = alpha_crop.astype(np.float32) / 255.0
    if alpha_f.ndim == 2:
        alpha_f = alpha_f[:, :, np.newaxis]

    blended = (fg_crop.astype(np.float32) * alpha_f +
               bg_roi.astype(np.float32) * (1.0 - alpha_f))

    result = background.copy()
    result[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

    return result


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0
) -> np.ndarray:
    """Adjust brightness and contrast of an image."""
    img = image.astype(np.float32)
    img = img * brightness
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - mean) * contrast + mean
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 0
) -> np.ndarray:
    """Apply Gaussian blur to an image."""
    return cv2.GaussianBlur(image, kernel_size, sigma)


def compute_tight_bbox(alpha: np.ndarray, threshold: int = 1) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute the tight bounding box of non-zero regions in the alpha mask.
    
    Args:
        alpha: Alpha mask (H, W) uint8.
        threshold: Minimum value to consider as visible.
    
    Returns:
        Tuple of (x, y, w, h) or None if the mask is empty.
    """
    if alpha is None:
        return None
    coords = cv2.findNonZero((alpha > threshold).astype(np.uint8))
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, w, h)