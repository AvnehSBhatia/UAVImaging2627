"""
Object augmentation module for applying random transforms to ODLC objects.

All augmentation functions guarantee that the returned image and alpha
have identical spatial dimensions (height, width).
"""

import cv2
import numpy as np
import random
from typing import Tuple, Optional
from .config import Config
from .image_utils import (
    adjust_brightness_contrast,
    apply_gaussian_blur,
)
import logging


class ObjectAugmentor:
    def __init__(self, config: Config):
        self.config = config
        self.aug_config = config.augmentation
        self.rng = random.Random(config.project.random_seed)

    def augment_object(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
        class_name: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to an object and its alpha mask.

        All transforms are applied consistently to both image and alpha,
        and the output dimensions always match.

        Args:
            image: Input image (H, W, 3) BGR.
            alpha: Alpha mask (H, W) uint8.
            class_name: Class name (for logging).

        Returns:
            Tuple of (augmented_image, augmented_alpha) with matching dimensions.
        """
        result = image.copy()
        result_alpha = alpha.copy()

        # Apply augmentations in order
        result, result_alpha = self._apply_rotation(result, result_alpha)
        result, result_alpha = self._apply_scale(result, result_alpha)
        result, result_alpha = self._apply_brightness_contrast(result, result_alpha)
        result, result_alpha = self._apply_blur(result, result_alpha)
        result, result_alpha = self._apply_perspective(result, result_alpha)

        # Final sanity check
        if result.shape[:2] != result_alpha.shape[:2]:
            logging.warning(
                f"Dimension mismatch after augmentation: "
                f"image {result.shape[:2]} vs alpha {result_alpha.shape[:2]}. "
                f"Resizing alpha to match."
            )
            result_alpha = cv2.resize(
                result_alpha,
                (result.shape[1], result.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        return result, result_alpha

    def _apply_rotation(
        self,
        image: np.ndarray,
        alpha: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random rotation to both image and alpha.

        Uses a large enough canvas to avoid clipping the rotated object,
        then crops back to the bounding box of visible content.
        """
        if not self.aug_config.rotation_degrees:
            return image, alpha

        angle = self.rng.uniform(
            self.aug_config.rotation_degrees[0],
            self.aug_config.rotation_degrees[1]
        )

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Compute the bounding box of the rotated image
        # to avoid clipping corners
        angle_rad = np.deg2rad(abs(angle))
        new_w = int(w * abs(np.cos(angle_rad)) + h * abs(np.sin(angle_rad)))
        new_h = int(w * abs(np.sin(angle_rad)) + h * abs(np.cos(angle_rad)))

        # Rotation matrix with adjusted center for the new canvas
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Adjust translation to center in the new canvas
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0

        # Apply the SAME transform to both image and alpha
        result = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        alpha_rot = cv2.warpAffine(
            alpha, M, (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return result, alpha_rot

    def _apply_scale(
        self,
        image: np.ndarray,
        alpha: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply small random scaling jitter to both image and alpha."""
        if not self.aug_config.scale_jitter:
            return image, alpha

        scale = self.rng.uniform(
            self.aug_config.scale_jitter[0],
            self.aug_config.scale_jitter[1]
        )

        h, w = image.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        alpha_scaled = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return result, alpha_scaled

    def _apply_brightness_contrast(
        self,
        image: np.ndarray,
        alpha: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply brightness/contrast to image only (alpha unchanged)."""
        brightness = self.rng.uniform(
            self.aug_config.brightness[0],
            self.aug_config.brightness[1]
        )
        contrast = self.rng.uniform(
            self.aug_config.contrast[0],
            self.aug_config.contrast[1]
        )

        result = adjust_brightness_contrast(image, brightness, contrast)
        return result, alpha

    def _apply_blur(
        self,
        image: np.ndarray,
        alpha: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random Gaussian blur to image only (alpha unchanged)."""
        if self.rng.random() < self.aug_config.blur_probability:
            ksize = self.rng.choice([3, 5, 7])
            ksize = (ksize, ksize)
            sigma = self.rng.uniform(0.5, 2.0)
            result = apply_gaussian_blur(image, ksize, sigma)
            return result, alpha
        return image, alpha

    def _apply_perspective(
        self,
        image: np.ndarray,
        alpha: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a random perspective transform to both image and alpha.

        Keeps the output canvas the same size as the input to avoid
        dimension changes.
        """
        if self.rng.random() >= self.aug_config.perspective_probability:
            return image, alpha

        h, w = image.shape[:2]

        # Small perturbation for realistic UAV perspective
        max_offset = int(min(w, h) * 0.1)

        # Source points (corners of the image)
        src_pts = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])

        # Destination points (slightly perturbed)
        dst_pts = np.float32([
            [self.rng.randint(0, max_offset), self.rng.randint(0, max_offset)],
            [w - self.rng.randint(0, max_offset), self.rng.randint(0, max_offset)],
            [w - self.rng.randint(0, max_offset), h - self.rng.randint(0, max_offset)],
            [self.rng.randint(0, max_offset), h - self.rng.randint(0, max_offset)],
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply the SAME transform to both image and alpha
        result = cv2.warpPerspective(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        alpha_warp = cv2.warpPerspective(
            alpha, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return result, alpha_warp