"""
Object placement logic for compositing ODLC objects onto runway backgrounds.
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from .config import Config
from .image_utils import apply_alpha_compositing, compute_tight_bbox
import logging


class ObjectPlacer:
    def __init__(self, config: Config):
        self.config = config
        self.obj_config = config.objects
        self.rng = random.Random(config.project.random_seed)

    def place_objects(
        self,
        background: np.ndarray,
        objects: List[Tuple[np.ndarray, np.ndarray, int]]
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """
        Place multiple objects on the background image.

        For each object:
          1. Determine a target size based on class-specific width ratios.
          2. Resize the object image and alpha to that target size.
          3. Find a valid position that doesn't overlap too much.
          4. Composite onto the background.
          5. Compute a tight bounding box from the visible alpha pixels.

        Args:
            background: Background image (H, W, 3) BGR.
            objects: List of (object_image, alpha_mask, class_id) tuples.
                     object_image: (h, w, 3) BGR
                     alpha_mask: (h, w) uint8

        Returns:
            Tuple of (composited_image, list of YOLO bboxes).
            Each YOLO bbox is (class_id, x_center, y_center, width, height)
            with normalized coordinates [0, 1].
        """
        result = background.copy()
        yolo_bboxes: List[Tuple[int, float, float, float, float]] = []
        placed_bboxes: List[Tuple[int, int, int, int]] = []

        bg_h, bg_w = result.shape[:2]

        for obj_img, obj_alpha, class_id in objects:
            # --- Step 1: Determine target size based on class ---
            class_name = self.config.classes[class_id]
            target_w, target_h = self._compute_target_size(
                obj_img.shape[1], obj_img.shape[0], class_id, bg_w, bg_h
            )

            if target_w is None or target_h is None:
                logging.warning(
                    f"Could not determine target size for {class_name} object. Skipping."
                )
                continue

            # --- Step 2: Resize object and alpha to target size ---
            obj_resized = cv2.resize(
                obj_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            alpha_resized = cv2.resize(
                obj_alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

            # --- Step 3: Find valid position ---
            pos = self._find_valid_position(
                bg_w, bg_h, target_w, target_h, placed_bboxes
            )

            if pos is None:
                logging.debug(
                    f"No valid position found for {class_name} object "
                    f"({target_w}x{target_h}). Skipping."
                )
                continue

            px, py = pos  # top-left corner on background

            # --- Step 4: Composite onto background ---
            result = apply_alpha_compositing(result, obj_resized, alpha_resized, px, py)

            # --- Step 5: Compute tight bounding box from visible (clipped) alpha ---
            # Determine the visible portion of the alpha mask after clipping
            # to background boundaries (same logic as apply_alpha_compositing)
            src_x1 = max(0, -px)
            src_y1 = max(0, -py)
            src_x2 = min(target_w, bg_w - px)
            src_y2 = min(target_h, bg_h - py)

            if src_x2 <= src_x1 or src_y2 <= src_y1:
                logging.debug(f"Object {class_name} fully outside background. Skipping bbox.")
                continue

            visible_alpha = alpha_resized[src_y1:src_y2, src_x1:src_x2]
            tight = compute_tight_bbox(visible_alpha, threshold=1)
            if tight is None:
                logging.debug(f"Object {class_name} has no visible pixels. Skipping bbox.")
                continue

            tx, ty, tw, th = tight
            # Convert to background coordinates (offset by the clip region start + placement pos)
            abs_x = px + src_x1 + tx
            abs_y = py + src_y1 + ty

            if tw <= 0 or th <= 0:
                continue

            # Check minimum visible area
            if tw * th < self.obj_config.min_visible_bbox_area_px:
                logging.debug(
                    f"Object {class_name} visible area {tw}x{th}={tw*th}px "
                    f"is below minimum {self.obj_config.min_visible_bbox_area_px}px. Skipping."
                )
                continue

            # Convert to YOLO format (normalized)
            x_center = (abs_x + tw / 2.0) / bg_w
            y_center = (abs_y + th / 2.0) / bg_h
            norm_w = tw / bg_w
            norm_h = th / bg_h

            yolo_bboxes.append((class_id, x_center, y_center, norm_w, norm_h))
            placed_bboxes.append((abs_x, abs_y, tw, th))

        return result, yolo_bboxes

    def _compute_target_size(
        self,
        src_w: int,
        src_h: int,
        class_id: int,
        bg_w: int,
        bg_h: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Compute the target size for an object based on class-specific width ratios.

        Args:
            src_w: Source object width.
            src_h: Source object height.
            class_id: Class ID.
            bg_w: Background image width.
            bg_h: Background image height.

        Returns:
            Tuple of (target_w, target_h) or (None, None) if invalid.
        """
        class_name = self.config.classes[class_id]
        width_ratio_key = f"{class_name}_width_ratio"

        if hasattr(self.obj_config, width_ratio_key):
            min_ratio, max_ratio = getattr(self.obj_config, width_ratio_key)
        else:
            min_ratio = self.obj_config.min_object_width_ratio
            max_ratio = self.obj_config.max_object_width_ratio

        # Random target width within the class-specific range
        target_w = int(self.rng.uniform(min_ratio, max_ratio) * bg_w)

        # Maintain aspect ratio
        aspect = src_h / src_w if src_w > 0 else 1.0
        target_h = max(1, int(target_w * aspect))

        # Clamp to background dimensions
        target_w = min(target_w, bg_w)
        target_h = min(target_h, bg_h)

        if target_w < 2 or target_h < 2:
            return None, None

        return target_w, target_h

    def _find_valid_position(
        self,
        bg_w: int,
        bg_h: int,
        obj_w: int,
        obj_h: int,
        placed_bboxes: List[Tuple[int, int, int, int]],
        max_attempts: int = 100
    ) -> Optional[Tuple[int, int]]:
        """
        Find a valid (x, y) position for an object without excessive overlap.

        Args:
            bg_w: Background width.
            bg_h: Background height.
            obj_w: Object width.
            obj_h: Object height.
            placed_bboxes: Already placed bboxes as (x, y, w, h).
            max_attempts: Maximum attempts.

        Returns:
            (x, y) top-left corner or None.
        """
        if obj_w > bg_w or obj_h > bg_h:
            return None

        for _ in range(max_attempts):
            x = self.rng.randint(0, bg_w - obj_w)
            y = self.rng.randint(0, bg_h - obj_h)

            candidate = (x, y, obj_w, obj_h)

            valid = True
            for existing in placed_bboxes:
                iou = self._calculate_iou(candidate, existing)
                if iou > self.obj_config.max_overlap_iou:
                    valid = False
                    break

            if valid:
                return (x, y)

        return None

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU between two bboxes (x, y, w, h)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_random_object_count(self) -> int:
        """Get a random number of objects to place."""
        return self.rng.randint(
            self.obj_config.min_objects_per_image,
            self.obj_config.max_objects_per_image + 1
        )

    def get_random_class(self) -> str:
        """Get a random class based on class probabilities."""
        classes = list(self.obj_config.class_probability.keys())
        probs = list(self.obj_config.class_probability.values())
        return self.rng.choices(classes, weights=probs, k=1)[0]