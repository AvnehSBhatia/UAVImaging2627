#!/usr/bin/env python3
"""
Main script for generating synthetic UAV/ODLC training data for YOLO object detection.

Memory-efficient design:
- Stores file paths, not image arrays.
- Generates and writes one image at a time.
- Releases intermediate arrays immediately.
"""

import argparse
import gc
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src import (
    Config,
    load_image_with_alpha,
    ObjectAugmentor,
    ObjectPlacer,
    DatasetWriter,
)


def list_image_paths(directory: Path, recursive: bool = False) -> List[Path]:
    """Collect image file paths from a directory without loading them."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    patterns = ["**/*.*"] if recursive else ["*.*"]
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    paths = []
    for pattern in patterns:
        for p in directory.glob(pattern):
            if p.is_file() and p.suffix.lower() in valid_extensions:
                paths.append(p)

    return sorted(paths)


def load_background(config: Config) -> np.ndarray:
    """
    Load a single random background image, resized to configured dimensions.
    """
    bg_path = random.choice(config._background_paths)
    img = cv2.imread(str(bg_path))
    if img is None:
        raise ValueError(f"Failed to load background: {bg_path}")
    if img.shape[1] != config.dataset.image_width or img.shape[0] != config.dataset.image_height:
        img = cv2.resize(
            img,
            (config.dataset.image_width, config.dataset.image_height),
        )
    return img


def load_and_augment_object(
    config: Config,
    class_name: str,
    augmentor: ObjectAugmentor,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load a single random object image for the given class, apply augmentation,
    and return (image, alpha, class_id).
    """
    class_id = config.classes.index(class_name)
    obj_paths: List[Path] = config._object_paths[class_name]
    mask_dir = Path(getattr(config.input, f"{class_name}_mask_dir", ""))

    obj_path = random.choice(obj_paths)

    # Find mask if available
    mask_path = None
    if mask_dir.exists():
        candidate = mask_dir / obj_path.name
        if candidate.exists():
            mask_path = candidate

    img, alpha = load_image_with_alpha(obj_path, mask_path)
    img_aug, alpha_aug = augmentor.augment_object(img, alpha, class_name)

    return img_aug, alpha_aug, class_id


def generate_single_image(
    config: Config,
    placer: ObjectPlacer,
    augmentor: ObjectAugmentor,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """
    Generate one synthetic image with placed objects.
    Loads background and objects on-demand.
    """
    # Load a fresh background
    bg_image = load_background(config)

    # Determine how many objects to place
    num_objects = placer.get_random_object_count()

    # Load and augment each object on-demand
    objects_to_place = []
    for _ in range(num_objects):
        class_name = placer.get_random_class()
        if class_name in config._object_paths and config._object_paths[class_name]:
            obj_img, obj_alpha, class_id = load_and_augment_object(
                config, class_name, augmentor
            )
            objects_to_place.append((obj_img, obj_alpha, class_id))

    # Place objects on background
    result, bboxes = placer.place_objects(bg_image, objects_to_place)

    return result, bboxes


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic UAV/ODLC training data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear output directories before generating new data",
    )
    parser.add_argument(
        "--total-images",
        type=int,
        default=None,
        help="Total number of images to generate (overrides config)",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_yaml(args.config)
        if not config.validate():
            logging.error("Invalid configuration")
            sys.exit(1)

        # Set random seed for reproducibility
        random.seed(config.project.random_seed)
        np.random.seed(config.project.random_seed)

        # Initialize components
        augmentor = ObjectAugmentor(config)
        placer = ObjectPlacer(config)
        dataset_writer = DatasetWriter(config)

        # Clear output directories if requested
        if args.clear:
            dataset_writer.clear_output_directories()

        # --- Store file paths only, not image arrays ---
        logging.info("Scanning background images...")
        bg_dir = Path(config.input.runway_background_dir)
        config._background_paths = list_image_paths(bg_dir)
        if not config._background_paths:
            raise ValueError(f"No background images found in {bg_dir}")
        logging.info(f"  Found {len(config._background_paths)} background images")

        logging.info("Scanning object images...")
        config._object_paths: Dict[str, List[Path]] = {}
        for class_name in config.classes:
            obj_dir = Path(getattr(config.input, f"{class_name}_dir"))
            recursive = class_name == "mannequin"
            paths = list_image_paths(obj_dir, recursive=recursive)
            config._object_paths[class_name] = paths
            logging.info(f"  Found {len(paths)} images for class '{class_name}'")

        # Override total images if specified
        if args.total_images is not None:
            config.dataset.total_images = args.total_images

        # Get split counts
        split_counts = dataset_writer.get_split_counts()

        logging.info(f"Generating {config.dataset.total_images} total images:")
        for split, count in split_counts.items():
            if count > 0:
                logging.info(f"  {split}: {count} images")

        # --- Stream-based generation: one image at a time ---
        logging.info("Generating dataset...")

        for split, count in split_counts.items():
            if count <= 0:
                continue

            logging.info(f"Generating {count} images for {split} set...")

            for i in tqdm(range(count), desc=f"Generating {split}"):
                try:
                    image, bboxes = generate_single_image(config, placer, augmentor)

                    # Write immediately — do NOT accumulate in a list
                    dataset_writer.write_single_image(
                        image, bboxes, split, i
                    )

                    # Explicitly release large arrays
                    del image
                    del bboxes

                except Exception as e:
                    logging.warning(f"Failed to generate image {i} for {split}: {e}")
                    continue

            # Force garbage collection between splits
            gc.collect()

        # Create data.yaml
        dataset_writer.create_data_yaml()

        logging.info("Dataset generation completed successfully!")

    except Exception as e:
        logging.exception("An error occurred during dataset generation")
        sys.exit(1)


if __name__ == "__main__":
    main()