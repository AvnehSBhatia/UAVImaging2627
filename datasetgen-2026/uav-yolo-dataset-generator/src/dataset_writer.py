import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import shutil
from tqdm import tqdm
import logging

from .config import Config
from .yolo_labels import YOLOLabelWriter

class DatasetWriter:
    def __init__(self, config: Config):
        self.config = config
        self.label_writer = YOLOLabelWriter(config.classes)
        self.rng = random.Random(config.project.random_seed)
        self._preview_count = 0  # Track total previews saved across all splits

        # Create output directories
        self.output_dirs = {
            'train': Path(config.output.dataset_dir) / 'images' / 'train',
            'val': Path(config.output.dataset_dir) / 'images' / 'val',
            'test': Path(config.output.dataset_dir) / 'images' / 'test',
            'train_labels': Path(config.output.dataset_dir) / 'labels' / 'train',
            'val_labels': Path(config.output.dataset_dir) / 'labels' / 'val',
            'test_labels': Path(config.output.dataset_dir) / 'labels' / 'test',
            'preview': Path(config.output.preview_dir),
            'logs': Path(config.output.logs_dir)
        }

        # Create all directories
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        log_file = self.output_dirs['logs'] / 'dataset_generation.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def write_single_image(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, float, float, float, float]],
        split: str,
        index: int,
    ) -> None:
        """
        Write a single image and its label file to disk immediately.

        This is the memory-efficient alternative to write_dataset().
        Each image is written as soon as it is generated, so the caller
        never needs to hold more than one image in memory at a time.

        Args:
            image: The composited image (H, W, 3) BGR.
            bboxes: List of YOLO-format bounding boxes.
            split: Dataset split ('train', 'val', or 'test').
            index: Sequential index within the split (used for filenames).
        """
        if split not in ('train', 'val', 'test'):
            raise ValueError("Split must be 'train', 'val', or 'test'")

        image_dir = self.output_dirs[split]
        label_dir = self.output_dirs[f"{split}_labels"]

        filename = f"{split}_{index:06d}.jpg"
        image_path = image_dir / filename
        label_path = label_dir / f"{Path(filename).stem}.txt"

        # Write image to disk
        cv2.imwrite(str(image_path), image)

        # Write label file to disk
        self.label_writer.write_label_file(
            label_path, bboxes, image.shape[1], image.shape[0]
        )

        # Save preview (only the first N across all splits)
        if self._preview_count < self.config.dataset.save_preview_count:
            preview_path = self.output_dirs['preview'] / f"preview_{filename}"
            self._save_preview_image(image, bboxes, preview_path)
            self._preview_count += 1

    def write_dataset(
        self,
        images: List[Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]],
        split: str = 'train'
    ) -> None:
        """
        Write a batch of images and labels to the dataset.

        Args:
            images: List of (image, bboxes) tuples
            split: Dataset split ('train', 'val', or 'test')
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        image_dir = self.output_dirs[split]
        label_dir = self.output_dirs[f"{split}_labels"]

        for idx, (image, bboxes) in enumerate(tqdm(images, desc=f"Writing {split} set")):
            filename = f"{split}_{idx:06d}.jpg"
            image_path = image_dir / filename
            label_path = label_dir / f"{Path(filename).stem}.txt"

            cv2.imwrite(str(image_path), image)

            self.label_writer.write_label_file(
                label_path, bboxes, image.shape[1], image.shape[0]
            )

            if idx < self.config.dataset.save_preview_count:
                preview_path = self.output_dirs['preview'] / f"preview_{filename}"
                self._save_preview_image(image, bboxes, preview_path)
    
    def _save_preview_image(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, float, float, float, float]],
        output_path: Path
    ) -> None:
        """
        Save a preview image with bounding boxes drawn.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            output_path: Path to save the preview image
        """
        # Draw bounding boxes
        preview = self.label_writer.visualize_bboxes(image, bboxes)
        
        # Resize if too large for preview
        max_preview_size = 800
        h, w = preview.shape[:2]
        if max(h, w) > max_preview_size:
            scale = max_preview_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            preview = cv2.resize(preview, (new_w, new_h))
        
        # Save preview
        cv2.imwrite(str(output_path), preview)
    
    def create_data_yaml(self) -> None:
        """Create the data.yaml file for YOLO training."""
        # Convert paths to absolute
        base_path = Path(self.config.output.dataset_dir).resolve()
        
        train_path = (base_path / 'images' / 'train').as_posix()
        val_path = (base_path / 'images' / 'val').as_posix()
        test_path = (base_path / 'images' / 'test').as_posix() if self.config.dataset.test_ratio > 0 else None
        
        # Create data.yaml
        yaml_path = base_path / 'data.yaml'
        self.label_writer.create_data_yaml(
            yaml_path,
            train_path,
            val_path,
            test_path,
            nc=len(self.config.classes),
            names=self.config.classes
        )
        
        logging.info(f"Created YOLO dataset configuration at: {yaml_path}")
    
    def clear_output_directories(self) -> None:
        """Clear all output directories."""
        for dir_name, dir_path in self.output_dirs.items():
            if dir_path.exists() and dir_path.is_dir():
                # Skip logs directory to preserve logs
                if dir_name == 'logs':
                    continue
                # Remove all files in the directory
                for item in dir_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                logging.info(f"Cleared directory: {dir_path}")
    
    def get_split_counts(self) -> Dict[str, int]:
        """
        Get the number of images for each split.
        
        Returns:
            Dictionary with counts for 'train', 'val', and 'test' splits
        """
        total = self.config.dataset.total_images
        train_count = int(total * self.config.dataset.train_ratio)
        val_count = int(total * self.config.dataset.val_ratio)
        test_count = total - train_count - val_count
        
        return {
            'train': train_count,
            'val': val_count,
            'test': test_count
        }