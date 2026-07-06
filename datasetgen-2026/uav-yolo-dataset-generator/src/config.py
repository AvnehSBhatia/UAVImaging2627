"""
Configuration module for the UAV/ODLC YOLO dataset generator.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging


@dataclass
class ProjectConfig:
    random_seed: int


@dataclass
class InputConfig:
    runway_background_dir: str
    mannequin_dir: str
    tent_dir: str
    mannequin_mask_dir: str
    tent_mask_dir: str


@dataclass
class OutputConfig:
    dataset_dir: str
    preview_dir: str
    logs_dir: str


@dataclass
class DatasetConfig:
    total_images: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    image_width: int
    image_height: int
    save_preview_count: int


@dataclass
class ObjectsConfig:
    min_objects_per_image: int
    max_objects_per_image: int
    class_probability: Dict[str, float]
    min_object_width_ratio: float
    max_object_width_ratio: float
    max_overlap_iou: float
    # Class-specific width ratios (optional, falls back to min/max above)
    mannequin_width_ratio: Optional[Tuple[float, float]] = None
    tent_width_ratio: Optional[Tuple[float, float]] = None
    # Minimum visible bounding box area in pixels
    min_visible_bbox_area_px: int = 64


@dataclass
class AugmentationConfig:
    rotation_degrees: Tuple[float, float] = (-180, 180)
    scale_jitter: Tuple[float, float] = (0.8, 1.2)
    brightness: Tuple[float, float] = (0.75, 1.25)
    contrast: Tuple[float, float] = (0.8, 1.2)
    blur_probability: float = 0.15
    shadow_probability: float = 0.35
    perspective_probability: float = 0.2


@dataclass
class Config:
    project: ProjectConfig
    input: InputConfig
    output: OutputConfig
    dataset: DatasetConfig
    classes: List[str]
    objects: ObjectsConfig
    augmentation: AugmentationConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Parse objects config with optional fields
        obj_data = config_data['objects']
        mannequin_width_ratio = None
        if 'mannequin_width_ratio' in obj_data:
            mannequin_width_ratio = tuple(obj_data['mannequin_width_ratio'])
        tent_width_ratio = None
        if 'tent_width_ratio' in obj_data:
            tent_width_ratio = tuple(obj_data['tent_width_ratio'])

        return cls(
            project=ProjectConfig(
                random_seed=config_data['project']['random_seed']
            ),
            input=InputConfig(
                runway_background_dir=config_data['input']['runway_background_dir'],
                mannequin_dir=config_data['input']['mannequin_dir'],
                tent_dir=config_data['input']['tent_dir'],
                mannequin_mask_dir=config_data['input']['mannequin_mask_dir'],
                tent_mask_dir=config_data['input']['tent_mask_dir']
            ),
            output=OutputConfig(
                dataset_dir=config_data['output']['dataset_dir'],
                preview_dir=config_data['output']['preview_dir'],
                logs_dir=config_data['output']['logs_dir']
            ),
            dataset=DatasetConfig(
                total_images=config_data['dataset']['total_images'],
                train_ratio=config_data['dataset']['train_ratio'],
                val_ratio=config_data['dataset']['val_ratio'],
                test_ratio=config_data['dataset']['test_ratio'],
                image_width=config_data['dataset']['image_width'],
                image_height=config_data['dataset']['image_height'],
                save_preview_count=config_data['dataset']['save_preview_count']
            ),
            classes=config_data['classes'],
            objects=ObjectsConfig(
                min_objects_per_image=obj_data['min_objects_per_image'],
                max_objects_per_image=obj_data['max_objects_per_image'],
                class_probability=obj_data['class_probability'],
                min_object_width_ratio=obj_data['min_object_width_ratio'],
                max_object_width_ratio=obj_data['max_object_width_ratio'],
                max_overlap_iou=obj_data['max_overlap_iou'],
                mannequin_width_ratio=mannequin_width_ratio,
                tent_width_ratio=tent_width_ratio,
                min_visible_bbox_area_px=obj_data.get('min_visible_bbox_area_px', 64)
            ),
            augmentation=AugmentationConfig(
                rotation_degrees=tuple(config_data['augmentation']['rotation_degrees']),
                scale_jitter=tuple(config_data['augmentation']['scale_jitter']),
                brightness=tuple(config_data['augmentation']['brightness']),
                contrast=tuple(config_data['augmentation']['contrast']),
                blur_probability=config_data['augmentation']['blur_probability'],
                shadow_probability=config_data['augmentation']['shadow_probability'],
                perspective_probability=config_data['augmentation']['perspective_probability']
            )
        )

    def validate(self) -> bool:
        """Validate the configuration values."""
        # Check ratios sum to 1.0 (within floating point tolerance)
        ratios_sum = (
            self.dataset.train_ratio +
            self.dataset.val_ratio +
            self.dataset.test_ratio
        )
        if not 0.999 <= ratios_sum <= 1.001:
            logging.error("Dataset ratios must sum to 1.0")
            return False

        # Check class probabilities sum to 1.0
        class_probs_sum = sum(self.objects.class_probability.values())
        if not 0.999 <= class_probs_sum <= 1.001:
            logging.error("Class probabilities must sum to 1.0")
            return False

        # Check object counts
        if self.objects.min_objects_per_image > self.objects.max_objects_per_image:
            logging.error("min_objects_per_image cannot be greater than max_objects_per_image")
            return False

        # Check image dimensions
        if self.dataset.image_width <= 0 or self.dataset.image_height <= 0:
            logging.error("Image dimensions must be positive")
            return False

        return True