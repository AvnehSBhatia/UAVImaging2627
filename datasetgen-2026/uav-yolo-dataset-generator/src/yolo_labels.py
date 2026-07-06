from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import os

class YOLOLabelWriter:
    def __init__(self, class_names: List[str]):
        """
        Initialize YOLO label writer.
        
        Args:
            class_names: List of class names in order of class IDs
        """
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    def bbox_to_yolo_format(
        self,
        bbox: Tuple[float, float, float, float, float],
        image_width: int,
        image_height: int
    ) -> str:
        """
        Convert bounding box to YOLO format string.
        
        Args:
            bbox: Tuple of (class_id, x_center, y_center, width, height)
            image_width: Width of the image
            image_height: Height of the image
        
        Returns:
            String in YOLO format: "class_id x_center y_center width height"
        """
        class_id, x_center, y_center, width, height = bbox
        
        # Ensure values are within [0,1] range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        return f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def write_label_file(
        self,
        label_path: Path,
        bboxes: List[Tuple[float, float, float, float, float]],
        image_width: int,
        image_height: int
    ) -> None:
        """
        Write YOLO format label file.
        
        Args:
            label_path: Path to save the label file
            bboxes: List of bounding boxes in (class_id, x_center, y_center, width, height) format
            image_width: Width of the corresponding image
            image_height: Height of the corresponding image
        """
        # Create parent directories if they don't exist
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write each bounding box to the file
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                yolo_line = self.bbox_to_yolo_format(bbox, image_width, image_height)
                f.write(yolo_line + '\n')
    
    def create_data_yaml(
        self,
        output_path: Path,
        train_dir: str,
        val_dir: str,
        test_dir: str = None,
        nc: int = None,
        names: List[str] = None
    ) -> None:
        """
        Create YOLO data.yaml file.
        
        Args:
            output_path: Path to save data.yaml
            train_dir: Path to training images directory
            val_dir: Path to validation images directory
            test_dir: Optional path to test images directory
            nc: Number of classes (if None, uses len(self.class_names))
            names: List of class names (if None, uses self.class_names)
        """
        if nc is None:
            nc = len(self.class_names)
        if names is None:
            names = self.class_names
            
        content = f"""# YOLO dataset configuration
train: {train_dir}
val: {val_dir}
"""
        if test_dir:
            content += f"test: {test_dir}\n"
            
        content += f"""
nc: {nc}  # number of classes
names: {names}  # class names
"""
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(content)
    
    def visualize_bboxes(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[float, float, float, float, float]],
        class_names: List[str] = None,
        colors: List[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes on an image for visualization.
        
        Args:
            image: Input image (BGR)
            bboxes: List of bounding boxes in YOLO format (class_id, x_center, y_center, width, height)
            class_names: List of class names (if None, uses self.class_names)
            colors: List of BGR colors for each class
        
        Returns:
            Image with drawn bounding boxes
        """
        if class_names is None:
            class_names = self.class_names
            
        if colors is None:
            # Generate distinct colors for each class
            np.random.seed(42)
            colors = [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(len(class_names))]
        
        img_h, img_w = image.shape[:2]
        result = image.copy()
        
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            
            # Convert from normalized to pixel coordinates
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h
            
            # Calculate top-left and bottom-right coordinates
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # Clip to image boundaries
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            
            # Draw rectangle
            color = colors[int(class_id) % len(colors)]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Add class label
            label = class_names[int(class_id)]
            cv2.putText(
                result, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        return result