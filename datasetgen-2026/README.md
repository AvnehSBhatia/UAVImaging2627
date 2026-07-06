# UAV/ODLC YOLO Dataset Generator

A Python tool for generating synthetic training data for YOLO object detection models, specifically designed for UAV-based ODLC (Object Detection, Localization, and Classification) tasks.

## Features

- Generate synthetic images with ODLC objects (mannequins and tents) on runway backgrounds
- Support for multiple object orientations and variations
- Configurable object placement and augmentation
- YOLO-format label generation
- Train/validation/test split support

## Requirements

- Python 3.10 or 3.11
- See [requirements.txt](requirements.txt) for Python dependencies

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip3 install -r requirements.txt
```

## Directory Structure

```
uav-yolo-dataset-generator/
├── README.md
├── requirements.txt
├── config.yaml
├── generate_dataset.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── image_utils.py
│   ├── augment.py
│   ├── placement.py
│   ├── yolo_labels.py
│   └── dataset_writer.py
├── input/
│   ├── backgrounds/
│   │   └── runway/          # Place runway background images here
│   ├── objects/
│   │   ├── mannequin/       # Place mannequin images here
│   │   └── tent/            # Place tent images here
│   ├── masks/               # Optional masks for objects
│   └── examples/            # Example images
├── output/
│   ├── generated_preview/   # Preview of generated images
│   └── logs/                # Log files
└── dataset/                 # Generated dataset
    ├── images/              # Generated images
    └── labels/              # YOLO format labels
```

## Configuration

Edit `config.yaml` to customize the dataset generation:

- Adjust dataset size and split ratios
- Configure object placement parameters
- Modify augmentation settings
- Set output directories

## Usage

1. Place your background images in `input/backgrounds/runway/`
2. Place mannequin images directly in `input/objects/mannequin/`
3. Place tent images in `input/objects/tent/`
4. (Optional) Add masks for objects in `input/masks/`
5. Run the generator:

```bash
python3 generate_dataset.py
```

### Command-Line Options

The generator supports several command-line options to customize the generation process:

- `--total-images`: Specify the total number of images to generate (overrides the value in `config.yaml`).
- `--clear`: Clear the output directories before generating new data.

Examples:

```bash
# Generate the default number of images specified in config.yaml
python3 generate_dataset.py

# Generate exactly 500 images
python3 generate_dataset.py --total-images 500

# Clear existing dataset and generate 1000 new images
python3 generate_dataset.py --clear --total-images 1000
```

### Managing Generated Data

To clear all generated dataset files (images, labels, and previews) while preserving your input data and configuration:

```bash
python3 clear_dataset.py
```

For automated scripts, you can skip the confirmation prompt:

```bash
python3 clear_dataset.py --confirm
```

The generated dataset will be saved in the `dataset/` directory with the following structure:

```
dataset/
├── images/
│   ├── train/      # Training images
│   ├── val/        # Validation images
│   └── test/       # Test images
└── labels/
    ├── train/      # Training labels (YOLO format)
    ├── val/        # Validation labels
    └── test/       # Test labels
```

## Output Format

Each generated image has a corresponding `.txt` file with the same name in the labels directory. The label format is:

```
class_id x_center y_center width height
```

Where:
- `class_id`: 0 for mannequin, 1 for tent
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized dimensions (0-1)

## Customization

To add new object classes:

1. Add the class name to the `classes` list in `config.yaml`
2. Create a corresponding directory in `input/objects/`
3. Update the `class_probability` section in `config.yaml`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.