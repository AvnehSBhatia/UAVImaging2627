#!/usr/bin/env python3
"""
Script to clear all generated dataset files while preserving input data and configuration.
"""

import shutil
from pathlib import Path
import argparse
import logging

def clear_dataset():
    """Clear all generated dataset files."""
    # Define directories to clear
    dirs_to_clear = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "output/generated_preview"
    ]
    
    # Keep these directories (don't clear logs)
    dirs_to_keep = [
        "output/logs"
    ]
    
    # Clear each directory
    for dir_path in dirs_to_clear:
        dir_path = Path(dir_path)
        if dir_path.exists() and dir_path.is_dir():
            # Remove all files in the directory
            for item in dir_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print(f"Cleared directory: {dir_path}")
    
    print("Dataset cleared successfully!")

def main():
    parser = argparse.ArgumentParser(description="Clear generated dataset files")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    args = parser.parse_args()
    
    if not args.confirm:
        response = input("Are you sure you want to clear all generated dataset files? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    clear_dataset()

if __name__ == "__main__":
    main()