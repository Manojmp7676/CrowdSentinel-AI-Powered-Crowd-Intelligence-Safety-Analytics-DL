"""
Convert UCF CC50 point annotations to YOLOv8 bounding box format.

UCF CC50 format: x_coordinate y_coordinate (one person per line)
YOLOv8 format: class_id x_center y_center width height (normalized 0-1)

Each point is converted to a fixed-size bounding box around the person.
"""

import os
import json
import shutil
from pathlib import Path
import yaml


def create_bounding_box(x, y, box_size=50, img_width=224, img_height=224):
    """
    Create a bounding box around a point annotation.

    Args:
        x, y: center point coordinates
        box_size: size of bounding box (default 50px for better detection)
        img_width, img_height: image dimensions

    Returns:
        x_center, y_center, width, height (normalized 0-1)
    """
    half = box_size / 2

    x_min = max(0, x - half)
    y_min = max(0, y - half)
    x_max = min(img_width, x + half)
    y_max = min(img_height, y + half)

    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return x_center, y_center, width, height


def convert_dataset(source_dir, output_dir, splits_path, box_size=20):
    """
    Convert UCF CC50 point annotations to YOLOv8 format.

    Args:
        source_dir: path to ucf_cc50 directory
        output_dir: path to output YOLO dataset
        splits_path: path to splits.json
        box_size: bounding box size around each point
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    with open(splits_path, "r") as f:
        splits = json.load(f)

    img_dir = source_dir / "images"
    gt_dir = source_dir / "ground_truth"

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Use train split as train, val split as val, and create a test split from val
    train_files = splits["train"]
    val_files = splits["val"]
    test_files = val_files[:len(val_files)//2]
    val_files = val_files[len(val_files)//2:]

    stats = {"train": 0, "val": 0, "test": 0}

    for split_name, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        print(f"\nProcessing {split_name} split ({len(file_list)} images)...")

        for filename in file_list:
            # Copy image
            src_img = img_dir / f"{filename}.jpg"
            dst_img = output_dir / "images" / split_name / f"{filename}.jpg"

            if src_img.exists():
                shutil.copy2(src_img, dst_img)

            # Convert annotations
            gt_file = gt_dir / f"{filename}.txt"
            label_file = output_dir / "labels" / split_name / f"{filename}.txt"

            if gt_file.exists():
                # Get image size from dataset_info.json
                with open(source_dir / "dataset_info.json", "r") as f:
                    dataset_info = json.load(f)

                img_info = next((item for item in dataset_info if item["filename"] == filename), None)
                if img_info:
                    img_w, img_h = img_info["original_size"]
                else:
                    img_w, img_h = 224, 224

                with open(gt_file, "r") as f:
                    lines = f.readlines()

                yolo_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        x_center, y_center, width, height = create_bounding_box(
                            x, y, box_size, img_w, img_h
                        )
                        # Class 0 = person
                        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines))

                stats[split_name] += 1
                print(f"  {filename}: {len(yolo_lines)} persons")

    # Create dataset YAML
    dataset_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["person"],
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\n{'='*50}")
    print("Dataset conversion complete!")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"\nStatistics:")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  Test:  {stats['test']} images")

    return yaml_path


if __name__ == "__main__":
    source_dir = "data/ucf_cc50"
    output_dir = "data/yolo_ucf_cc50_v2"
    splits_path = "data/ucf_cc50/splits.json"

    convert_dataset(source_dir, output_dir, splits_path, box_size=50)
