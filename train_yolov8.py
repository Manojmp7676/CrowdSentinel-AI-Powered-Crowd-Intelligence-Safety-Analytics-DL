"""
Train YOLOv8 on UCF CC50 Crowd Dataset.

This script:
1. Loads a pretrained YOLOv8 model (COCO weights)
2. Fine-tunes it on UCF CC50 person detection dataset
3. Saves the best model based on validation mAP

Usage:
    python train_yolov8.py
"""

import os  # Operating system interface
import json  # JSON file handling
import yaml  # YAML file handling
from pathlib import Path  # Object-oriented paths
from ultralytics import YOLO  # YOLOv8 implementation


def train_yolov8(
    dataset_yaml="data/yolo_ucf_cc50_real/dataset.yaml",  # Path to dataset config
    model_size="yolov8s",  # Model variant (n/s/m/l/x)
    epochs=150,  # Number of training epochs
    img_size=1280,  # Input image size
    batch_size=4,  # Batch size (reduce if OOM)
    learning_rate=0.01,  # Initial learning rate
    patience=25,  # Early stopping patience
    output_dir="models/yolo_ucf_cc50_real",  # Output directory
):
    """
    Train YOLOv8 model on UCF CC50 dataset.
    
    Args:
        dataset_yaml: Path to dataset YAML configuration file
        model_size: YOLOv8 variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Maximum number of training epochs
        img_size: Input image resolution (larger = better detection, slower training)
        batch_size: Number of images per batch
        learning_rate: Initial learning rate for optimizer
        patience: Early stopping patience (stops if no improvement for N epochs)
        output_dir: Directory to save trained models
    
    Returns:
        results: Training results object
    """
    # Print training configuration header
    print("=" * 60)
    print("YOLOv8 Training on UCF CC50 Dataset")
    print("=" * 60)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the pretrained YOLOv8 model
    # This loads weights from COCO dataset (pretrained on 80 classes)
    model_name = f"{model_size}.pt"
    print(f"\nLoading pretrained {model_size} model...")
    model = YOLO(model_name)

    # Display training configuration
    print(f"\nTraining Configuration:")
    print(f"  Dataset:      {dataset_yaml}")
    print(f"  Model:        {model_size} (pretrained on COCO)")
    print(f"  Epochs:       {epochs}")
    print(f"  Image Size:   {img_size}")
    print(f"  Batch Size:   {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Patience:     {patience}")
    print(f"  Output:       {output_path}")

    # Print training start message
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    # Train the model
    # This will:
    # 1. Load the dataset from dataset_yaml
    # 2. Fine-tune the pretrained model on person detection
    # 3. Save checkpoints during training
    # 4. Evaluate on validation set after each epoch
    results = model.train(
        data=str(Path(dataset_yaml).absolute()),  # Absolute path to dataset YAML
        epochs=epochs,  # Number of training epochs
        imgsz=img_size,  # Input image size
        batch=batch_size,  # Batch size
        lr0=learning_rate,  # Initial learning rate
        patience=patience,  # Early stopping patience
        project=str(output_path),  # Project directory
        name="train",  # Experiment name
        exist_ok=True,  # Overwrite existing experiment
        pretrained=True,  # Use pretrained weights
        optimizer="auto",  # Auto-select optimizer (SGD, Adam, etc.)
        verbose=True,  # Print detailed logs
        seed=42,  # Random seed for reproducibility
        deterministic=True,  # Deterministic training
        workers=4,  # Number of data loading workers
        plots=True,  # Generate training plots
    )

    # Save training results to JSON
    results_path = output_path / "train" / "training_results.json"
    results_data = {
        "model_size": model_size,
        "epochs_trained": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dataset": dataset_yaml,
        "best_model": str(output_path / "train" / "weights" / "best.pt"),
        "last_model": str(output_path / "train" / "weights" / "last.pt"),
    }

    # Write results to file
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Print completion message
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best model: {output_path / 'train' / 'weights' / 'best.pt'}")
    print(f"Last model: {output_path / 'train' / 'weights' / 'last.pt'}")
    print(f"Results:    {results_path}")

    return results


def evaluate_model(model_path, dataset_yaml="data/yolo_ucf_cc50_real/dataset.yaml"):
    """
    Evaluate the trained model on validation set.
    
    Args:
        model_path: Path to trained model weights
        dataset_yaml: Path to dataset YAML configuration
    
    Returns:
        metrics: Evaluation metrics object
    """
    # Print evaluation header
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60)

    # Load the trained model
    model = YOLO(model_path)

    # Run evaluation on validation set
    # This computes mAP, precision, recall, etc.
    metrics = model.val(data=str(Path(dataset_yaml).absolute()))

    # Print evaluation metrics
    print(f"\nValidation Metrics:")
    print(f"  mAP50:       {metrics.box.map50:.4f}")
    print(f"  mAP50-95:    {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")

    # Save evaluation results
    eval_results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    # Save to JSON file
    eval_path = Path(model_path).parent / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nEvaluation results saved to: {eval_path}")

    return metrics


if __name__ == "__main__":
    # Train YOLOv8s on UCF CC50 real dataset
    results = train_yolov8(
        dataset_yaml="data/yolo_ucf_cc50_real/dataset.yaml",
        model_size="yolov8s",
        epochs=150,
        img_size=1280,
        batch_size=4,
        learning_rate=0.01,
        patience=25,
        output_dir="models/yolo_ucf_cc50_real",
    )

    # Evaluate the trained model
    best_model_path = "models/yolo_ucf_cc50_real/train/weights/best.pt"
    if Path(best_model_path).exists():
        evaluate_model(best_model_path, "data/yolo_ucf_cc50_real/dataset.yaml")
