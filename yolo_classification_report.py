"""
YOLOv8 Classification Report and Loss Analysis.

Generates detailed metrics including:
- Per-class precision, recall, F1-score
- Confusion matrix
- Training loss curves
- Detection statistics
"""

import torch
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import pandas as pd


def compute_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    box1: (N, 4) array of [x1, y1, x2, y2]
    box2: (M, 4) array of [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    N = box1.shape[0]
    M = box2.shape[0]

    # Expand dimensions for broadcasting
    box1 = box1[:, np.newaxis, :]  # (N, 1, 4)
    box2 = box2[np.newaxis, :, :]  # (1, M, 4)

    x1 = np.maximum(box1[..., 0], box2[..., 0])
    y1 = np.maximum(box1[..., 1], box2[..., 1])
    x2 = np.minimum(box1[..., 2], box2[..., 2])
    y2 = np.minimum(box1[..., 3], box2[..., 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def evaluate_detections(model_path, dataset_yaml, iou_threshold=0.5, conf_threshold=0.25):
    """
    Evaluate detection results and compute classification metrics.
    """
    model = YOLO(model_path)

    # Load dataset info
    with open(dataset_yaml, "r") as f:
        dataset_config = yaml.safe_load(f)

    data_dir = Path(dataset_config["path"])
    val_dir = data_dir / "images" / "val"
    label_dir = data_dir / "labels" / "val"

    # Get all validation images
    val_images = list(val_dir.glob("*.jpg"))

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_gt_count = 0
    all_pred_count = 0
    all_matched = 0

    per_image_results = []

    for img_path in val_images:
        # Get ground truth
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        # Keep normalized coordinates (0-1)
                        x1 = xc - w/2
                        y1 = yc - h/2
                        x2 = xc + w/2
                        y2 = yc + h/2
                        gt_boxes.append([x1, y1, x2, y2])

        gt_count = len(gt_boxes)
        all_gt_count += gt_count

        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)

        pred_boxes = []
        if results and len(results) > 0 and results[0].boxes is not None:
            img_h, img_w = results[0].orig_shape
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                # Normalize to 0-1 range
                xyxy[0] /= img_w
                xyxy[1] /= img_h
                xyxy[2] /= img_w
                xyxy[3] /= img_h
                pred_boxes.append(xyxy.tolist())

        pred_count = len(pred_boxes)
        all_pred_count += pred_count

        # Match predictions to ground truth
        if gt_count > 0 and pred_count > 0:
            gt_array = np.array(gt_boxes)
            pred_array = np.array(pred_boxes)

            ious = compute_iou(gt_array, pred_array)  # (gt_count, pred_count)

            matched_gt = set()
            matched_pred = set()

            # Greedy matching - for each GT, find best matching pred
            for i in range(gt_count):
                best_iou = 0
                best_j = -1
                for j in range(pred_count):
                    if j not in matched_pred and ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_j = j
                if best_iou >= iou_threshold and best_j >= 0:
                    matched_gt.add(i)
                    matched_pred.add(best_j)

            tp = len(matched_gt)
            fp = pred_count - len(matched_pred)
            fn = gt_count - tp

            all_tp += tp
            all_fp += fp
            all_fn += fn
            all_matched += tp
        elif pred_count > 0:
            all_fp += pred_count
        elif gt_count > 0:
            all_fn += gt_count

        per_image_results.append({
            "image": img_path.name,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "tp": tp if gt_count > 0 and pred_count > 0 else 0,
        })

    # Compute metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": all_tp,
        "false_positives": all_fp,
        "false_negatives": all_fn,
        "total_gt": all_gt_count,
        "total_pred": all_pred_count,
        "matched": all_matched,
        "per_image": per_image_results,
    }


def plot_training_losses(output_dir):
    """Plot training loss curves."""
    results_csv = Path("runs/detect/models/yolo_ucf_cc50/train/results.csv")
    if not results_csv.exists():
        print("Training results not found!")
        return

    df = pd.read_csv(results_csv)

    # Get column names
    cols = df.columns.tolist()
    box_col = [c for c in cols if 'box_loss' in c][0]
    cls_col = [c for c in cols if 'cls_loss' in c][0]
    dfl_col = [c for c in cols if 'dfl_loss' in c][0]
    map50_col = [c for c in cols if 'mAP50' in c and '95' not in c][0]
    prec_col = [c for c in cols if 'precision' in c][0]
    rec_col = [c for c in cols if 'recall' in c][0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("YOLOv8 Training Results", fontsize=16, fontweight="bold")

    # Box Loss
    axes[0, 0].plot(df['epoch'], df[box_col], 'b-', linewidth=2, label='Box Loss')
    axes[0, 0].set_title("Box Loss (Localization)", fontsize=12)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Classification Loss
    axes[0, 1].plot(df['epoch'], df[cls_col], 'r-', linewidth=2, label='Cls Loss')
    axes[0, 1].set_title("Classification Loss", fontsize=12)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # DFL Loss
    axes[1, 0].plot(df['epoch'], df[dfl_col], 'g-', linewidth=2, label='DFL Loss')
    axes[1, 0].set_title("Distribution Focal Loss", fontsize=12)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # mAP and Metrics
    axes[1, 1].plot(df['epoch'], df[map50_col], 'm-', linewidth=2, label='mAP50')
    axes[1, 1].plot(df['epoch'], df[prec_col], 'c--', linewidth=2, label='Precision')
    axes[1, 1].plot(df['epoch'], df[rec_col], 'y--', linewidth=2, label='Recall')
    axes[1, 1].set_title("Detection Metrics", fontsize=12)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "yolo_training_losses.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training loss plot saved to: {plot_path}")


def print_classification_report(metrics):
    """Print formatted classification report."""
    print("\n" + "=" * 70)
    print("YOLOv8 PERSON DETECTION - CLASSIFICATION REPORT")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Precision':<25} {metrics['precision']:>15.4f}")
    print(f"{'Recall':<25} {metrics['recall']:>15.4f}")
    print(f"{'F1-Score':<25} {metrics['f1_score']:>15.4f}")

    print(f"\n{'Detection Statistics':<25} {'Count':>15}")
    print("-" * 40)
    print(f"{'True Positives (TP)':<25} {metrics['true_positives']:>15}")
    print(f"{'False Positives (FP)':<25} {metrics['false_positives']:>15}")
    print(f"{'False Negatives (FN)':<25} {metrics['false_negatives']:>15}")
    print(f"{'Total Ground Truth':<25} {metrics['total_gt']:>15}")
    print(f"{'Total Predictions':<25} {metrics['total_pred']:>15}")

    print("\n" + "=" * 70)


def print_loss_summary():
    """Print training loss summary."""
    results_csv = Path("runs/detect/models/yolo_ucf_cc50/train/results.csv")
    if not results_csv.exists():
        print("Training results not found!")
        return

    df = pd.read_csv(results_csv)
    cols = df.columns.tolist()
    box_col = [c for c in cols if 'box_loss' in c][0]
    cls_col = [c for c in cols if 'cls_loss' in c][0]
    dfl_col = [c for c in cols if 'dfl_loss' in c][0]

    print("\n" + "=" * 70)
    print("YOLOv8 TRAINING LOSS SUMMARY")
    print("=" * 70)
    print(f"\n{'Loss Type':<25} {'Initial':>12} {'Final':>12} {'Best':>12}")
    print("-" * 61)
    print(f"{'Box Loss':<25} {df[box_col].iloc[0]:>12.4f} {df[box_col].iloc[-1]:>12.4f} {df[box_col].min():>12.4f}")
    print(f"{'Classification Loss':<25} {df[cls_col].iloc[0]:>12.4f} {df[cls_col].iloc[-1]:>12.4f} {df[cls_col].min():>12.4f}")
    print(f"{'DFL Loss':<25} {df[dfl_col].iloc[0]:>12.4f} {df[dfl_col].iloc[-1]:>12.4f} {df[dfl_col].min():>12.4f}")

    total_loss = df[box_col] + df[cls_col] + df[dfl_col]
    print(f"\n{'Total Loss':<25} {total_loss.iloc[0]:>12.4f} {total_loss.iloc[-1]:>12.4f} {total_loss.min():>12.4f}")
    print("=" * 70)


def main():
    """Run full evaluation."""
    print("=" * 70)
    print("YOLOv8 MODEL EVALUATION - UCF CC50")
    print("=" * 70)

    # Configuration
    model_path = "runs/detect/models/yolo_ucf_cc50/train/weights/best.pt"
    dataset_yaml = "data/yolo_ucf_cc50/dataset.yaml"

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    # Print loss summary
    print_loss_summary()

    # Evaluate detections
    print("\nEvaluating detections on validation set...")
    metrics = evaluate_detections(model_path, dataset_yaml, iou_threshold=0.5, conf_threshold=0.25)

    # Print classification report
    print_classification_report(metrics)

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results = {
        "model": model_path,
        "dataset": dataset_yaml,
        "metrics": {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1_score"], 4),
        },
        "statistics": {
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "total_gt": metrics["total_gt"],
            "total_pred": metrics["total_pred"],
        },
    }

    results_path = output_dir / "yolo_classification_report.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nClassification report saved to: {results_path}")

    # Plot training losses
    plot_training_losses(output_dir)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
