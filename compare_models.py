"""
Compare old and new YOLOv8 models.
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
    """Compute IoU between two sets of boxes."""
    N = box1.shape[0]
    M = box2.shape[0]
    box1 = box1[:, np.newaxis, :]
    box2 = box2[np.newaxis, :, :]
    x1 = np.maximum(box1[..., 0], box2[..., 0])
    y1 = np.maximum(box1[..., 1], box2[..., 1])
    x2 = np.minimum(box1[..., 2], box2[..., 2])
    y2 = np.minimum(box1[..., 3], box2[..., 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)


def evaluate_model(model_path, dataset_yaml, iou_threshold=0.5, conf_threshold=0.25):
    """Evaluate a model and return metrics."""
    model = YOLO(model_path)
    
    with open(dataset_yaml, "r") as f:
        dataset_config = yaml.safe_load(f)
    
    data_dir = Path(dataset_config["path"])
    val_dir = data_dir / "images" / "val"
    label_dir = data_dir / "labels" / "val"
    
    val_images = list(val_dir.glob("*.jpg"))
    
    all_tp, all_fp, all_fn = 0, 0, 0
    all_gt_count, all_pred_count = 0, 0
    
    for img_path in val_images:
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = xc - w/2
                        y1 = yc - h/2
                        x2 = xc + w/2
                        y2 = yc + h/2
                        gt_boxes.append([x1, y1, x2, y2])
        
        gt_count = len(gt_boxes)
        all_gt_count += gt_count
        
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        pred_boxes = []
        if results and len(results) > 0 and results[0].boxes is not None:
            img_h, img_w = results[0].orig_shape
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                xyxy[0] /= img_w
                xyxy[1] /= img_h
                xyxy[2] /= img_w
                xyxy[3] /= img_h
                pred_boxes.append(xyxy.tolist())
        
        pred_count = len(pred_boxes)
        all_pred_count += pred_count
        
        if gt_count > 0 and pred_count > 0:
            gt_array = np.array(gt_boxes)
            pred_array = np.array(pred_boxes)
            ious = compute_iou(gt_array, pred_array)
            
            matched_gt = set()
            matched_pred = set()
            
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
        elif pred_count > 0:
            all_fp += pred_count
        elif gt_count > 0:
            all_fn += gt_count
    
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "gt_count": all_gt_count,
        "pred_count": all_pred_count,
    }


def get_training_stats(results_csv):
    """Get training statistics from CSV."""
    if not Path(results_csv).exists():
        return None
    
    df = pd.read_csv(results_csv)
    cols = df.columns.tolist()
    box_col = [c for c in cols if 'box_loss' in c][0]
    cls_col = [c for c in cols if 'cls_loss' in c][0]
    dfl_col = [c for c in cols if 'dfl_loss' in c][0]
    map50_col = [c for c in cols if 'mAP50' in c and '95' not in c][0]
    prec_col = [c for c in cols if 'precision' in c][0]
    rec_col = [c for c in cols if 'recall' in c][0]
    
    return {
        "epochs": len(df),
        "final_box_loss": float(df[box_col].iloc[-1]),
        "final_cls_loss": float(df[cls_col].iloc[-1]),
        "final_dfl_loss": float(df[dfl_col].iloc[-1]),
        "best_map50": float(df[map50_col].max()),
        "best_precision": float(df[prec_col].max()),
        "best_recall": float(df[rec_col].max()),
    }


def main():
    print("=" * 70)
    print("YOLOv8 MODEL COMPARISON - OLD vs IMPROVED")
    print("=" * 70)
    
    # Old model
    old_model = "runs/detect/models/yolo_ucf_cc50/train/weights/best.pt"
    old_dataset = "data/yolo_ucf_cc50/dataset.yaml"
    old_csv = "runs/detect/models/yolo_ucf_cc50/train/results.csv"
    
    # New model
    new_model = "runs/detect/models/yolo_ucf_cc50_v2/train/weights/best.pt"
    new_dataset = "data/yolo_ucf_cc50_v2/dataset.yaml"
    new_csv = "runs/detect/models/yolo_ucf_cc50_v2/train/results.csv"
    
    print("\nEvaluating OLD model (20px boxes, 640px images)...")
    old_metrics = evaluate_model(old_model, old_dataset)
    old_stats = get_training_stats(old_csv)
    
    print("\nEvaluating IMPROVED model (50px boxes, 1280px images)...")
    new_metrics = evaluate_model(new_model, new_dataset)
    new_stats = get_training_stats(new_csv)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'OLD Model':>15} {'NEW Model':>15} {'Improvement':>15}")
    print("-" * 70)
    
    metrics = [
        ("Precision", old_metrics["precision"], new_metrics["precision"]),
        ("Recall", old_metrics["recall"], new_metrics["recall"]),
        ("F1-Score", old_metrics["f1_score"], new_metrics["f1_score"]),
    ]
    
    for name, old_val, new_val in metrics:
        improvement = ((new_val - old_val) / old_val * 100) if old_val > 0 else 0
        print(f"{name:<25} {old_val:>14.4f} {new_val:>14.4f} {improvement:>+14.1f}%")
    
    print(f"\n{'Detection Stats':<25} {'OLD Model':>15} {'NEW Model':>15}")
    print("-" * 55)
    print(f"{'True Positives':<25} {old_metrics['tp']:>15} {new_metrics['tp']:>15}")
    print(f"{'False Positives':<25} {old_metrics['fp']:>15} {new_metrics['fp']:>15}")
    print(f"{'False Negatives':<25} {old_metrics['fn']:>15} {new_metrics['fn']:>15}")
    print(f"{'Total GT':<25} {old_metrics['gt_count']:>15} {new_metrics['gt_count']:>15}")
    print(f"{'Total Pred':<25} {old_metrics['pred_count']:>15} {new_metrics['pred_count']:>15}")
    
    if old_stats and new_stats:
        print(f"\n{'Training Stats':<25} {'OLD Model':>15} {'NEW Model':>15}")
        print("-" * 55)
        print(f"{'Epochs Trained':<25} {old_stats['epochs']:>15} {new_stats['epochs']:>15}")
        print(f"{'Best mAP50':<25} {old_stats['best_map50']:>15.4f} {new_stats['best_map50']:>15.4f}")
        print(f"{'Final Box Loss':<25} {old_stats['final_box_loss']:>15.4f} {new_stats['final_box_loss']:>15.4f}")
        print(f"{'Final Cls Loss':<25} {old_stats['final_cls_loss']:>15.4f} {new_stats['final_cls_loss']:>15.4f}")
        print(f"{'Final DFL Loss':<25} {old_stats['final_dfl_loss']:>15.4f} {new_stats['final_dfl_loss']:>15.4f}")
    
    print("=" * 70)
    
    # Save comparison
    comparison = {
        "old_model": {
            "metrics": {k: round(v, 4) for k, v in old_metrics.items()},
            "training": old_stats,
        },
        "new_model": {
            "metrics": {k: round(v, 4) for k, v in new_metrics.items()},
            "training": new_stats,
        },
    }
    
    with open("outputs/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("\nComparison saved to outputs/model_comparison.json")


if __name__ == "__main__":
    main()
