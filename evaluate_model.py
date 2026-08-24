"""
Model Evaluation Script for Crowd Density Estimation.

Provides:
- MSE Loss on validation set
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Classification report by density levels
- Training history visualization
"""

import torch
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset.crowd_dataset import create_dataloaders
from src.model.csrnet import CSRNet


def load_trained_model(config, device):
    """Load the trained CSRNet model."""
    model = CSRNet(pretrained_backbone=False)
    checkpoint_path = config["model"]["best_model_path"]

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
        print(f"Trained for {checkpoint.get('epoch', 'N/A')} epochs")
        print(f"Best Val MAE: {checkpoint.get('val_mae', 'N/A'):.4f}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        sys.exit(1)

    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, val_loader, device, config):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    num_batches = 0

    all_pred_counts = []
    all_gt_counts = []
    all_pred_densities = []
    all_gt_densities = []

    density_thresholds = config["risk"]["density_thresholds"]

    with torch.no_grad():
        for images, densities, counts in val_loader:
            images = images.to(device)
            densities = densities.to(device)

            pred_densities = model(images)

            # Compute MSE loss
            mse_loss = torch.nn.functional.mse_loss(pred_densities, densities)
            total_loss += mse_loss.item()

            # Compute counts
            pred_counts = pred_densities.sum(dim=(1, 2, 3)).cpu().numpy()
            gt_counts = densities.sum(dim=(1, 2, 3)).cpu().numpy()

            # Compute metrics
            mae = np.mean(np.abs(pred_counts - gt_counts))
            rmse = np.sqrt(np.mean((pred_counts - gt_counts) ** 2))

            total_mae += mae
            total_rmse += rmse
            num_batches += 1

            all_pred_counts.extend([float(c) for c in pred_counts])
            all_gt_counts.extend([float(c) for c in gt_counts])

            # Compute density stats
            pred_mean_density = pred_densities.mean(dim=(2, 3)).cpu().numpy().flatten()
            gt_mean_density = densities.mean(dim=(2, 3)).cpu().numpy().flatten()
            all_pred_densities.extend(pred_mean_density.tolist())
            all_gt_densities.extend(gt_mean_density.tolist())

    avg_loss = total_loss / max(num_batches, 1)
    avg_mae = total_mae / max(num_batches, 1)
    avg_rmse = total_rmse / max(num_batches, 1)

    return {
        "mse_loss": avg_loss,
        "rmse_loss": np.sqrt(avg_loss),
        "mae": avg_mae,
        "rmse": avg_rmse,
        "pred_counts": all_pred_counts,
        "gt_counts": all_gt_counts,
        "pred_densities": all_pred_densities,
        "gt_densities": all_gt_densities,
    }


def get_density_level(density, thresholds):
    """Classify density into levels."""
    if density <= thresholds["low"]:
        return "Low"
    elif density <= thresholds["moderate"]:
        return "Moderate"
    elif density <= thresholds["high"]:
        return "High"
    else:
        return "Very High"


def classification_report(gt_counts, pred_counts, gt_densities, pred_densities, thresholds):
    """Generate classification report based on density levels."""
    gt_levels = [get_density_level(float(d), thresholds) for d in gt_densities]
    pred_levels = [get_density_level(float(d), thresholds) for d in pred_densities]

    classes = ["Low", "Moderate", "High", "Very High"]
    report = {}

    for cls in classes:
        gt_mask = np.array(gt_levels) == cls
        pred_mask = np.array(pred_levels) == cls

        tp = np.sum(gt_mask & pred_mask)
        fp = np.sum(~gt_mask & pred_mask)
        fn = np.sum(gt_mask & ~pred_mask)
        tn = np.sum(~gt_mask & ~pred_mask)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(gt_mask)

        report[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": int(support),
        }

    # Overall accuracy
    correct = np.sum(np.array(gt_levels) == np.array(pred_levels))
    total = len(gt_levels)
    accuracy = correct / total if total > 0 else 0

    return report, accuracy


def plot_training_history(history_path, output_dir):
    """Plot training history from JSON file."""
    if not Path(history_path).exists():
        print(f"No training history found at {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CSRNet Training History", fontsize=16, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0, 0].set_title("MSE Loss", fontsize=12)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE plot
    axes[0, 1].plot(epochs, history["train_mae"], "b-", label="Train MAE", linewidth=2)
    axes[0, 1].plot(epochs, history["val_mae"], "r-", label="Val MAE", linewidth=2)
    axes[0, 1].set_title("Mean Absolute Error (MAE)", fontsize=12)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # RMSE plot
    axes[1, 0].plot(epochs, history["val_rmse"], "g-", label="Val RMSE", linewidth=2)
    axes[1, 0].set_title("Root Mean Squared Error (RMSE)", fontsize=12)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("RMSE")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate plot
    axes[1, 1].plot(epochs, history["learning_rate"], "m-", label="Learning Rate", linewidth=2)
    axes[1, 1].set_title("Learning Rate Schedule", fontsize=12)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "training_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to {plot_path}")


def plot_predictions(gt_counts, pred_counts, output_dir):
    """Plot predicted vs actual counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Predictions vs Ground Truth", fontsize=14, fontweight="bold")

    # Scatter plot
    axes[0].scatter(gt_counts, pred_counts, alpha=0.6, c="blue", edgecolors="k", s=50)
    min_val = min(min(gt_counts), min(pred_counts))
    max_val = max(max(gt_counts), max(pred_counts))
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    axes[0].set_xlabel("Ground Truth Count")
    axes[0].set_ylabel("Predicted Count")
    axes[0].set_title("Predicted vs Actual Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = np.array(pred_counts) - np.array(gt_counts)
    axes[1].hist(errors, bins=15, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    axes[1].set_xlabel("Prediction Error (Pred - GT)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "prediction_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction plot saved to {plot_path}")


def print_classification_report(report, accuracy):
    """Print formatted classification report."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (by Density Level)")
    print("=" * 70)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)

    for cls in ["Low", "Moderate", "High", "Very High"]:
        metrics = report[cls]
        print(f"{cls:<12} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1-score']:>10.4f} {metrics['support']:>10}")

    print("-" * 70)

    # Macro avg
    macro_precision = np.mean([report[c]["precision"] for c in report])
    macro_recall = np.mean([report[c]["recall"] for c in report])
    macro_f1 = np.mean([report[c]["f1-score"] for c in report])
    total_support = sum(report[c]["support"] for c in report)

    print(f"{'Macro Avg':<12} {macro_precision:>10.4f} {macro_recall:>10.4f} "
          f"{macro_f1:>10.4f} {total_support:>10}")
    print("=" * 70)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    """Run full model evaluation."""
    print("=" * 70)
    print("CROWD DENSITY MODEL EVALUATION")
    print("=" * 70)

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Load model
    print("\nLoading trained model...")
    model = load_trained_model(config, device)

    # Evaluate
    print("\nEvaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, device, config)

    # Print metrics
    print("\n" + "=" * 70)
    print("MODEL METRICS")
    print("=" * 70)
    print(f"MSE Loss:        {metrics['mse_loss']:.6f}")
    print(f"RMSE Loss:       {metrics['rmse_loss']:.6f}")
    print(f"MAE:             {metrics['mae']:.4f}")
    print(f"RMSE:            {metrics['rmse']:.4f}")
    print("=" * 70)

    # Classification report
    density_thresholds = config["risk"]["density_thresholds"]
    report, accuracy = classification_report(
        metrics["gt_counts"], metrics["pred_counts"],
        metrics["gt_densities"], metrics["pred_densities"],
        density_thresholds
    )
    print_classification_report(report, accuracy)

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results = {
        "model_metrics": {
            "mse_loss": round(metrics["mse_loss"], 6),
            "rmse_loss": round(metrics["rmse_loss"], 6),
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "accuracy": round(accuracy, 4),
        },
        "classification_report": report,
        "per_sample_results": [
            {
                "gt_count": round(float(gt), 2),
                "pred_count": round(float(pred), 2),
                "error": round(float(pred - gt), 2),
                "abs_error": round(float(abs(pred - gt)), 2),
            }
            for gt, pred in zip(metrics["gt_counts"], metrics["pred_counts"])
        ],
    }

    results_path = output_dir / "model_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {results_path}")

    # Plot training history
    history_path = "models/training_history.json"
    plot_training_history(history_path, output_dir)

    # Plot predictions
    plot_predictions(metrics["gt_counts"], metrics["pred_counts"], output_dir)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
