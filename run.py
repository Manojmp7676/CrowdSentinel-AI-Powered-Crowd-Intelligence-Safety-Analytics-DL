"""
Main Pipeline Script for Kumbh Mela Crowd Detection System.

This script runs the complete end-to-end pipeline:
1. Prepare dataset (UCF CC50 or synthetic)
2. Train YOLOv8 model for person detection
3. Evaluate model performance
4. Process video for crowd analysis
5. Generate risk analysis report

Usage:
    python run.py --video path/to/video.mp4
    python run.py --skip-training  # Skip training, use existing model
"""

import sys  # System-specific parameters and functions
import yaml  # YAML file parsing
import argparse  # Command-line argument parsing
from pathlib import Path  # Object-oriented filesystem paths

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: config.yaml)
    
    Returns:
        config: Dictionary containing all configuration settings
    """
    with open(config_path, "r") as f:  # Open config file for reading
        return yaml.safe_load(f)  # Parse YAML and return as dictionary


def step_prepare_dataset(config):
    """
    Step 1: Prepare the UCF CC50 dataset for training.
    
    This function:
    - Inspects if dataset exists
    - Generates synthetic data if needed
    - Converts annotations to YOLO format
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 60)  # Print section separator
    print("STEP 1: Preparing Dataset")  # Print step name
    print("=" * 60)  # Print section separator
    
    # Import dataset preparation functions
    from src.dataset.prepare_ucf import inspect_dataset, prepare_dataset, generate_sample_data

    has_data = inspect_dataset(config)  # Check if dataset exists
    
    if not has_data:  # If no dataset found
        print("\nGenerating synthetic sample data for testing...")  # Inform user
        generate_sample_data(config)  # Create synthetic dataset
    
    prepare_dataset(config)  # Prepare dataset in YOLO format
    print("Dataset preparation complete.")  # Confirm completion


def step_train_model(config):
    """
    Step 2: Train the YOLOv8 model on UCF CC50 dataset.
    
    This function:
    - Loads YOLOv8s pretrained on COCO
    - Fine-tunes on person detection
    - Saves best model based on mAP
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 60)  # Print section separator
    print("STEP 2: Training Model")  # Print step name
    print("=" * 60)  # Print section separator
    
    # Import training function
    from src.training.train import train
    train("config.yaml")  # Run training with config
    print("Training complete.")  # Confirm completion


def step_evaluate_model(config):
    """
    Step 3: Evaluate the trained model on validation set.
    
    This function:
    - Loads trained model
    - Runs validation
    - Computes MAE, RMSE, and loss metrics
    
    Args:
        config: Configuration dictionary
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)  # Print section separator
    print("STEP 3: Evaluating Model")  # Print step name
    print("=" * 60)  # Print section separator
    
    # Import evaluation functions
    from src.training.train import validate  # Validation function
    from src.model.csrnet import load_trained_model  # Model loading
    from src.dataset.crowd_dataset import create_dataloaders  # Data loading
    import torch  # PyTorch library

    # Select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = load_trained_model(config, device)
    
    # Create validation dataloader
    _, val_loader = create_dataloaders(config)

    # Import loss function
    import torch.nn as nn
    criterion = nn.MSELoss()  # Mean Squared Error loss for density estimation
    
    # Run validation and get metrics
    val_loss, val_mae, val_rmse, pred_counts, gt_counts = validate(
        model, val_loader, criterion, device
    )

    # Print evaluation results
    print(f"\nEvaluation Results:")
    print(f"  MAE:  {val_mae:.2f}")  # Mean Absolute Error
    print(f"  RMSE: {val_rmse:.2f}")  # Root Mean Squared Error
    print(f"  Loss: {val_loss:.4f}")  # Validation loss

    return {"mae": val_mae, "rmse": val_rmse}  # Return metrics


def step_process_video(config, video_path):
    """
    Step 4: Process video through the YOLOv8 detection pipeline.
    
    This function:
    - Loads video file
    - Runs YOLOv8 detection on each frame
    - Generates annotated video with bounding boxes
    - Computes crowd statistics
    
    Args:
        config: Configuration dictionary
        video_path: Path to input video file
    
    Returns:
        results: Dictionary with video analysis results
    """
    print("\n" + "=" * 60)  # Print section separator
    print("STEP 4: Processing Video (YOLOv8 Person Detection)")  # Print step name
    print("=" * 60)  # Print section separator
    
    # Import video processor
    from src.video.processor import VideoProcessor

    processor = VideoProcessor(config)  # Initialize processor with config
    results = processor.process_video(video_path)  # Process video
    return results  # Return analysis results


def step_analyze_results(config, video_results):
    """
    Step 5: Generate risk analysis and save results.
    
    This function:
    - Extracts statistics from video results
    - Prints summary metrics
    - Saves full analysis to JSON file
    
    Args:
        config: Configuration dictionary
        video_results: Results from video processing
    
    Returns:
        video_results: Complete results dictionary
    """
    print("\n" + "=" * 60)  # Print section separator
    print("STEP 5: Results Summary")  # Print step name
    print("=" * 60)  # Print section separator
    
    # Import required modules
    from pathlib import Path  # File path handling
    import json  # JSON file handling

    # Extract statistics from results
    stats = video_results.get("statistics", {})  # Get statistics dictionary
    duration = video_results.get("video_duration_seconds", 0)  # Get video duration

    # Print person count results
    print(f"\nPerson Count Results:")
    print(f"  Average Count:  {stats['avg_count']:.1f}")  # Average persons per frame
    print(f"  Peak Count:     {stats['peak_count']}")  # Maximum persons in a frame
    print(f"  Min Count:      {stats['min_count']}")  # Minimum persons in a frame
    print(f"  Total Frames:   {video_results['total_frames']}")  # Total frames processed
    print(f"  Duration:       {duration:.1f}s")  # Video duration in seconds
    print(f"  Resolution:     {video_results['resolution']}")  # Video resolution

    # Create output directory if needed
    output_path = Path("outputs") / "analysis_report.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Save results to JSON file
    with open(output_path, "w") as f:
        json.dump(video_results, f, indent=2)  # Write with indentation

    print(f"\nResults saved to {output_path}")  # Confirm save

    return video_results  # Return complete results


def run_pipeline(video_path=None, skip_training=False):
    """
    Run the complete crowd detection pipeline.
    
    This is the main orchestrator function that runs all steps in sequence.
    
    Args:
        video_path: Path to video file (optional)
        skip_training: If True, skip training and use existing model
    """
    # Load configuration
    config = load_config()

    # Step 1: Prepare dataset
    step_prepare_dataset(config)

    # Step 2: Training (can skip with flag)
    if not skip_training:  # If training is not skipped
        step_train_model(config)  # Run training
    else:  # Training skipped
        print("\nSkipping training (--skip-training flag).")  # Inform user
        model_path = config["model"]["best_model_path"]  # Get model path
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"WARNING: No trained model found at {model_path}")  # Warn user
            print("Training is required for first run.")  # Explain requirement
            step_train_model(config)  # Run training anyway

    # Step 3: Evaluation
    step_evaluate_model(config)

    # Step 4 & 5: Video processing (if video provided)
    if video_path:  # If video path was provided
        video_results = step_process_video(config, video_path)  # Process video
        step_analyze_results(config, video_results)  # Analyze and save results
    else:  # No video provided
        print("\nNo video provided. Pipeline complete.")  # Inform user
        print("Use --video <path> to process a video.")  # Show usage


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Kumbh Mela Crowd Detection Pipeline"
    )
    parser.add_argument(
        "--video",  # Argument name
        type=str,  # Type: string
        help="Path to video file to analyze"  # Help text
    )
    parser.add_argument(
        "--skip-training",  # Argument name
        action="store_true",  # Boolean flag (present = True)
        help="Skip model training"  # Help text
    )
    parser.add_argument(
        "--config",  # Argument name
        type=str,  # Type: string
        default="config.yaml",  # Default value
        help="Config file path"  # Help text
    )

    args = parser.parse_args()  # Parse arguments
    
    # Run pipeline with provided arguments
    run_pipeline(
        video_path=args.video,  # Video path (or None)
        skip_training=args.skip_training  # Skip training flag
    )
