# Kumbh Mela Crowd Detection System

A real-time crowd detection and analysis system for the Kumbh Mela festival, using YOLOv8 for person detection with proximity-based risk analysis.

## Overview

This system provides:
- **Real-time person detection** using YOLOv8 (COCO pretrained)
- **Crowd density estimation** with heatmap generation
- **Proximity-based risk detection** to identify dangerous clusters
- **Video analysis pipeline** with annotated output
- **Streamlit dashboard** for interactive monitoring

## How It Works

### 1. Person Detection (YOLOv8)

The system uses YOLOv8 (You Only Look Once) for fast and accurate person detection:

```
Input Frame → YOLOv8 → Bounding Boxes → Person Count
```

**Key Features:**
- **Model:** YOLOv8s (small variant, good balance of speed/accuracy)
- **Class:** Person only (COCO class 0)
- **Confidence:** 0.25 minimum threshold
- **TTA:** Test-time augmentation (original + flipped)
- **Tiling:** Splits large images for better small object detection

### 2. Detection Pipeline

```python
# Simplified detection flow
def detect_persons(frame):
    # 1. Run YOLOv8 inference
    boxes = model.predict(frame, conf=0.25, classes=[0])
    
    # 2. Apply Non-Maximum Suppression
    boxes = nms(boxes, iou_threshold=0.5)
    
    # 3. Generate density heatmap
    density_map = generate_density_map(boxes)
    
    # 4. Analyze proximity risk
    risk_info = analyze_proximity(boxes)
    
    return count, boxes, density_map, risk_info
```

### 3. Proximity Risk Analysis

The system detects dangerous crowd clusters:

1. **Calculate distances** between all person pairs
2. **Find close pairs** within threshold (default: 80px)
3. **Build clusters** using DFS (Depth-First Search)
4. **Classify risk levels:**
   - **Safe:** Adequate spacing
   - **Low:** Some close pairs
   - **Moderate:** Small clusters forming
   - **High:** Multiple people very close
   - **Critical:** Large tight clusters (4+ people)

### 4. Video Processing

```python
# Video processing pipeline
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons
        count, boxes, density, risk = detector.detect_persons(frame)
        
        # Generate annotated frame
        annotated = draw_detections(frame, boxes, count, risk)
        
        # Save to output video
        writer.write(annotated)
```

## Model Performance

### YOLOv8 Person Detection Results

| Metric | Value |
|--------|-------|
| **Precision** | 76.92% |
| **Recall** | 28.95% |
| **F1-Score** | 42.07% |
| **mAP50** | 47.27% |
| **mAP50-95** | 13.29% |

### Detection Statistics

| Statistic | Count |
|-----------|-------|
| True Positives (TP) | 130 |
| False Positives (FP) | 39 |
| False Negatives (FN) | 319 |
| Total Ground Truth | 449 |
| Total Predictions | 169 |

### Training Loss Summary

| Loss Type | Initial | Final | Best |
|-----------|---------|-------|------|
| Box Loss (Localization) | 3.8050 | 2.2633 | 2.2633 |
| Classification Loss | 6.7190 | 1.5288 | 1.5288 |
| DFL Loss | 2.9360 | 1.8467 | 1.8467 |
| **Total Loss** | 13.4600 | 5.6388 | 5.6388 |

### Improved Model (YOLOv8s, 1280px, 200 epochs)

| Metric | Old Model | New Model | Improvement |
|--------|-----------|-----------|-------------|
| Precision | 76.92% | 43.01% | -44.1% |
| Recall | 28.95% | **72.61%** | **+150.8%** |
| F1-Score | 42.07% | **54.02%** | **+28.4%** |
| True Positives | 130 | **326** | **+150.8%** |
| False Negatives | 319 | **123** | **-61.4%** |

**Key Improvement:** Recall increased from 29% to 73%, detecting 2.5x more persons.

## Project Structure

```
Kumbhmela/
├── config.yaml                 # Main configuration file
├── run.py                      # Main pipeline script
├── train_yolov8.py             # YOLOv8 training script
├── evaluate_model.py           # Model evaluation
├── yolo_classification_report.py # Classification report
├── compare_models.py           # Model comparison
├── prepare_yolo_dataset.py     # Dataset preparation
├── download_ucf_cc50.py        # Dataset download
│
├── src/
│   ├── detection/
│   │   └── person_detector.py  # YOLOv8 person detection
│   ├── video/
│   │   └── processor.py        # Video processing pipeline
│   ├── analysis/
│   │   └── risk.py             # Risk analysis module
│   ├── inference/
│   │   └── predictor.py        # Density prediction
│   ├── model/
│   │   └── csrnet.py           # CSRNet model (density)
│   ├── dataset/
│   │   ├── crowd_dataset.py    # Dataset class
│   │   └── prepare_ucf.py      # UCF CC50 preparation
│   └── training/
│       └── train.py            # Training loop
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── data/
│   ├── ucf_cc50/               # Original dataset
│   ├── yolo_ucf_cc50/          # YOLO format dataset
│   └── yolo_ucf_cc50_v2/       # Improved dataset
│
├── models/                     # Trained models
│   └── yolo_ucf_cc50_real/
│       └── train/weights/      # Model checkpoints
│
├── runs/                       # YOLOv8 training runs
│   └── detect/
│       └── models/
│           └── yolo_ucf_cc50_v2/
│               └── train/
│                   └── weights/
│                       └── best.pt  # Best model
│
└── outputs/                    # Analysis results
    ├── yolo_classification_report.json
    ├── yolo_evaluation.json
    └── model_comparison.json
```

## Usage

### 1. Analyze a Video

```bash
# Run full pipeline with video
python run.py --video path/to/video.mp4

# Skip training (use existing model)
python run.py --video path/to/video.mp4 --skip-training
```

### 2. Train YOLOv8 Model

```bash
# Train on UCF CC50 dataset
python train_yolov8.py
```

### 3. Generate Classification Report

```bash
# Generate detailed metrics
python yolo_classification_report.py
```

### 4. Launch Dashboard

```bash
# Start Streamlit dashboard
cd dashboard
streamlit run app.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
video:
  yolo_model: "yolov8s"           # Model size
  confidence: 0.25                # Detection threshold
  proximity_threshold: 80         # Risk distance (px)
  cluster_min_size: 4             # Min people for risk
  use_tta: true                   # Test-time augmentation
  tile_size: 640                  # Tile size
  tile_overlap: 128               # Tile overlap
  custom_model_path: "path/to/model.pt"  # Custom model
```

## Dataset

### UCF CC50

- **Images:** 50 crowd scenes
- **Annotations:** Point coordinates (x, y) for each person
- **Scenes:** Stadiums, protests, marathons, concerts, streets
- **Density:** 10 to 1000+ people per image

### Dataset Conversion

Point annotations are converted to YOLO bounding boxes:

```python
# Point annotation: x y
# YOLO format: class_id x_center y_center width height

def create_bounding_box(x, y, box_size=50):
    half = box_size / 2
    x_center = x / img_width
    y_center = y / img_height
    width = box_size / img_width
    height = box_size / img_height
    return f"0 {x_center} {y_center} {width} {height}"
```

## Risk Detection Algorithm

```python
def analyze_proximity(boxes, threshold=80):
    # 1. Calculate centers
    centers = [(box[0]+box[2])/2, (box[1]+box[3])/2 for box in boxes]
    
    # 2. Find close pairs
    close_pairs = []
    for i, j in combinations(range(len(centers)), 2):
        dist = distance(centers[i], centers[j])
        if dist < threshold:
            close_pairs.append((i, j, dist))
    
    # 3. Build clusters (DFS)
    clusters = find_connected_components(close_pairs)
    
    # 4. Classify risk
    for cluster in clusters:
        if len(cluster) >= 8:
            risk = "critical"
        elif len(cluster) >= 6:
            risk = "high"
        elif len(cluster) >= 4:
            risk = "moderate"
    
    return risk_info
```

## Requirements

```
- Python 3.8+
- ultralytics (YOLOv8)
- opencv-python
- numpy
- torch
- streamlit
- imageio
- matplotlib
- pandas
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Kumbhmela

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_ucf_cc50.py

# Train model
python train_yolov8.py

# Run analysis
python run.py --video video.mp4
```

## License

This project is for educational and research purposes.

## Disclaimer

**IMPORTANT:** Risk scores generated by this system are PROTOTYPE/MODEL-DERIVED indicators only. They are NOT official safety assessments and should not be used as the sole basis for safety decisions.
