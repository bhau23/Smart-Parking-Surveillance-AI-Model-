# Fall Detection System
## Smart Parking & Surveillance AI Model Challenge - Task 3

This project implements a real-time fall detection system using YOLOv8 with support for both camera feed and video file inputs.

## Key Features
- Real-time fall detection using YOLOv8
- Supports both webcam and video file inputs
- Streamlit web interface for easy interaction
- Advanced pose analysis for accurate fall detection
- Real-time alerts when falls are detected

## Setup and Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Project Structure:
```
fall_detection_assignment/
├── data/
│   └── data.yaml         # Dataset configuration
├── scripts/
│   ├── fall_detection.py # Core detection logic
│   └── streamlit_app.py  # Web interface
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Usage

### Using Streamlit Interface
```bash
streamlit run scripts/streamlit_app.py
```

#### Features:
- Select between webcam or video file input
- Adjust confidence threshold
- Real-time fall detection visualization
- Fall alerts display
- Easy-to-use interface

### Detection Logic
The system uses multiple techniques for accurate fall detection:
- Aspect ratio analysis of person bounding box
- Temporal smoothing for reduced false positives
- Fall duration verification
- Confidence thresholding

## Technical Details

### Model
- Base: YOLOv8
- Input size: 640x640
- Classes: 1 (fall)

### Fall Detection Parameters
- Confidence threshold: 0.45 (adjustable)
- Minimum fall duration: 0.3 seconds
- Aspect ratio threshold: 1.2

## Dataset
The model was trained on the Fall Detection Dataset from Roboflow:
https://universe.roboflow.com/hero-d6kgf/yolov5-fall-detection