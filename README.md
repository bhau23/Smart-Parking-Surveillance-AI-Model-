# Smart Parking & Surveillance AI System

This project implements two main components as part of the Smart Parking & Surveillance AI Model Challenge:

1. Parking Space Detection: Real-time monitoring system that detects available and occupied parking spots using computer vision and deep learning, achieving over 97% accuracy using a hybrid approach.

2. Fall Detection: YOLO-based human action recognition system that detects fall incidents in surveillance footage.

## Performance Metrics

### Parking Detection
- **Accuracy**: 97.47% (exceeds required 90%)
- **Precision**: 92.42%
- **Recall**: 100.00%
- **Processing Speed**: 0.61 FPS
- [Detailed Analysis](validation_results/analysis.md)

### Fall Detection
- Implemented using custom-trained YOLO model
- Real-time fall incident detection
- Demo video available in `FALL_DETECTION/fall_detection_assignment/demo_videos/videoplayback.mp4`

## Project Structure

```
├── src/                    # Parking Detection Implementation
│   ├── parking_detector/   # Main package
│   │   ├── config.py      # Configuration management
│   │   ├── detector.py    # YOLO vehicle detection
│   │   ├── classifier.py  # Space classification
│   │   └── ...           # Other modules
│   └── main.py           # Main detection script
├── FALL_DETECTION/        # Fall Detection Implementation
│   └── fall_detection_assignment/
│       ├── scripts/       # Python implementation
│       │   ├── fall_detection.py  # Core detection logic
│       │   ├── demo.py           # Demo script
│       │   └── streamlit_app.py  # UI implementation
│       └── demo_videos/   # Example videos
├── validation_results/    # Performance analysis
└── docs/                 # Project documentation
```

## Features

### Parking Detection
- Hybrid detection combining:
  * Traditional computer vision for space detection
  * Pre-trained YOLO model for vehicle detection
  * Machine learning classifier for space occupancy
- Interactive Streamlit UI for:
  * Real-time parking space monitoring
  * Live statistics and occupancy tracking
  * Easy video and mask file upload
  * [Watch Demo Video](validation_results/streamlit-app-2025-04-10-04-04-64.webm)
- Comprehensive validation framework
- Configurable processing parameters

### Fall Detection
- Custom YOLO model for human action recognition
- Real-time fall incident detection
- Features:
  * Person detection and tracking
  * Fall action classification
  * Real-time alert system
- Interactive Streamlit interface
- Demonstration video included

## Setup & Usage

### Parking Detection

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the detection:
```bash
python src/main.py --video dataset/parking/parking_1920_1080.mp4 --mask dataset/parking/mask_1920_1080.png
```

### Fall Detection

1. Navigate to fall detection directory:
```bash
cd FALL_DETECTION/fall_detection_assignment
```

2. Run the Streamlit app:
```bash
python -m streamlit run scripts/streamlit_app.py
```

## Configuration

The parking detection system can be configured using environment variables or a .env file:

- `PARKING_DETECTOR_FRAME_SKIP`: Process every nth frame (default: 2)
- `PARKING_DETECTOR_CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.7)
- `PARKING_DETECTOR_IOU_THRESHOLD`: IoU threshold for overlap detection (default: 0.5)
- `PARKING_DETECTOR_CLASSIFIER_WEIGHT`: Weight for classifier vs YOLO (default: 0.6)

## Performance & Validation

### Parking Detection Results
- 97.47% overall accuracy
- Perfect recall (100%) - no missed occupied spaces
- 92.42% precision - very few false positives
- Detailed metrics in [validation_results/analysis.md](validation_results/analysis.md)

### Fall Detection Results
- Successfully detects fall incidents in real-time
- Tested with various scenarios and angles
- Demo video showcases detection capabilities

## License

MIT License