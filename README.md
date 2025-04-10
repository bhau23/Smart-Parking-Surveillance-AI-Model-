# Smart Parking & Surveillance AI System

This project implements two main components as part of the Smart Parking & Surveillance AI Model Challenge:

1. Parking Space Detection: Real-time monitoring system that detects available and occupied parking spots using computer vision and deep learning, achieving over 97% accuracy using a hybrid approach.

2. Fall Detection: YOLO-based human action recognition system that detects fall incidents in surveillance footage.

Both components are documented below with their respective implementations and results.

[View Project Status and Task Completion Details](PROJECT_STATUS.md)

## Performance Metrics

- **Accuracy**: 97.47% (exceeds 90% requirement)
- **Precision**: 92.42%
- **Recall**: 100.00%
- **Processing Speed**: 0.61 FPS
- [Detailed Analysis](validation_results/analysis.md)

## Project Structure

```
├── src/
│   ├── parking_detector/      # Main package
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   ├── video_processor.py # Video processing
│   │   ├── space_detector.py  # Parking space detection
│   │   ├── classifier.py     # Space classification
│   │   ├── detector.py       # YOLO vehicle detection
│   │   ├── hybrid_detector.py # Combined detection system
│   │   ├── visualizer.py     # Results visualization
│   │   |── validation.py     # Accuracy validation
|   |   |___app.py            # streamlit ui 
│   ├── main.py               # Main detection script
│   ├── validate_accuracy.py  # Validation script
│   └── create_ground_truth.py # Annotation tool
├── dataset/                   # Training data and models
├── weights/                   # Model weights
├── model/                     # Classifier model
├── validation_results/        # Validation outputs
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Features

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
- Ground truth annotation tool

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Detection

Process a video file:
```bash
python src/main.py --video dataset/parking/parking_1920_1080.mp4 --mask dataset/parking/mask_1920_1080.png
```

Optional arguments:
- `--output`: Path for output video (default: output.mp4)

### Validation

Run accuracy validation:
```bash
# Extract frames and create ground truth
python src/validate_accuracy.py --video <video_path> --mask <mask_path> --ground-truth <annotations_path>
```

Create ground truth annotations:
```bash
python src/create_ground_truth.py --frames-dir <frames_dir> --mask <mask_path> --output <output_path>
```

## Configuration

The system can be configured using environment variables or a .env file:

- `PARKING_DETECTOR_FRAME_SKIP`: Process every nth frame (default: 2)
- `PARKING_DETECTOR_CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.7)
- `PARKING_DETECTOR_IOU_THRESHOLD`: IoU threshold for overlap detection (default: 0.5)
- `PARKING_DETECTOR_CLASSIFIER_WEIGHT`: Weight for classifier vs YOLO (default: 0.6)

## Related Projects

### Fall Detection System
The fall detection implementation is maintained in a separate repository:
- Location: `D:/fall_detection/`
- Features:
  * Custom YOLO model for human action detection
  * Real-time fall incident recognition
  * Demonstration video available at `D:/fall_detection/demo.mp4`

## Future Enhancements

While both main tasks have been completed successfully, there are potential areas for future enhancement:

1. Task 2: Edge Device Deployment (on hold)
   - Model optimization for resource-constrained devices
   - Real-time processing on edge hardware
   - Performance profiling and optimization
   - Integration of both detection systems on edge devices

2. System Integration
   - Combining parking and fall detection into a unified surveillance system
   - Shared notification and alerting system
   - Integrated dashboard for all monitoring functions

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed task status and implementation plans.

## Validation Results

The system has been validated against manually annotated ground truth data:
- 97.47% overall accuracy
- Perfect recall (100%) - no missed occupied spaces
- 92.42% precision - very few false positives
- Detailed metrics available in [validation_results/analysis.md](validation_results/analysis.md)

## License

MIT License