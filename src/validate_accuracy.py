import argparse
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
import time

from parking_detector import (
    Config,
    VideoProcessor,
    ParkingSpaceDetector,
    SpaceClassifier,
    VehicleDetector,
    HybridDetector
)
from parking_detector.validation import AccuracyValidator

def extract_test_frames(video_path: str, 
                       output_dir: str,
                       num_frames: int = 50) -> list:
    """Extract evenly spaced frames from video for testing.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        num_frames: Number of frames to extract
        
    Returns:
        list: List of (frame_id, frame_path) tuples
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Use consistent frame ID format
            frame_id = f"frame_{idx:06d}"
            frame_path = output_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append((frame_id, str(frame_path)))
    
    cap.release()
    return extracted_frames

def load_ground_truth(ground_truth_path: str) -> dict:
    """Load ground truth annotations.
    
    Args:
        ground_truth_path: Path to ground truth JSON file
        
    Returns:
        dict: Ground truth data
    """
    with open(ground_truth_path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Validate parking detection accuracy")
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to test video"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to parking space mask"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground truth annotations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Directory for validation results"
    )
    parser.add_argument(
        "--num-test-frames",
        type=int,
        default=50,
        help="Number of frames to test"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    config = Config.from_env()
    
    print("\nInitializing detectors...")
    space_detector = ParkingSpaceDetector(args.mask)
    classifier = SpaceClassifier(config.SPACE_CLASSIFIER_PATH)
    vehicle_detector = VehicleDetector(
        config.YOLO_MODEL_PATH,
        device=config.DEVICE
    )
    
    detector = HybridDetector(
        classifier,
        vehicle_detector,
        classifier_weight=config.CLASSIFIER_WEIGHT,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )
    
    # Initialize validator
    validator = AccuracyValidator(args.output_dir)
    
    # Extract test frames
    print("\nExtracting test frames...")
    test_frames = extract_test_frames(
        args.video,
        Path(args.output_dir) / "test_frames",
        args.num_test_frames
    )
    
    # Load ground truth
    print("\nLoading ground truth data...")
    ground_truth = load_ground_truth(args.ground_truth)
    
    # Process test frames
    print("\nProcessing test frames...")
    for frame_id, frame_path in tqdm(test_frames):
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        try:
            # Add ground truth
            if frame_id in ground_truth:
                validator.add_ground_truth(frame_id, ground_truth[frame_id])
            else:
                print(f"Warning: No ground truth for frame {frame_id}")
                continue
            
            # Get parking space regions
            space_regions = space_detector.get_space_regions(frame)
            
            # Process frame
            start_time = time.time()
            is_empty_list, confidence_list = detector.detect_spaces(
                frame,
                space_regions,
                space_detector.parking_spaces
            )
            processing_time = time.time() - start_time
            
            # Add prediction
            validator.add_prediction(
                frame_id,
                is_empty_list,
                confidence_list,
                processing_time
            )
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            continue  # Skip this frame but continue with others
    
    try:
        # Generate validation results
        print("\nGenerating validation report...")
        metrics = validator.calculate_metrics()
        
        print("\nValidation Results:")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"F1 Score: {metrics['f1_score']:.2%}")
        print(f"Average Confidence: {metrics['average_confidence']:.2f}")
        print(f"Average Processing Time: {metrics['average_processing_time']*1000:.1f}ms")
        print(f"Average FPS: {metrics['fps']:.1f}")
        
        # Generate visualizations
        validator.plot_performance_graphs()
        confusion_matrix_path = validator.generate_confusion_matrix_plot()
        
        # Save full report
        report_path = validator.generate_report()
        print(f"\nFull report saved to: {report_path}")
        print(f"Confusion matrix plot saved to: {confusion_matrix_path}")
        
        # Final status
        if metrics['accuracy'] > 0.9:
            print("\nValidation PASSED - Accuracy > 90%")
            return 0
        else:
            print("\nValidation FAILED - Accuracy < 90%")
            return 1
            
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("No valid frames were processed. Please ensure ground truth data matches extracted frames.")
        return 1

if __name__ == "__main__":
    exit(main())