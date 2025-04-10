import argparse
from pathlib import Path
import time
import torch

from parking_detector import (
    Config,
    VideoProcessor,
    ParkingSpaceDetector,
    ResultsVisualizer,
    SpaceClassifier,
    VehicleDetector,
    HybridDetector
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Smart Parking Detection System")
    
    parser.add_argument(
        "--video", 
        type=str, 
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--mask", 
        type=str, 
        required=True,
        help="Path to parking space mask image"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.mp4",
        help="Path to output video file"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model file"
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default="model/model.p",
        help="Path to space classifier model file"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = Config.from_env()
    
    # Override model paths from arguments
    config.YOLO_MODEL_PATH = args.yolo_model
    config.SPACE_CLASSIFIER_PATH = args.classifier_model
    
    # Validate configuration
    config.validate()
    
    # Initialize components
    try:
        # Initialize detectors
        print("Initializing models...")
        space_detector = ParkingSpaceDetector(args.mask)
        classifier = SpaceClassifier(config.SPACE_CLASSIFIER_PATH)
        vehicle_detector = VehicleDetector(
            config.YOLO_MODEL_PATH,
            device=config.DEVICE
        )
        
        # Create hybrid detector
        detector = HybridDetector(
            classifier,
            vehicle_detector,
            classifier_weight=config.CLASSIFIER_WEIGHT,
            confidence_threshold=config.CONFIDENCE_THRESHOLD
        )
        
        # Initialize visualizer
        visualizer = ResultsVisualizer(
            config.RESIZE_WIDTH,
            config.RESIZE_HEIGHT
        )
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process video
        with VideoProcessor(config) as video_proc:
            print(f"\nDevice: {vehicle_detector.device_info}")
            
            # Open video
            video_proc.open(args.video)
            video_props = video_proc.get_video_properties()
            visualizer.create_output_video(str(output_path), video_props['fps'])
            
            print(f"\nProcessing video with {video_props['frame_count']} frames...")
            print(f"Total parking spaces detected: {space_detector.get_space_count()}")
            
            # Process each frame
            for frame, frame_idx in video_proc.process_frames():
                start_time = time.time()
                
                # Get parking space regions
                space_regions = space_detector.get_space_regions(frame)
                
                # Detect parking space occupancy
                is_empty_list, confidence_list = detector.detect_spaces(
                    frame,
                    space_regions,
                    space_detector.parking_spaces
                )
                
                # Draw results
                frame_with_spaces = space_detector.draw_spaces(
                    frame, is_empty_list, confidence_list
                )
                
                processing_time = time.time() - start_time
                
                # Add statistics overlay
                output_frame = visualizer.draw_frame(
                    frame_with_spaces,
                    sum(1 for s in is_empty_list if not s),  # occupied spots
                    space_detector.get_space_count(),
                    confidence_list,
                    processing_time
                )
                
                # Write frame
                visualizer.write_frame(output_frame)
                
                # Print progress
                if frame_idx % 100 == 0:
                    print(f"Processed frame {frame_idx}/{video_props['frame_count']}")
            
            # Print final stats
            stats = visualizer.get_performance_stats()
            print("\nProcessing complete!")
            print(f"Average FPS: {stats['average_fps']:.2f}")
            print(f"Total time: {stats['elapsed_time']:.2f} seconds")
            print(f"Output saved to: {output_path}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())