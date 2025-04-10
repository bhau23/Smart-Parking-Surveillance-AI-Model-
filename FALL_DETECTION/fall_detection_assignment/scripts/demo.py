import os
from fall_detection import FallDetector
import cv2
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fall Detection Demo')
    parser.add_argument('--train', action='store_true',
                      help='Train the model')
    parser.add_argument('--conf', type=float, default=0.45,
                      help='Detection confidence threshold')
    parser.add_argument('--weights', type=str, default=None,
                      help='Path to model weights')
    parser.add_argument('--video', type=str, default=None,
                      help='Path to video file for testing')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize paths
    base_dir = Path(__file__).resolve().parent.parent
    data_yaml = str(base_dir / 'data' / 'data.yaml')
    models_dir = base_dir / 'models'
    demo_dir = base_dir / 'demo_videos'
    
    # Ensure directories exist
    models_dir.mkdir(exist_ok=True)
    demo_dir.mkdir(exist_ok=True)
    
    # Initialize detector
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from: {args.weights}")
        detector = FallDetector(args.weights)
    else:
        print("Initializing new detector")
        detector = FallDetector()
    
    # Training phase
    if args.train:
        print("\nStarting model training...")
        print("This may take a while depending on your hardware.")
        print("Training parameters:")
        print("- Data config:", data_yaml)
        print("- Epochs: 100")
        print("- Image size: 640")
        print("- Batch size: 8")
        
        results = detector.train(
            data_yaml_path=data_yaml,
            epochs=100,
            imgsz=640,
            batch=8
        )
        print("Training completed!")
        
        # Load the best weights after training
        best_weights = str(models_dir / 'best.pt')
        if os.path.exists(best_weights):
            detector = FallDetector(best_weights)
            print(f"Loaded best weights from {best_weights}")
    
    # Test on video if provided
    if args.video and os.path.exists(args.video):
        print(f"\nProcessing video: {args.video}")
        print(f"Using confidence threshold: {args.conf}")
        results = detector.predict_video(
            args.video,
            save=True,
            conf=args.conf
        )
        print(f"Completed processing video")
    
    # Start live detection
    print("\nStarting live fall detection demo...")
    print(f"Using confidence threshold: {args.conf}")
    print("Detection improvements enabled:")
    print("- Pose analysis")
    print("- Temporal smoothing")
    print("- Fall duration check")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'c' to capture current frame")
    
    detector.process_live_feed(conf=args.conf)

if __name__ == "__main__":
    main()