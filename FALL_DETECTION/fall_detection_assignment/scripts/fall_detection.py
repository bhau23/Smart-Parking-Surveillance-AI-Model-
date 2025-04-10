import os
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import torch
from collections import deque
import time

class FallDetector:
    def __init__(self, weights_path=None):
        """
        Initialize the Fall Detector with optional pre-trained weights
        Args:
            weights_path (str, optional): Path to pre-trained weights file
        """
        if weights_path and os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            print(f"Loaded weights from: {weights_path}")
        else:
            print("No weights provided, starting with base YOLOv8n model...")
            self.model = YOLO('yolov8n.pt')
        
        # Detection history for temporal smoothing
        self.detection_history = deque(maxlen=10)  # Increased history size
        self.fall_start_time = None
        self.MIN_FALL_DURATION = 0.3  # Reduced minimum duration for quicker detection
        self.DEFAULT_CONF = 0.5  # Increased confidence threshold
        self.aspect_ratio_history = deque(maxlen=5)  # Track aspect ratio changes
        
    def analyze_pose(self, bbox):
        """
        Enhanced pose analysis using aspect ratio and position
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        Returns:
            bool: True if pose likely represents a fall
        """
        if len(bbox) < 4:
            return False
            
        # Calculate aspect ratio of bounding box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height if height != 0 else 0
        
        # Store aspect ratio history
        self.aspect_ratio_history.append(aspect_ratio)
        
        # Check for sudden change in aspect ratio (indicating fall)
        if len(self.aspect_ratio_history) >= 2:
            aspect_ratio_change = abs(self.aspect_ratio_history[-1] - self.aspect_ratio_history[-2])
            sudden_change = aspect_ratio_change > 0.5
        else:
            sudden_change = False
        
        # Conditions for fall detection:
        # 1. Width greater than height (horizontal position)
        # 2. Sudden change in aspect ratio
        # 3. Aspect ratio within typical fall range
        is_horizontal = aspect_ratio > 1.2
        typical_fall_ratio = 0.8 < aspect_ratio < 2.5
        
        return (is_horizontal and typical_fall_ratio) or sudden_change
    
    def smooth_detections(self, is_fall):
        """
        Enhanced temporal smoothing with weighted recent history
        Args:
            is_fall (bool): Current frame fall detection
        Returns:
            bool: Smoothed detection result
        """
        self.detection_history.append(is_fall)
        
        # Weight recent detections more heavily
        if len(self.detection_history) >= 5:
            recent_weight = 0.7
            older_weight = 0.3
            
            recent_detections = list(self.detection_history)[-3:]
            older_detections = list(self.detection_history)[:-3]
            
            recent_score = sum(recent_detections) / len(recent_detections) * recent_weight
            older_score = sum(older_detections) / len(older_detections) * older_weight
            
            return (recent_score + older_score) > 0.5
        
        # If not enough history, use simple majority
        return sum(self.detection_history) > len(self.detection_history) * 0.6
    
    def train(self, data_yaml_path, epochs=100, imgsz=640, batch=8):
        """
        Train the model on fall detection dataset
        Args:
            data_yaml_path (str): Path to data.yaml file
            epochs (int): Number of training epochs
            imgsz (int): Input image size
            batch (int): Batch size
        Returns:
            dict: Training results
        """
        print(f"\nStarting model training...")
        print(f"Data config: {data_yaml_path}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch}\n")
        
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            patience=50,  # Early stopping patience
            save=True,  # Save best model
            project='runs/train',
            name='fall_detection'
        )
        
        # Load the best weights after training
        best_weights = 'runs/train/fall_detection/weights/best.pt'
        if os.path.exists(best_weights):
            print(f"\nLoading best weights from training: {best_weights}")
            self.model = YOLO(best_weights)
        
        return results
    
    def process_live_feed(self, source=0, conf=None):
        """
        Process live video feed for fall detection
        Args:
            source (int/str): Camera index or video stream URL
            conf (float): Confidence threshold
        """
        if conf is None:
            conf = self.DEFAULT_CONF
        
        print("\nFall Detection Started")
        print("-----------------------")
        print(f"Confidence threshold: {conf}")
        print("Press 'q' to quit")
        print("Green box: Standing")
        print("Yellow box: Possible fall")
        print("Red box: Fall detected")
        print("-----------------------\n")
        
        cap = cv2.VideoCapture(source)
        fall_detected = False
        possible_fall = False
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Perform detection
            results = self.model.predict(
                source=frame,
                conf=conf,
                verbose=False
            )
            
            # Process results
            annotated_frame = frame.copy()
            current_fall = False
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Analyze pose
                    is_fall_pose = self.analyze_pose([x1, y1, x2, y2])
                    
                    # Apply temporal smoothing
                    is_fall = self.smooth_detections(is_fall_pose)
                    
                    if is_fall:
                        if not fall_detected:
                            self.fall_start_time = time.time()
                            fall_detected = True
                            possible_fall = True
                        
                        # Check if fall duration exceeds minimum
                        if self.fall_start_time is not None and time.time() - self.fall_start_time >= self.MIN_FALL_DURATION:
                            current_fall = True
                            # Draw red box for confirmed fall
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, 'FALL DETECTED', (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        elif possible_fall:
                            # Draw yellow box for possible fall
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                            cv2.putText(annotated_frame, 'Possible Fall', (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    else:
                        # Draw green box for standing pose
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, 'Standing', (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        fall_detected = False
                        possible_fall = False
                        self.fall_start_time = None
            
            if not current_fall:
                self.fall_start_time = None
            
            # Display results
            cv2.imshow("Fall Detection", annotated_frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize detector
    detector = FallDetector()
    
    # Train the model
    print("Starting fall detection training...")
    detector.train(
        data_yaml_path='data/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8
    )
    
    # Start live detection
    detector.process_live_feed(conf=0.5)

if __name__ == "__main__":
    main()