from typing import List, Tuple
import numpy as np

from .classifier import SpaceClassifier
from .detector import VehicleDetector

class HybridDetector:
    """Combines space classifier and YOLO detector for robust parking space detection."""
    
    def __init__(self, 
                 classifier: SpaceClassifier,
                 detector: VehicleDetector,
                 classifier_weight: float = 0.6,
                 confidence_threshold: float = 0.7):
        """Initialize the hybrid detector.
        
        Args:
            classifier: Initialized space classifier
            detector: Initialized vehicle detector
            classifier_weight: Weight given to classifier results (0-1)
            confidence_threshold: Threshold for confident decisions
        """
        self.classifier = classifier
        self.detector = detector
        self.classifier_weight = classifier_weight
        self.detector_weight = 1.0 - classifier_weight
        self.confidence_threshold = confidence_threshold
        
        # Keep track of previous results for temporal smoothing
        self.previous_results: List[Tuple[bool, float]] = []
        self.smooth_window = 5
    
    def detect_spaces(self,
                     frame: np.ndarray,
                     space_regions: List[np.ndarray],
                     space_boxes: List[Tuple[int, int, int, int]]
                     ) -> Tuple[List[bool], List[float]]:
        """Detect parking space occupancy using both methods.
        
        Args:
            frame: Full frame
            space_regions: List of cropped space images
            space_boxes: List of space bounding boxes (x, y, w, h)
            
        Returns:
            Tuple[List[bool], List[float]]: (is_empty_list, confidence_list)
        """
        # Get classifier predictions
        clf_empty, clf_conf = self.classifier.predict_batch(space_regions)
        
        # Get YOLO detections
        vehicle_detections = self.detector.detect_vehicles(frame)
        
        # Check each space with YOLO
        yolo_results = [
            self.detector.check_space_occupied(vehicle_detections, space_box)
            for space_box in space_boxes
        ]
        yolo_occupied = [not res[0] for res in yolo_results]  # Invert because check_space_occupied returns (is_occupied, conf)
        yolo_conf = [res[1] for res in yolo_results]
        
        # Combine results
        final_results = []
        for i in range(len(space_regions)):
            # Weight the predictions
            clf_score = clf_conf[i] * self.classifier_weight
            yolo_score = yolo_conf[i] * self.detector_weight
            
            # If predictions agree, use higher confidence
            if clf_empty[i] == yolo_occupied[i]:
                is_empty = clf_empty[i]
                confidence = max(clf_score, yolo_score)
            else:
                # If predictions disagree, use weighted average
                total_score = clf_score + yolo_score
                # Classifier says empty, YOLO says occupied
                if clf_empty[i]:
                    empty_score = clf_score
                    occupied_score = yolo_score
                else:
                    empty_score = yolo_score
                    occupied_score = clf_score
                    
                is_empty = empty_score > occupied_score
                confidence = max(empty_score, occupied_score) / total_score
            
            final_results.append((is_empty, confidence))
        
        # Apply temporal smoothing
        smoothed_results = self._apply_temporal_smoothing(final_results)
        
        # Split results
        is_empty_list = [res[0] for res in smoothed_results]
        confidence_list = [res[1] for res in smoothed_results]
        
        return is_empty_list, confidence_list
    
    def _apply_temporal_smoothing(self, 
                                current_results: List[Tuple[bool, float]]
                                ) -> List[Tuple[bool, float]]:
        """Apply temporal smoothing to detection results.
        
        Args:
            current_results: List of (is_empty, confidence) tuples
            
        Returns:
            List[Tuple[bool, float]]: Smoothed results
        """
        # Initialize previous results if needed
        if not self.previous_results:
            self.previous_results = current_results
            return current_results
            
        smoothed = []
        for i, (current_empty, current_conf) in enumerate(current_results):
            prev_empty, prev_conf = self.previous_results[i]
            
            # If current prediction is very confident, use it
            if current_conf > self.confidence_threshold:
                smoothed.append((current_empty, current_conf))
            # If previous prediction was very confident, maintain it
            elif prev_conf > self.confidence_threshold:
                smoothed.append((prev_empty, prev_conf * 0.9))  # Decay confidence
            # If both are low confidence, use current but with reduced confidence
            else:
                smoothed.append((current_empty, current_conf * 0.8))
        
        # Update previous results
        self.previous_results = smoothed
        
        return smoothed
    
    def reset_smoothing(self):
        """Reset temporal smoothing state."""
        self.previous_results = []