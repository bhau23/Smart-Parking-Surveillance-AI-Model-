from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from datetime import datetime

class AccuracyValidator:
    """Validates accuracy of parking space detection system."""
    
    def __init__(self, output_dir: str = "validation_results"):
        """Initialize the validator.
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.ground_truth: Dict[str, List[bool]] = {}  # frame_id -> list of space states
        self.predictions: Dict[str, List[bool]] = {}   # frame_id -> list of space states
        self.confidences: Dict[str, List[float]] = {}  # frame_id -> list of confidence values
        
        # Performance metrics
        self.processing_times: List[float] = []
        self.frame_timestamps: List[datetime] = []
    
    def add_ground_truth(self, frame_id: str, space_states: List[bool]):
        """Add ground truth data for a frame.
        
        Args:
            frame_id: Unique identifier for the frame
            space_states: List of boolean states for each parking space (True=empty)
        """
        self.ground_truth[frame_id] = space_states
    
    def add_prediction(self, 
                      frame_id: str, 
                      space_states: List[bool],
                      confidences: List[float],
                      processing_time: float):
        """Add prediction data for a frame.
        
        Args:
            frame_id: Unique identifier for the frame
            space_states: List of boolean states for each parking space (True=empty)
            confidences: List of confidence values for predictions
            processing_time: Time taken to process the frame
        """
        self.predictions[frame_id] = space_states
        self.confidences[frame_id] = confidences
        self.processing_times.append(processing_time)
        self.frame_timestamps.append(datetime.now())
    
    def calculate_metrics(self) -> dict:
        """Calculate validation metrics.
        
        Returns:
            dict: Dictionary containing various accuracy metrics
        """
        if not self.ground_truth or not self.predictions:
            raise ValueError("No data available for validation")
            
        # Collect all ground truth and predictions
        y_true = []
        y_pred = []
        all_confidences = []
        
        for frame_id in self.ground_truth.keys():
            if frame_id in self.predictions:
                y_true.extend(self.ground_truth[frame_id])
                y_pred.extend(self.predictions[frame_id])
                all_confidences.extend(self.confidences[frame_id])
        
        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'average_confidence': np.mean(all_confidences),
            'average_processing_time': np.mean(self.processing_times),
            'fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0
        }
        
        return metrics
    
    def generate_confusion_matrix_plot(self) -> str:
        """Generate confusion matrix visualization.
        
        Returns:
            str: Path to saved plot
        """
        y_true = []
        y_pred = []
        
        for frame_id in self.ground_truth.keys():
            if frame_id in self.predictions:
                y_true.extend(self.ground_truth[frame_id])
                y_pred.extend(self.predictions[frame_id])
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Occupied', 'Empty'],
            yticklabels=['Occupied', 'Empty']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
    
    def generate_report(self) -> str:
        """Generate validation report.
        
        Returns:
            str: Path to saved report
        """
        metrics = self.calculate_metrics()
        confusion_matrix_plot = self.generate_confusion_matrix_plot()
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'plots': {
                'confusion_matrix': confusion_matrix_plot
            },
            'summary': {
                'total_frames': len(self.ground_truth),
                'total_predictions': len(self.predictions),
                'total_spaces_validated': sum(len(states) for states in self.ground_truth.values()),
                'validation_passed': metrics['accuracy'] > 0.9  # 90% threshold
            }
        }
        
        # Save report
        report_path = self.output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return str(report_path)
    
    def plot_performance_graphs(self):
        """Generate performance visualization graphs."""
        # Processing time graph
        plt.figure(figsize=(10, 5))
        plt.plot(self.processing_times)
        plt.title('Processing Time per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Processing Time (seconds)')
        plt.grid(True)
        
        plot_path = self.output_dir / 'processing_times.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Confidence distribution
        all_confidences = []
        for conf_list in self.confidences.values():
            all_confidences.extend(conf_list)
            
        plt.figure(figsize=(10, 5))
        plt.hist(all_confidences, bins=50)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True)
        
        plot_path = self.output_dir / 'confidence_distribution.png'
        plt.savefig(plot_path)
        plt.close()
    
    def save_error_cases(self, 
                        frame_generator,
                        max_cases: int = 10):
        """Save visualization of error cases.
        
        Args:
            frame_generator: Generator that yields (frame_id, frame) tuples
            max_cases: Maximum number of error cases to save
        """
        error_dir = self.output_dir / 'error_cases'
        error_dir.mkdir(exist_ok=True)
        
        error_count = 0
        for frame_id, frame in frame_generator:
            if frame_id not in self.ground_truth or frame_id not in self.predictions:
                continue
                
            gt = self.ground_truth[frame_id]
            pred = self.predictions[frame_id]
            
            # Check for errors
            for i, (g, p) in enumerate(zip(gt, pred)):
                if g != p and error_count < max_cases:
                    # Save error case
                    error_path = error_dir / f'error_{error_count}_space_{i}.jpg'
                    cv2.imwrite(str(error_path), frame)
                    error_count += 1
                    
                if error_count >= max_cases:
                    break