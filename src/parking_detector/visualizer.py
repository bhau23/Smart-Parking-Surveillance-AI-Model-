import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime

class ResultsVisualizer:
    """Handles visualization of parking detection results."""
    
    def __init__(self, output_width: int = 1920, output_height: int = 1080):
        """Initialize the results visualizer.
        
        Args:
            output_width: Width of output frames
            output_height: Height of output frames
        """
        self.output_width = output_width
        self.output_height = output_height
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.start_time = time.time()
        self.frame_count = 0
    
    def create_output_video(self, output_path: str, fps: float = 30.0) -> bool:
        """Create a video writer for saving output.
        
        Args:
            output_path: Path to save the output video
            fps: Frames per second for output video
            
        Returns:
            bool: True if writer was created successfully
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, fps,
            (self.output_width, self.output_height)
        )
        
        return self.video_writer.isOpened()
    
    def draw_frame(self, 
                  frame: np.ndarray,
                  occupied_spots: int,
                  total_spots: int,
                  detection_confidence: List[float],
                  processing_time: float) -> np.ndarray:
        """Draw detection results on frame.
        
        Args:
            frame: Input frame to draw on
            occupied_spots: Number of occupied parking spots
            total_spots: Total number of parking spots
            detection_confidence: List of confidence values for detections
            processing_time: Time taken to process the frame
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        result = frame.copy()
        
        # Draw statistics box
        stats_height = 120
        cv2.rectangle(result, (10, 10), (400, stats_height), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (400, stats_height), (255, 255, 255), 2)
        
        # Draw statistics text
        empty_spots = total_spots - occupied_spots
        occupancy_rate = (occupied_spots / total_spots) * 100 if total_spots > 0 else 0
        
        texts = [
            f"Empty Spots: {empty_spots}/{total_spots}",
            f"Occupancy Rate: {occupancy_rate:.1f}%",
            f"Avg Confidence: {np.mean(detection_confidence):.2f}",
            f"Processing Time: {processing_time*1000:.1f}ms",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(
                result, text, (20, 35 + i*20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
        
        return result
    
    def write_frame(self, frame: np.ndarray):
        """Write a frame to the output video.
        
        Args:
            frame: Frame to write
        """
        if self.video_writer:
            if (frame.shape[1], frame.shape[0]) != (self.output_width, self.output_height):
                frame = cv2.resize(frame, (self.output_width, self.output_height))
            self.video_writer.write(frame)
            self.frame_count += 1
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics.
        
        Returns:
            dict: Dictionary containing performance stats
        """
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'elapsed_time': elapsed_time,
            'processed_frames': self.frame_count,
            'average_fps': fps
        }
    
    def close(self):
        """Close the video writer and release resources."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()