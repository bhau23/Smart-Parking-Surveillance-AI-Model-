import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Generator, Tuple
from .config import Config

class VideoProcessor:
    """Handles video input processing and frame extraction."""
    
    def __init__(self, config: Config):
        """Initialize the video processor.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.current_frame_idx = 0
        
    def open(self, video_path: str | Path) -> bool:
        """Open a video file for processing.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            bool: True if video was opened successfully
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video file cannot be opened
            ValueError: If video_path is invalid
        """
        if not video_path:
            raise ValueError("video_path cannot be None or empty")
            
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            self.cap = cv2.VideoCapture(str(video_path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.frame_count <= 0:
                raise RuntimeError(f"Invalid video file: {video_path}")
                
            self.current_frame_idx = 0
            return True
            
        except Exception as e:
            # Clean up if initialization fails
            if self.cap:
                self.cap.release()
                self.cap = None
            raise RuntimeError(f"Error opening video file: {str(e)}")
    
    def get_video_properties(self) -> dict:
        """Get properties of the currently opened video.
        
        Returns:
            dict: Dictionary containing video properties
        """
        if not self.cap:
            raise RuntimeError("No video file is currently open")
            
        return {
            'frame_count': self.frame_count,
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fourcc': self.cap.get(cv2.CAP_PROP_FOURCC)
        }
    
    def process_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Generate processed frames from the video.
        
        Yields:
            Tuple[np.ndarray, int]: Processed frame and its index
        """
        if not self.cap:
            raise RuntimeError("No video file is currently open")
            
        frame_idx = 0
        while frame_idx < self.frame_count:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                    
                if frame_idx % self.config.FRAME_SKIP == 0:
                    # Resize frame if needed
                    if (frame.shape[1] != self.config.RESIZE_WIDTH or
                        frame.shape[0] != self.config.RESIZE_HEIGHT):
                        frame = cv2.resize(
                            frame,
                            (self.config.RESIZE_WIDTH, self.config.RESIZE_HEIGHT)
                        )
                    
                    yield frame, frame_idx
                
                frame_idx += 1
                self.current_frame_idx = frame_idx
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                break
    
    def close(self):
        """Close the video file and release resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.frame_count = 0
            self.current_frame_idx = 0
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()