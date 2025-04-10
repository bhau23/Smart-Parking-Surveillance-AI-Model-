from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

@dataclass
class Config:
    """Configuration class for the parking detector system."""
    
    # Video processing
    FRAME_SKIP: int = 2  # Process every nth frame
    RESIZE_WIDTH: int = 1920
    RESIZE_HEIGHT: int = 1080
    
    # Model paths
    YOLO_MODEL_PATH: str = "weights/yolov8n.pt"
    SPACE_CLASSIFIER_PATH: str = "model/model.p"
    
    # Detection parameters
    CONFIDENCE_THRESHOLD: float = 0.7  # Minimum confidence for detections
    IOU_THRESHOLD: float = 0.5  # IoU threshold for overlap detection
    
    # Hybrid detector settings
    CLASSIFIER_WEIGHT: float = 0.6  # Weight given to classifier vs YOLO
    TEMPORAL_SMOOTH_WINDOW: int = 5  # Number of frames for temporal smoothing
    
    # Device settings
    DEVICE: str = "cuda"  # Use "cpu" if no GPU available
    
    # Visualization
    DRAW_BOXES: bool = True  # Draw bounding boxes
    DRAW_CONFIDENCE: bool = True  # Show confidence scores
    DRAW_COUNT: bool = True  # Show parking space counts
    DRAW_VEHICLE_DETECTIONS: bool = True  # Show YOLO vehicle detections
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        load_dotenv()
        
        config = cls()
        
        # Override defaults with environment variables if they exist
        for field in config.__dataclass_fields__:
            env_value = os.getenv(f"PARKING_DETECTOR_{field}")
            if env_value is not None:
                field_type = type(getattr(config, field))
                # Handle boolean values specially
                if field_type == bool:
                    setattr(config, field, env_value.lower() == 'true')
                else:
                    setattr(config, field, field_type(env_value))
        
        # Auto-detect device if not set
        if not os.getenv("PARKING_DETECTOR_DEVICE"):
            import torch
            config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        return config
    
    def validate(self) -> bool:
        """Validate the configuration."""
        # Ensure model directories exist
        model_paths = [self.YOLO_MODEL_PATH, self.SPACE_CLASSIFIER_PATH]
        for path in model_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model not found at {path}")
        
        # Validate numeric parameters
        if self.FRAME_SKIP < 1:
            raise ValueError("FRAME_SKIP must be >= 1")
        
        if not (0 < self.CONFIDENCE_THRESHOLD <= 1):
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if not (0 < self.IOU_THRESHOLD <= 1):
            raise ValueError("IOU_THRESHOLD must be between 0 and 1")
            
        if not (0 < self.CLASSIFIER_WEIGHT < 1):
            raise ValueError("CLASSIFIER_WEIGHT must be between 0 and 1")
            
        if self.TEMPORAL_SMOOTH_WINDOW < 1:
            raise ValueError("TEMPORAL_SMOOTH_WINDOW must be >= 1")
        
        # Validate device
        if self.DEVICE not in ['cuda', 'cpu']:
            raise ValueError("DEVICE must be either 'cuda' or 'cpu'")
            
        return True
    
    def save_to_env(self, env_file: str = ".env"):
        """Save current configuration to .env file.
        
        Args:
            env_file: Path to .env file
        """
        with open(env_file, 'w') as f:
            for field in self.__dataclass_fields__:
                value = getattr(self, field)
                f.write(f"PARKING_DETECTOR_{field}={value}\n")