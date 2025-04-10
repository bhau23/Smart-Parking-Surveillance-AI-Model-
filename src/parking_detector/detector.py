from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import torch

class VehicleDetector:
    """Vehicle detector using YOLOv8."""
    
    # COCO class indices for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize the vehicle detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            device: Device to run inference on ('cpu', 'cuda', etc.)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Load model
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
            
        self.device = device
    
    def detect_vehicles(self, 
                       frame: np.ndarray,
                       conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45
                       ) -> List[Tuple[List[float], float, int]]:
        """Detect vehicles in a frame.
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List[Tuple[List[float], float, int]]: List of (bbox, confidence, class_id)
                where bbox is [x1, y1, x2, y2] in pixel coordinates
        """
        # Run inference
        results = self.model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=list(self.VEHICLE_CLASSES.keys())
        )
        
        detections = []
        if len(results) > 0:
            # Get boxes, scores and class ids
            boxes = results[0].boxes
            
            for box in boxes:
                # Get box coordinates (x1, y1, x2, y2)
                bbox = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                
                detections.append((bbox.tolist(), conf, class_id))
        
        return detections
    
    def check_space_occupied(self, 
                           vehicle_boxes: List[Tuple[List[float], float, int]],
                           space_box: Tuple[int, int, int, int],
                           iou_threshold: float = 0.3
                           ) -> Tuple[bool, float]:
        """Check if a parking space is occupied by any detected vehicle.
        
        Args:
            vehicle_boxes: List of vehicle detections (bbox, conf, class_id)
            space_box: Parking space box (x, y, w, h)
            iou_threshold: IoU threshold for overlap detection
            
        Returns:
            Tuple[bool, float]: (is_occupied, confidence)
        """
        # Convert space box from (x, y, w, h) to (x1, y1, x2, y2)
        x, y, w, h = space_box
        space_box_xyxy = [x, y, x + w, y + h]
        
        max_iou = 0.0
        max_conf = 0.0
        
        for bbox, conf, _ in vehicle_boxes:
            iou = self._calculate_iou(bbox, space_box_xyxy)
            if iou > max_iou:
                max_iou = iou
                max_conf = conf
                
        is_occupied = max_iou > iou_threshold
        return is_occupied, max_conf if is_occupied else 1.0 - max_iou
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes.
        
        Args:
            box1: First box coordinates [x1, y1, x2, y2]
            box2: Second box coordinates [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    @property
    def device_info(self) -> str:
        """Get information about the device being used.
        
        Returns:
            str: Device information string
        """
        if self.device == 'cuda':
            return f"CUDA - {torch.cuda.get_device_name()}"
        return "CPU"