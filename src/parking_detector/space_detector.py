import cv2
import numpy as np
from typing import List, Tuple, Union
from pathlib import Path

class ParkingSpaceDetector:
    """Handles detection and management of parking spaces."""
    
    def __init__(self, mask_path: Union[str, Path]):
        """Initialize the parking space detector.
        
        Args:
            mask_path: Path to the mask image defining parking spaces
            
        Raises:
            FileNotFoundError: If mask file doesn't exist
            ValueError: If mask is invalid or None
        """
        if not mask_path:
            raise ValueError("mask_path cannot be None or empty")
            
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
        try:
            self.mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if self.mask is None:
                raise ValueError(f"Failed to load mask image: {mask_path}")
                
            if self.mask.size == 0:
                raise ValueError(f"Empty mask image: {mask_path}")
        except Exception as e:
            raise ValueError(f"Error loading mask image: {str(e)}")
            
        # Initialize parking space data
        self.parking_spaces = self._extract_parking_spaces()
        self.total_spaces = len(self.parking_spaces)
    
    def _extract_parking_spaces(self) -> List[Tuple[int, int, int, int]]:
        """Extract parking space bounding boxes from the mask.
        
        Returns:
            List[Tuple[int, int, int, int]]: List of (x, y, w, h) bounding boxes
        """
        # Find connected components in the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.mask, connectivity=8
        )
        
        # Skip background (index 0)
        spaces = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter out potentially invalid spaces
            if w > 0 and h > 0 and w < self.mask.shape[1] and h < self.mask.shape[0]:
                spaces.append((x, y, w, h))
        
        return spaces
    
    def get_space_regions(self, frame: np.ndarray) -> List[np.ndarray]:
        """Extract image regions for each parking space.
        
        Args:
            frame: Input frame to extract regions from
            
        Returns:
            List[np.ndarray]: List of image regions for each space
        """
        regions = []
        for x, y, w, h in self.parking_spaces:
            region = frame[y:y+h, x:x+w]
            regions.append(region)
        return regions
    
    def draw_spaces(self, frame: np.ndarray, 
                   status: List[bool], 
                   confidence: List[float] = None) -> np.ndarray:
        """Draw parking space boxes on the frame.
        
        Args:
            frame: Input frame to draw on
            status: List of boolean status for each space (True=empty)
            confidence: Optional list of confidence scores
            
        Returns:
            np.ndarray: Frame with drawn parking spaces
        """
        result = frame.copy()
        
        for idx, ((x, y, w, h), is_empty) in enumerate(zip(self.parking_spaces, status)):
            # Choose color based on status
            color = (0, 255, 0) if is_empty else (0, 0, 255)
            
            # Draw box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence if provided
            if confidence and len(confidence) > idx:
                conf_text = f"{confidence[idx]:.2f}"
                cv2.putText(
                    result, conf_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
        
        # Draw total count
        empty_count = sum(1 for s in status if s)
        count_text = f"Empty: {empty_count}/{self.total_spaces}"
        cv2.putText(
            result, count_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        return result
    
    def get_space_count(self) -> int:
        """Get total number of parking spaces.
        
        Returns:
            int: Total number of detected parking spaces
        """
        return self.total_spaces