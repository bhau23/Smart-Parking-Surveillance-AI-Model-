import argparse
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

class GroundTruthAnnotator:
    """Simple tool to annotate parking spaces as empty/occupied."""
    
    def __init__(self, mask_path: str):
        """Initialize the annotator.
        
        Args:
            mask_path: Path to parking space mask image
        """
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
            
        # Get connected components for parking spaces
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.mask, connectivity=8
        )
        
        # Store space coordinates (skip background at index 0)
        self.spaces = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            self.spaces.append((x, y, w, h))
    
    def annotate_frame(self, 
                      frame: np.ndarray,
                      current_states: List[bool] = None
                      ) -> Tuple[List[bool], bool]:
        """Run annotation interface for a frame.
        
        Args:
            frame: Frame to annotate
            current_states: Optional list of current space states
            
        Returns:
            Tuple[List[bool], bool]: (space_states, completed)
            completed is False if user requested to quit
        """
        # Create window
        window_name = "Parking Space Annotation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # If no current states, initialize all as occupied
        if current_states is None:
            current_states = [False] * len(self.spaces)
            
        # Create copy of frame for drawing
        display = frame.copy()
        
        # Draw initial state
        self._draw_spaces(display, current_states)
        cv2.imshow(window_name, display)
        
        space_idx = 0
        while space_idx < len(self.spaces):
            # Draw current space highlight
            highlight = display.copy()
            x, y, w, h = self.spaces[space_idx]
            cv2.rectangle(
                highlight,
                (x, y), (x + w, y + h),
                (0, 255, 255),  # Yellow highlight
                2
            )
            cv2.imshow(window_name, highlight)
            
            # Wait for keypress
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('e'):  # Mark as empty
                current_states[space_idx] = True
                space_idx += 1
            elif key == ord('o'):  # Mark as occupied
                current_states[space_idx] = False
                space_idx += 1
            elif key == ord('b'):  # Go back
                space_idx = max(0, space_idx - 1)
            elif key == ord('q'):  # Quit
                cv2.destroyWindow(window_name)
                return current_states, False
            
            # Update display
            display = frame.copy()
            self._draw_spaces(display, current_states)
            
        cv2.destroyWindow(window_name)
        return current_states, True
    
    def _draw_spaces(self, frame: np.ndarray, states: List[bool]):
        """Draw parking spaces on frame.
        
        Args:
            frame: Frame to draw on
            states: List of space states (True=empty)
        """
        for (x, y, w, h), is_empty in zip(self.spaces, states):
            color = (0, 255, 0) if is_empty else (0, 0, 255)  # Green=empty, Red=occupied
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def main():
    parser = argparse.ArgumentParser(description="Create ground truth annotations")
    
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing frames to annotate"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to parking space mask"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ground_truth.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Initialize annotator
    annotator = GroundTruthAnnotator(args.mask)
    
    # Get list of frames
    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    
    if not frame_paths:
        print(f"No frames found in {args.frames_dir}")
        return 1
        
    print("\nParking Space Annotation Tool")
    print("----------------------------")
    print("Controls:")
    print("  'e' - Mark space as Empty")
    print("  'o' - Mark space as Occupied")
    print("  'b' - Go back to previous space")
    print("  'q' - Quit\n")
    
    # Process each frame
    annotations = {}
    current_states = None
    
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        print(f"\nAnnotating {frame_path.name}...")
        states, completed = annotator.annotate_frame(frame, current_states)
        
        if not completed:
            print("\nAnnotation cancelled.")
            break
            
        # Save states for next frame (for efficiency)
        current_states = states
        
        # Store annotations
        annotations[frame_path.stem] = states
        
        print(f"Frame {frame_path.name} annotated: "
              f"{sum(1 for s in states if s)} empty, "
              f"{sum(1 for s in states if not s)} occupied")
    
    # Save annotations
    if annotations:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"\nAnnotations saved to {output_path}")
    else:
        print("\nNo annotations created")
    
    return 0

if __name__ == "__main__":
    exit(main())