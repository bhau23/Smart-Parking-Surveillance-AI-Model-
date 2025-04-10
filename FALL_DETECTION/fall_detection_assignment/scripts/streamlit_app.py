import streamlit as st
import cv2
import tempfile
from fall_detection import FallDetector
import time
import os

def main():
    st.title("Fall Detection System")
    st.sidebar.title("Settings")

    # Load pre-trained model
    weights_path = st.sidebar.selectbox(
        "Select Model Weights",
        options=[
            "yolov8n.pt",  # Base model
            "runs/train/fall_detection/weights/best.pt"  # Trained model
        ],
        index=1
    )

    # Initialize detector
    if os.path.exists(weights_path):
        detector = FallDetector(weights_path)
        st.sidebar.success(f"Model loaded: {weights_path}")
    else:
        detector = FallDetector()
        st.sidebar.warning("Using base model (not trained for fall detection)")

    # Confidence threshold
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    # Input source selection
    source = st.sidebar.radio("Select Input Source", ["Camera", "Video File"])

    if source == "Video File":
        video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        if video_file is not None:
            try:
                # Save uploaded video to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                vf_path = tfile.name
                tfile.close()
                
                st.video(video_file)
                
                if st.button("Start Detection"):
                    stframe = st.empty()
                    cap = cv2.VideoCapture(vf_path)
                    
                    try:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Perform detection
                            results = detector.model.predict(
                                source=frame,
                                conf=confidence,
                                verbose=False
                            )
                            
                            # Process results
                            annotated_frame = frame.copy()
                            
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                for box in results[0].boxes:
                                    # Get box coordinates
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    
                                    # Analyze pose
                                    is_fall_pose = detector.analyze_pose([x1, y1, x2, y2])
                                    is_fall = detector.smooth_detections(is_fall_pose)
                                    
                                    if is_fall:
                                        # Draw red box for fall detection
                                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                        cv2.putText(annotated_frame, 'FALL DETECTED', (int(x1), int(y1) - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                        
                                        # Add fall alert
                                        st.warning("⚠️ Fall Detected!")
                                    else:
                                        # Draw green box for normal pose
                                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                        cv2.putText(annotated_frame, 'Normal', (int(x1), int(y1) - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            # Convert BGR to RGB
                            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            stframe.image(annotated_frame)
                            time.sleep(0.1)  # Add small delay for smooth playback
                    
                    finally:
                        cap.release()
                        
                    try:
                        # Try to remove temp file
                        if os.path.exists(vf_path):
                            os.close(tfile.fileno())  # Ensure file is closed
                            os.unlink(vf_path)
                    except Exception as e:
                        st.warning(f"Note: Temporary file cleanup will be handled by the system")
            
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
    
    else:  # Camera input
        if st.button("Start Camera"):
            stframe = st.empty()
            cap = cv2.VideoCapture(0)
            
            stop_button = st.button("Stop")
            
            try:
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Perform detection
                    results = detector.model.predict(
                        source=frame,
                        conf=confidence,
                        verbose=False
                    )
                    
                    # Process results
                    annotated_frame = frame.copy()
                    
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Analyze pose
                            is_fall_pose = detector.analyze_pose([x1, y1, x2, y2])
                            is_fall = detector.smooth_detections(is_fall_pose)
                            
                            if is_fall:
                                # Draw red box for fall detection
                                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(annotated_frame, 'FALL DETECTED', (int(x1), int(y1) - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                
                                # Add fall alert
                                st.warning("⚠️ Fall Detected!")
                            else:
                                # Draw green box for normal pose
                                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, 'Normal', (int(x1), int(y1) - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Convert BGR to RGB
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated_frame)
                    time.sleep(0.1)  # Add small delay for smooth playback
            
            finally:
                cap.release()

if __name__ == "__main__":
    main()