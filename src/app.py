import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import shutil
from typing import Optional
from parking_detector import (
    Config,
    VideoProcessor,
    ParkingSpaceDetector,
    SpaceClassifier,
    VehicleDetector,
    HybridDetector
)

class ParkingDashboard:
    def __init__(self):
        """Initialize the parking dashboard."""
        st.set_page_config(
            page_title="Smart Parking Detection",
            page_icon="🅿️",
            layout="wide"
        )
        
        # Initialize configuration
        self.config = Config.from_env()
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize or reset session state
        if 'initialized' not in st.session_state:
            self.reset_session_state()
            st.session_state.initialized = True
            st.session_state.previous_video = None
    
    def reset_session_state(self):
        """Reset all session state variables."""
        st.session_state.occupancy_history = []
        st.session_state.processing_stats = []
        st.session_state.temp_files = []
    
    def save_uploaded_file(self, uploaded_file) -> Optional[Path]:
        """Save an uploaded file to temporary storage.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Path to saved file or None if save failed
        """
        if uploaded_file is None:
            return None
            
        try:
            # Create temp file path
            temp_path = self.temp_dir / uploaded_file.name
            
            # Save file
            temp_path.write_bytes(uploaded_file.getvalue())
            
            # Track for cleanup
            st.session_state.temp_files.append(temp_path)
            
            return temp_path
            
        except Exception as e:
            st.error(f"Failed to save uploaded file: {str(e)}")
            return None
    
    def cleanup_temp_files(self):
        """Remove all temporary files."""
        for temp_file in st.session_state.temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass
                
        st.session_state.temp_files = []
        
        # Try to remove temp directory if empty
        try:
            self.temp_dir.rmdir()
        except Exception:
            pass
    
    def init_detectors(self, mask_path: Path):
        """Initialize detection components.
        
        Args:
            mask_path: Path to parking space mask
        """
        self.space_detector = ParkingSpaceDetector(mask_path)
        self.classifier = SpaceClassifier(self.config.SPACE_CLASSIFIER_PATH)
        self.vehicle_detector = VehicleDetector(
            self.config.YOLO_MODEL_PATH,
            device=self.config.DEVICE
        )
        
        self.detector = HybridDetector(
            self.classifier,
            self.vehicle_detector,
            classifier_weight=self.config.CLASSIFIER_WEIGHT,
            confidence_threshold=self.config.CONFIDENCE_THRESHOLD
        )
    
    def update_history(self, occupied_count: int, total_spaces: int, processing_time: float):
        """Update occupancy and performance history.
        
        Args:
            occupied_count: Number of occupied spaces
            total_spaces: Total number of spaces
            processing_time: Frame processing time
        """
        timestamp = datetime.now()
        
        # Update occupancy history
        st.session_state.occupancy_history.append({
            'timestamp': timestamp,
            'occupied': occupied_count,
            'available': total_spaces - occupied_count,
            'occupancy_rate': (occupied_count / total_spaces) * 100
        })
        
        # Update processing stats
        st.session_state.processing_stats.append({
            'timestamp': timestamp,
            'processing_time': processing_time,
            'fps': 1.0 / processing_time if processing_time > 0 else 0
        })
        
        # Keep last hour of data
        cutoff = timestamp - timedelta(hours=1)
        st.session_state.occupancy_history = [
            x for x in st.session_state.occupancy_history
            if x['timestamp'] > cutoff
        ]
        st.session_state.processing_stats = [
            x for x in st.session_state.processing_stats
            if x['timestamp'] > cutoff
        ]
    
    def plot_occupancy_history(self, frame_idx: int):
        """Plot historical occupancy data.
        
        Args:
            frame_idx: Current frame index for unique keys
        """
        if not st.session_state.occupancy_history:
            return
            
        df = pd.DataFrame(st.session_state.occupancy_history)
        
        # Occupancy rate over time
        fig = px.line(
            df,
            x='timestamp',
            y='occupancy_rate',
            title='Parking Lot Occupancy Rate'
        )
        fig.update_layout(
            yaxis_title='Occupancy Rate (%)',
            xaxis_title='Time'
        )
        st.plotly_chart(fig, use_container_width=True, key=f"occupancy_timeline_{frame_idx}")
        
        # Current distribution
        latest = df.iloc[-1]
        fig = go.Figure(data=[
            go.Pie(
                values=[latest['occupied'], latest['available']],
                labels=['Occupied', 'Available'],
                hole=0.3
            )
        ])
        fig.update_layout(title='Current Space Distribution')
        st.plotly_chart(fig, key=f"space_distribution_{frame_idx}")
    
    def plot_performance_stats(self, frame_idx: int):
        """Plot system performance statistics.
        
        Args:
            frame_idx: Current frame index for unique keys
        """
        if not st.session_state.processing_stats:
            return
            
        df = pd.DataFrame(st.session_state.processing_stats)
        
        # FPS over time
        fig = px.line(
            df,
            x='timestamp',
            y='fps',
            title='Processing Speed'
        )
        fig.update_layout(
            yaxis_title='Frames per Second',
            xaxis_title='Time'
        )
        st.plotly_chart(fig, use_container_width=True, key=f"fps_timeline_{frame_idx}")
        
        # Average stats
        avg_fps = df['fps'].mean()
        avg_time = df['processing_time'].mean() * 1000  # to ms
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average FPS", f"{avg_fps:.1f}")
        with col2:
            st.metric("Average Processing Time", f"{avg_time:.1f} ms")
    
    def process_frame(self, frame: np.ndarray):
        """Process a single frame.
        
        Args:
            frame: Input frame to process
            
        Returns:
            tuple: (processed_frame, occupied_count, processing_time)
        """
        # Get parking space regions
        space_regions = self.space_detector.get_space_regions(frame)
        
        # Process frame
        start_time = time.time()
        is_empty_list, confidence_list = self.detector.detect_spaces(
            frame,
            space_regions,
            self.space_detector.parking_spaces
        )
        processing_time = time.time() - start_time
        
        # Draw results
        output_frame = self.space_detector.draw_spaces(
            frame, is_empty_list, confidence_list
        )
        
        occupied_count = sum(1 for x in is_empty_list if not x)
        return output_frame, occupied_count, processing_time
    
    def run(self):
        """Run the Streamlit application."""
        st.title("Smart Parking Detection System")
        
        # Sidebar configuration
        st.sidebar.header("Configuration")
        video_file = st.sidebar.file_uploader(
            "Upload Video",
            type=['mp4', 'avi']
        )
        mask_file = st.sidebar.file_uploader(
            "Upload Mask",
            type=['png', 'jpg']
        )
        
        # Main interface
        if not (video_file and mask_file):
            st.info("Please upload a video file and mask image to begin.")
            return
            
        try:
            # Save uploaded files
            video_path = self.save_uploaded_file(video_file)
            mask_path = self.save_uploaded_file(mask_file)
            
            if not (video_path and mask_path):
                st.error("Failed to process uploaded files")
                return

            # Reset state if video changed
            if (st.session_state.previous_video is None or
                st.session_state.previous_video != video_file.name):
                self.reset_session_state()
                st.session_state.previous_video = video_file.name

            # Initialize detectors
            self.init_detectors(mask_path)
            
            # Create display elements with better performance
            frame_col, stats_col = st.columns([0.7, 0.3])
            with frame_col:
                frame_placeholder = st.empty()
            with stats_col:
                stats_container = st.container()
                with stats_container:
                    st.subheader("System Statistics")
                    metrics_container = st.empty()
                    st.subheader("Historical Data")
                    history_container = st.empty()
                    st.subheader("Performance Metrics")
                    perf_container = st.empty()
            
            # Process frames continuously
            while True:
                # Process video
                video_processor = VideoProcessor(self.config)
                video_processor.open(video_path)
                
                try:
                    for frame, frame_idx in video_processor.process_frames():
                        # Process frame
                        output_frame, occupied_count, processing_time = self.process_frame(frame)
                        
                        # Update history
                        self.update_history(
                            occupied_count,
                            self.space_detector.get_space_count(),
                            processing_time
                        )
                        
                        # Display frame
                        frame_placeholder.image(
                            output_frame,
                            channels="BGR",
                            use_container_width=True,
                            caption=f"Frame {frame_idx}"
                        )
                        
                        # Update metrics
                        metrics_cols = metrics_container.columns(3)
                        metrics_cols[0].metric("Occupied Spaces", occupied_count)
                        metrics_cols[1].metric(
                            "Available Spaces",
                            self.space_detector.get_space_count() - occupied_count
                        )
                        metrics_cols[2].metric(
                            "Processing Time",
                            f"{processing_time*1000:.1f} ms"
                        )
                        
                        # Update plots (less frequently to improve performance)
                        if frame_idx % 5 == 0:  # Update every 5 frames
                            with history_container:
                                self.plot_occupancy_history(frame_idx)
                            with perf_container:
                                self.plot_performance_stats(frame_idx)
                            
                except Exception as e:
                    st.error(f"Error processing frame: {str(e)}")
                finally:
                    video_processor.close()
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()

if __name__ == "__main__":
    dashboard = ParkingDashboard()
    dashboard.run()