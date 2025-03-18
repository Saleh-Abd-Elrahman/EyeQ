"""
Camera module for handling video input from webcams or other camera devices.
"""

import cv2
import time
import logging
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Camera:
    """
    Camera class for handling video input from webcams or other camera devices.
    """
    
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize the camera.
        
        Args:
            camera_index: Index of the camera to use (default: 0 for primary webcam)
            width: Width of the video frame (default: 1280)
            height: Height of the video frame (default: 720)
            fps: Frames per second (default: 30)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.last_frame_time = 0
        self.frame_count = 0
        
    def start(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            bool: True if the camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}")
                return False
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings were applied
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized with resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            self.is_running = True
            self.last_frame_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
            
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame) pair
        """
        if not self.is_running or self.cap is None:
            logger.warning("Attempted to read from camera that is not running")
            return False, None
            
        success, frame = self.cap.read()
        
        if success:
            self.frame_count += 1
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            if time_diff > 0:
                current_fps = 1 / time_diff
                # Uncomment for debugging:
                # logger.debug(f"Current FPS: {current_fps:.2f}")
            self.last_frame_time = current_time
        else:
            logger.warning("Failed to read frame from camera")
            
        return success, frame
        
    def stop(self) -> None:
        """
        Stop the camera capture.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            logger.info(f"Camera stopped after capturing {self.frame_count} frames")
            
    def __del__(self):
        """
        Destructor to ensure camera resources are released.
        """
        self.stop()
        

class VideoFileCamera(Camera):
    """
    Extension of Camera class that reads from a video file instead of a live camera.
    Useful for testing and demonstrations.
    """
    
    def __init__(self, video_path: str, loop: bool = True):
        """
        Initialize with a video file path.
        
        Args:
            video_path: Path to the video file
            loop: Whether to loop the video when it ends (default: True)
        """
        super().__init__(camera_index=0)  # Index doesn't matter here
        self.video_path = video_path
        self.loop = loop
        
    def start(self) -> bool:
        """
        Start reading from the video file.
        
        Returns:
            bool: True if the video file was opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.video_path}")
                return False
                
            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video file opened: {self.video_path} ({self.width}x{self.height} @ {self.fps} FPS)")
            
            self.is_running = True
            self.last_frame_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error opening video file: {e}")
            return False
            
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video file.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame) pair
        """
        if not self.is_running or self.cap is None:
            return False, None
            
        success, frame = self.cap.read()
        
        # If video ends and looping is enabled, reset to beginning
        if not success and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()
            logger.info("Video loop restarted")
            
        if success:
            self.frame_count += 1
            
            # Simulate real-time speed
            current_time = time.time()
            target_frame_time = 1.0 / self.fps
            elapsed = current_time - self.last_frame_time
            
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
                
            self.last_frame_time = time.time()
            
        return success, frame


def list_available_cameras() -> List[int]:
    """
    List all available camera devices.
    
    Returns:
        List[int]: List of available camera indices
    """
    available_cameras = []
    
    # Try the first 5 camera indices (0-4)
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            
    return available_cameras 