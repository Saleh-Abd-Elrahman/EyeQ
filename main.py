#!/usr/bin/env python3
"""
Improved main application for Smart Eye-Tracking on Amazon pages.
Tracks gaze, gracefully handles browser closure, and reports only viewed products.
"""

import time
import logging
import numpy as np
import cv2
import config
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from models.eye_tracking_model import EyeTracker
from models.calibration import Calibration
from models.attention_analytics import AttentionAnalytics
from models.dynamic_grid_extractor import extract_amazon_product_regions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchWindowException

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class EyeTrackingSystem:
    """
    Main class for the eye tracking system.
    """
    
    def __init__(self, 
                 camera_index: int = 0,
                 screen_width: int = 1920,
                 screen_height: int = 1080,
                 calibration_duration: float = 1.0,
                 min_calibration_samples: int = 30):
        """
        Initialize the eye tracking system.
        
        Args:
            camera_index: Index of the camera to use
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
            calibration_duration: Duration to collect samples for each calibration point
            min_calibration_samples: Minimum number of samples required per calibration point
        """
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
            
        # Get camera properties
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        
        # Initialize components
        self.eye_tracker = EyeTracker(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1,
            invert_x_gaze=True,
            vertical_sensitivity=1.5,
            smoothing_window=5
        )
        
        self.calibration = Calibration(
            screen_width=screen_width,
            screen_height=screen_height,
            num_horizontal_points=5,
            num_vertical_points=3,
            sample_duration=calibration_duration,
            min_samples=min_calibration_samples
        )
        
        self.analytics = AttentionAnalytics()
        
        # Initialize browser
        self.browser = None
        self.setup_browser()
        
        # State variables
        self.is_running = False
        self.is_calibrating = False
        self.current_session = None
        self.last_gaze_point = None
        self.last_product_check = 0
        self.product_regions = []
        
        logger.info("Eye tracking system initialized")
        
    def setup_browser(self):
        """Setup the Chrome browser with appropriate options."""
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.browser = webdriver.Chrome(options=chrome_options)
        
    def start_calibration(self):
        """Start the calibration process."""
        self.is_calibrating = True
        self.calibration.start_calibration()
        logger.info("Starting calibration process")
        
    def process_calibration(self, frame: np.ndarray) -> bool:
        """
        Process a frame during calibration.
        
        Args:
            frame: Input video frame
            
        Returns:
            bool: True if calibration is still in progress
        """
        if not self.is_calibrating:
            return False
            
        # Process frame with eye tracker
        result = self.eye_tracker.process_frame(frame)
        
        if result["success"] and "gaze_point" in result:
            gaze_x, gaze_y = result["gaze_point"]
            
            # Add sample to calibration
            if not self.calibration.add_sample(gaze_x, gaze_y):
                self.is_calibrating = False
                logger.info("Calibration complete")
                return False
                
        return True
        
    def start_session(self, url: str):
        """
        Start a new eye tracking session.
        
        Args:
            url: URL to load in the browser
        """
        if not self.calibration.is_complete():
            logger.error("Calibration must be completed before starting a session")
            return
            
        # Load URL in browser
        try:
            self.browser.get(url)
            time.sleep(2)  # Wait for page to load
        except Exception as e:
            logger.error(f"Failed to load URL: {e}")
            return
            
        # Extract product regions
        self.product_regions = extract_amazon_product_regions(self.browser)
        
        # Start new session
        self.current_session = {
            "url": url,
            "start_time": time.time(),
            "gaze_points": [],
            "product_fixations": {},
            "scroll_positions": []
        }
        
        self.is_running = True
        logger.info("Started new eye tracking session")
        
    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Process a video frame during a session.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (x, y) coordinates if gaze point is detected, None otherwise
        """
        if not self.is_running:
            return None
            
        # Process frame with eye tracker
        result = self.eye_tracker.process_frame(frame)
        
        if result["success"] and "gaze_point" in result:
            gaze_x, gaze_y = result["gaze_point"]
            
            # Apply calibration
            calibrated_x, calibrated_y = self.calibration.apply_calibration(gaze_x, gaze_y)
            
            # Store gaze point
            self.current_session["gaze_points"].append({
                "timestamp": time.time(),
                "x": calibrated_x,
                "y": calibrated_y,
                "quality": result.get("quality_metrics", {}).get("confidence_score", 0.0)
            })
            
            # Check product regions periodically
            current_time = time.time()
            if current_time - self.last_product_check >= 1.0:  # Check every second
                self._check_product_regions(calibrated_x, calibrated_y)
                self.last_product_check = current_time
                
            return calibrated_x, calibrated_y
            
        return None
        
    def _check_product_regions(self, gaze_x: float, gaze_y: float):
        """Check if gaze point is within any product region."""
        try:
            # Get current scroll position
            scroll_y = self.browser.execute_script("return window.pageYOffset;")
            self.current_session["scroll_positions"].append({
                "timestamp": time.time(),
                "scroll_y": scroll_y
            })
            
            # Adjust gaze Y coordinate for scroll position
            adjusted_y = gaze_y + scroll_y
            
            # Check each product region
            for product_id, region in self.product_regions.items():
                if (region["x"] <= gaze_x <= region["x"] + region["width"] and
                    region["y"] <= adjusted_y <= region["y"] + region["height"]):
                    
                    if product_id not in self.current_session["product_fixations"]:
                        self.current_session["product_fixations"][product_id] = []
                        
                    self.current_session["product_fixations"][product_id].append({
                        "timestamp": time.time(),
                        "duration": 1.0  # Approximate duration
                    })
                    
        except NoSuchWindowException:
            logger.error("Browser window was closed")
            self.stop_session()
        except Exception as e:
            logger.error(f"Error checking product regions: {e}")
            
    def stop_session(self):
        """Stop the current eye tracking session."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Generate analytics
        if self.current_session:
            self.analytics.add_session(self.current_session)
            
        self.current_session = None
        logger.info("Stopped eye tracking session")
        
    def export_report(self, output_path: str):
        """
        Export the analytics report.
        
        Args:
            output_path: Path to save the report
        """
        self.analytics.export_analytics_report(output_path)
        
    def cleanup(self):
        """Clean up resources."""
        if self.camera is not None:
            self.camera.release()
            
        if self.browser is not None:
            try:
                self.browser.quit()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
                
        logger.info("Cleaned up resources")
        
def main():
    """Main function to run the eye tracking system."""
    try:
        # Initialize system
        system = EyeTrackingSystem()
        
        # Start calibration
        system.start_calibration()
        
        # Calibration loop
        while system.is_calibrating:
            ret, frame = system.camera.read()
            if not ret:
                logger.error("Failed to read frame")
                break
                
            # Process calibration
            system.process_calibration(frame)
            
            # Show calibration progress
            progress = system.calibration.get_progress()
            current_point = system.calibration.get_current_point()
            
            if current_point:
                cv2.circle(frame, 
                          (int(current_point[0]), int(current_point[1])), 
                          20, (0, 255, 0), -1)
                
            cv2.putText(frame, f"Calibration: {progress:.1f}%", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
        # Start session
        system.start_session("https://www.amazon.com/s?k=laptop")
        
        # Main loop
        while True:
            ret, frame = system.camera.read()
            if not ret:
                logger.error("Failed to read frame")
                break
                
            # Process frame
            gaze_point = system.process_frame(frame)
            
            # Visualize results
            if gaze_point:
                x, y = map(int, gaze_point)
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                
            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
        # Export report
        system.export_report("analytics_report.html")
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        system.cleanup()
        
if __name__ == "__main__":
    main()