"""
Calibration module for eye tracking system.
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalibrationPoint:
    """Represents a calibration point on the screen."""
    x: float
    y: float
    is_vertical: bool = False
    collected_samples: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        self.collected_samples = []

class Calibration:
    """
    Handles the calibration process for eye tracking.
    """
    
    def __init__(self, 
                 screen_width: int,
                 screen_height: int,
                 num_horizontal_points: int = 5,
                 num_vertical_points: int = 3,
                 sample_duration: float = 1.0,
                 min_samples: int = 30):
        """
        Initialize calibration.
        
        Args:
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
            num_horizontal_points: Number of horizontal calibration points
            num_vertical_points: Number of vertical calibration points
            sample_duration: Duration to collect samples for each point
            min_samples: Minimum number of samples required per point
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_horizontal_points = num_horizontal_points
        self.num_vertical_points = num_vertical_points
        self.sample_duration = sample_duration
        self.min_samples = min_samples
        
        self.points: List[CalibrationPoint] = []
        self.current_point_index = 0
        self.samples_start_time = 0
        self.is_calibrating = False
        self.calibration_matrix = None
        
        self._generate_calibration_points()
        
    def _generate_calibration_points(self):
        """Generate calibration points on the screen."""
        # Generate horizontal points
        x_spacing = self.screen_width / (self.num_horizontal_points + 1)
        for i in range(self.num_horizontal_points):
            x = (i + 1) * x_spacing
            y = self.screen_height / 2
            self.points.append(CalibrationPoint(x, y))
            
        # Generate vertical points
        y_spacing = self.screen_height / (self.num_vertical_points + 1)
        for i in range(self.num_vertical_points):
            x = self.screen_width / 2
            y = (i + 1) * y_spacing
            self.points.append(CalibrationPoint(x, y, is_vertical=True))
            
        # Add corner points for better calibration
        self.points.append(CalibrationPoint(0, 0))
        self.points.append(CalibrationPoint(self.screen_width, 0))
        self.points.append(CalibrationPoint(0, self.screen_height))
        self.points.append(CalibrationPoint(self.screen_width, self.screen_height))
        
    def start_calibration(self):
        """Start the calibration process."""
        self.is_calibrating = True
        self.current_point_index = 0
        self.samples_start_time = time.time()
        logger.info("Starting calibration process")
        
    def add_sample(self, gaze_x: float, gaze_y: float) -> bool:
        """
        Add a gaze sample for the current calibration point.
        
        Args:
            gaze_x: X coordinate of the gaze point
            gaze_y: Y coordinate of the gaze point
            
        Returns:
            bool: True if sample was added, False if calibration is complete
        """
        if not self.is_calibrating:
            return False
            
        current_point = self.points[self.current_point_index]
        current_point.collected_samples.append((gaze_x, gaze_y))
        
        # Check if we've collected enough samples for this point
        elapsed_time = time.time() - self.samples_start_time
        if elapsed_time >= self.sample_duration and len(current_point.collected_samples) >= self.min_samples:
            # Move to next point
            self.current_point_index += 1
            if self.current_point_index >= len(self.points):
                self.is_calibrating = False
                self._calculate_calibration_matrix()
                return False
                
            self.samples_start_time = time.time()
            
        return True
        
    def _calculate_calibration_matrix(self):
        """
        Calculate the calibration matrix from collected samples.
        """
        # Prepare input and output points for calibration
        input_points = []
        output_points = []
        
        for point in self.points:
            if len(point.collected_samples) >= self.min_samples:
                # Calculate mean gaze point for this calibration point
                mean_gaze = np.mean(point.collected_samples, axis=0)
                input_points.append(mean_gaze)
                output_points.append([point.x, point.y])
                
        if len(input_points) < 4:
            logger.error("Not enough valid calibration points")
            return
            
        # Convert to numpy arrays
        input_points = np.array(input_points, dtype=np.float32)
        output_points = np.array(output_points, dtype=np.float32)
        
        # Calculate homography matrix
        self.calibration_matrix, mask = cv2.findHomography(
            input_points, output_points, cv2.RANSAC, 5.0
        )
        
        if self.calibration_matrix is None:
            logger.error("Failed to calculate calibration matrix")
            return
            
        logger.info("Calibration matrix calculated successfully")
        
    def apply_calibration(self, gaze_x: float, gaze_y: float) -> Tuple[float, float]:
        """
        Apply calibration to a gaze point.
        
        Args:
            gaze_x: Raw gaze X coordinate
            gaze_y: Raw gaze Y coordinate
            
        Returns:
            Tuple of calibrated (x, y) coordinates
        """
        if self.calibration_matrix is None:
            return gaze_x, gaze_y
            
        # Convert gaze point to homogeneous coordinates
        gaze_point = np.array([[gaze_x], [gaze_y], [1.0]], dtype=np.float32)
        
        # Apply calibration matrix
        calibrated_point = self.calibration_matrix @ gaze_point
        
        # Convert back to Cartesian coordinates
        return (calibrated_point[0, 0] / calibrated_point[2, 0],
                calibrated_point[1, 0] / calibrated_point[2, 0])
                
    def get_current_point(self) -> Optional[Tuple[float, float]]:
        """
        Get the current calibration point coordinates.
        
        Returns:
            Tuple of (x, y) coordinates or None if calibration is complete
        """
        if not self.is_calibrating or self.current_point_index >= len(self.points):
            return None
            
        point = self.points[self.current_point_index]
        return (point.x, point.y)
        
    def get_progress(self) -> float:
        """
        Get the calibration progress as a percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if not self.is_calibrating:
            return 100.0
            
        return (self.current_point_index / len(self.points)) * 100.0
        
    def is_complete(self) -> bool:
        """
        Check if calibration is complete.
        
        Returns:
            bool: True if calibration is complete
        """
        return not self.is_calibrating and self.calibration_matrix is not None
        
    def get_quality_metrics(self) -> Dict[str, float]:
        """
        Calculate quality metrics for the calibration.
        
        Returns:
            Dictionary of quality metrics
        """
        if not self.is_complete():
            return {}
            
        metrics = {
            "num_points": len(self.points),
            "valid_points": sum(1 for p in self.points if len(p.collected_samples) >= self.min_samples),
            "mean_samples_per_point": np.mean([len(p.collected_samples) for p in self.points]),
            "std_samples_per_point": np.std([len(p.collected_samples) for p in self.points])
        }
        
        # Calculate reprojection error
        if self.calibration_matrix is not None:
            errors = []
            for point in self.points:
                if len(point.collected_samples) >= self.min_samples:
                    mean_gaze = np.mean(point.collected_samples, axis=0)
                    calibrated_x, calibrated_y = self.apply_calibration(*mean_gaze)
                    error = np.sqrt((calibrated_x - point.x)**2 + (calibrated_y - point.y)**2)
                    errors.append(error)
                    
            metrics["mean_reprojection_error"] = np.mean(errors)
            metrics["std_reprojection_error"] = np.std(errors)
            
        return metrics 