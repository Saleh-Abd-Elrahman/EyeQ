"""
Eye tracking model using MediaPipe face mesh to detect and track eye movements.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GazeQualityMetrics:
    """Metrics for evaluating gaze tracking quality."""
    vertical_stability: float = 0.0
    vertical_range: float = 0.0
    vertical_consistency: float = 0.0
    head_angle: float = 0.0
    confidence_score: float = 0.0

class EyeTracker:
    """
    Eye tracker using MediaPipe face mesh to detect and track eye movements.
    """
    
    # MediaPipe face mesh eye landmark indices
    # For reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS_INDICES = [474, 475, 476, 477]
    RIGHT_IRIS_INDICES = [469, 470, 471, 472]
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 max_num_faces: int = 1,
                 invert_x_gaze: bool = True,
                 vertical_sensitivity: float = 1.5,
                 smoothing_window: int = 5):
        """
        Initialize the eye tracker.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            max_num_faces: Maximum number of faces to track
            invert_x_gaze: Whether to invert horizontal gaze direction
            vertical_sensitivity: Sensitivity multiplier for vertical gaze
            smoothing_window: Number of frames to use for gaze smoothing
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_faces = max_num_faces
        self.invert_x_gaze = invert_x_gaze
        self.vertical_sensitivity = vertical_sensitivity
        self.smoothing_window = smoothing_window
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize frame dimensions
        self.frame_width = 0
        self.frame_height = 0
        
        # Initialize gaze tracking variables
        self.last_gaze_point = None
        self.last_detection_time = 0
        self.frame_count = 0
        self.successful_detections = 0
        
        # Initialize gaze smoothing
        self.gaze_history = deque(maxlen=smoothing_window)
        self.vertical_history = deque(maxlen=smoothing_window)
        
        # Initialize quality metrics
        self.quality_metrics = GazeQualityMetrics()
        self.last_face_angle = None
        
        logger.info("Eye tracker initialized")
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a video frame to detect and track eye movements.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Dict containing detection results and gaze information
        """
        if frame is None:
            return {
                "success": False,
                "error": "Invalid frame input"
            }
            
        # Update frame dimensions
        self.frame_height, self.frame_width, _ = frame.shape
            
        self.frame_count += 1
        
        # Convert the image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        start_time = time.time()
        results = self.face_mesh.process(rgb_frame)
        process_time = time.time() - start_time
        
        # Prepare output dictionary
        output = {
            "success": False,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "process_time_ms": process_time * 1000,
            "face_detected": False,
            "landmarks": None,
            "left_eye": None,
            "right_eye": None,
            "gaze_point": None,
            "iris_landmarks": None,
            "quality_metrics": None
        }
        
        # Check if face detection was successful
        if results.multi_face_landmarks:
            self.successful_detections += 1
            self.last_detection_time = time.time()
            output["success"] = True
            output["face_detected"] = True
            
            # We only process the first face if multiple faces are detected
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array for easier processing
            landmarks_array = np.array([
                [lm.x * self.frame_width, lm.y * self.frame_height, lm.z * self.frame_width]
                for lm in face_landmarks.landmark
            ])
            
            output["landmarks"] = landmarks_array
            
            # Extract eye landmarks
            left_eye_landmarks = landmarks_array[self.LEFT_EYE_INDICES]
            right_eye_landmarks = landmarks_array[self.RIGHT_EYE_INDICES]
            
            # Extract iris landmarks if available
            left_iris = landmarks_array[self.LEFT_IRIS_INDICES]
            right_iris = landmarks_array[self.RIGHT_IRIS_INDICES]
            
            # Calculate eye centers and dimensions
            left_eye_center = np.mean(left_eye_landmarks[:, :2], axis=0)
            right_eye_center = np.mean(right_eye_landmarks[:, :2], axis=0)
            
            # Calculate eye dimensions for vertical ratio
            left_eye_height = np.max(left_eye_landmarks[:, 1]) - np.min(left_eye_landmarks[:, 1])
            right_eye_height = np.max(right_eye_landmarks[:, 1]) - np.min(right_eye_landmarks[:, 1])
            
            # Calculate iris centers
            left_iris_center = np.mean(left_iris[:, :2], axis=0)
            right_iris_center = np.mean(right_iris[:, :2], axis=0)
            
            # Calculate face orientation
            face_angle = self._calculate_face_angle(landmarks_array)
            
            # Calculate gaze vectors relative to eye centers
            # Note: We swap x and y components to fix the axis mapping
            left_eye_vector = np.array([
                left_iris_center[1] - left_eye_center[1],  # Use y for horizontal
                left_iris_center[0] - left_eye_center[0],  # Use x for vertical
                0
            ])
            right_eye_vector = np.array([
                right_iris_center[1] - right_eye_center[1],  # Use y for horizontal
                right_iris_center[0] - right_eye_center[0],  # Use x for vertical
                0
            ])
            
            # Calculate vertical ratios
            left_vertical_ratio = (left_iris_center[0] - left_eye_center[0]) / left_eye_height
            right_vertical_ratio = (right_iris_center[0] - right_eye_center[0]) / right_eye_height
            
            # Average the two eye vectors for the final gaze vector
            gaze_vector = (left_eye_vector + right_eye_vector) / 2
            
            # Normalize the gaze vector
            gaze_magnitude = np.linalg.norm(gaze_vector)
            if gaze_magnitude > 0:
                gaze_vector = gaze_vector / gaze_magnitude
            
            # Apply vertical sensitivity adjustment
            gaze_vector[1] *= self.vertical_sensitivity
            
            # Compensate for head position
            gaze_vector = self._compensate_head_position(gaze_vector, face_angle)
            
            # Estimate gaze point on the frame
            gaze_scale = 200  # Arbitrary scale factor, would be calibrated in real use
            center_point = np.array([self.frame_width/2, self.frame_height/2])
            
            # Map gaze vector to screen coordinates
            # Invert X direction if needed
            if self.invert_x_gaze:
                gaze_vector[0] = -gaze_vector[0]
                
            # Note: We swap x and y back for screen coordinates
            gaze_point = center_point + np.array([gaze_vector[0], gaze_vector[1]]) * gaze_scale
            
            # Ensure gaze point is within frame boundaries
            gaze_point[0] = max(0, min(self.frame_width, gaze_point[0]))
            gaze_point[1] = max(0, min(self.frame_height, gaze_point[1]))
            
            # Apply smoothing
            gaze_point = self._smooth_gaze(gaze_point)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_gaze_quality(gaze_point, face_angle)
            output["quality_metrics"] = quality_metrics
            
            # Store eye and gaze information in output dictionary
            output["left_eye"] = {
                "center": left_eye_center.tolist(),
                "landmarks": left_eye_landmarks.tolist(),
                "height": float(left_eye_height),
                "vertical_ratio": float(left_vertical_ratio)
            }
            
            output["right_eye"] = {
                "center": right_eye_center.tolist(),
                "landmarks": right_eye_landmarks.tolist(),
                "height": float(right_eye_height),
                "vertical_ratio": float(right_vertical_ratio)
            }
            
            output["iris_landmarks"] = {
                "left": left_iris.tolist(),
                "right": right_iris.tolist()
            }
            
            output["gaze_point"] = gaze_point.tolist()
            
        else:
            # No face detected
            logger.debug("No face detected in frame")
            
            # If we haven't seen a face for a while, reset the last gaze point
            if time.time() - self.last_detection_time > 1.0:
                self.last_gaze_point = None
                self.gaze_history.clear()
                self.vertical_history.clear()
                
        # Calculate detection rate
        output["detection_rate"] = self.successful_detections / self.frame_count if self.frame_count > 0 else 0
        
        return output
    
    def _calculate_face_angle(self, landmarks: np.ndarray) -> float:
        """Calculate the face angle from landmarks."""
        # Use nose and eyes to estimate face angle
        nose_tip = landmarks[5]  # Nose tip landmark
        left_eye = np.mean(landmarks[self.LEFT_EYE_INDICES], axis=0)
        right_eye = np.mean(landmarks[self.RIGHT_EYE_INDICES], axis=0)
        
        # Calculate angle between eyes and nose
        eye_center = (left_eye + right_eye) / 2
        face_vector = nose_tip - eye_center
        face_angle = np.arctan2(face_vector[1], face_vector[0])
        
        return face_angle
    
    def _compensate_head_position(self, gaze_vector: np.ndarray, face_angle: float) -> np.ndarray:
        """
        Compensate gaze vector for head position.
        
        Args:
            gaze_vector: 3D gaze vector (x, y, z)
            face_angle: Face angle in radians
            
        Returns:
            Compensated 3D gaze vector
        """
        # Ensure gaze_vector is 3D
        if len(gaze_vector) == 2:
            gaze_vector = np.append(gaze_vector, 0.0)
            
        # Create rotation matrix for compensation
        # Note: We only rotate around the Z axis (yaw) for head position compensation
        compensation_matrix = np.array([
            [np.cos(face_angle), -np.sin(face_angle), 0],
            [np.sin(face_angle), np.cos(face_angle), 0],
            [0, 0, 1]
        ])
        
        # Apply compensation
        compensated_vector = compensation_matrix @ gaze_vector
        
        # Normalize the compensated vector
        magnitude = np.linalg.norm(compensated_vector)
        if magnitude > 0:
            compensated_vector = compensated_vector / magnitude
            
        return compensated_vector
    
    def _smooth_gaze(self, gaze_point: np.ndarray) -> np.ndarray:
        """Apply smoothing to gaze point."""
        self.gaze_history.append(gaze_point)
        
        if len(self.gaze_history) < 2:
            return gaze_point
            
        # Apply weighted moving average
        weights = np.linspace(0.1, 0.3, len(self.gaze_history))
        weights = weights / np.sum(weights)
        
        smoothed_point = np.average(self.gaze_history, weights=weights, axis=0)
        return smoothed_point
    
    def _calculate_gaze_quality(self, gaze_point: np.ndarray, face_angle: float) -> Dict[str, float]:
        """
        Calculate quality metrics for gaze tracking.
        
        Args:
            gaze_point: Current gaze point coordinates
            face_angle: Current face angle
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = GazeQualityMetrics()
        
        # Calculate vertical stability
        if len(self.vertical_history) > 1:
            vertical_std = np.std(self.vertical_history)
            metrics.vertical_stability = 1.0 / (1.0 + vertical_std)
        
        # Calculate vertical range
        if self.frame_height > 0:
            metrics.vertical_range = min(1.0, 
                abs(gaze_point[1] - self.frame_height/2) / (self.frame_height/2))
        
        # Calculate head angle influence
        metrics.head_angle = abs(face_angle)
        
        # Calculate overall confidence score
        metrics.confidence_score = (
            metrics.vertical_stability * 0.4 +
            metrics.vertical_range * 0.3 +
            (1.0 - min(metrics.head_angle / np.pi, 1.0)) * 0.3
        )
        
        return asdict(metrics)
    
    def visualize(self, frame: np.ndarray, result: Dict[str, Any], 
                  show_mesh: bool = False, 
                  show_gaze: bool = True) -> np.ndarray:
        """
        Visualize the eye tracking results on the frame.
        
        Args:
            frame: Input video frame
            result: Result dictionary from process_frame
            show_mesh: Whether to show the full face mesh
            show_gaze: Whether to show the gaze point
            
        Returns:
            Annotated frame
        """
        if not result["success"]:
            return frame
            
        annotated_frame = frame.copy()
        
        if show_mesh and "landmarks" in result and result["landmarks"] is not None:
            landmark_list = self.landmarks_to_proto(result["landmarks"], self.frame_width, self.frame_height)
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=landmark_list,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Draw eye contours
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=landmark_list,
                connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=landmark_list,
                connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Draw eye centers
        if "left_eye" in result and result["left_eye"] is not None:
            left_center = tuple(map(int, result["left_eye"]["center"]))
            cv2.circle(annotated_frame, left_center, 5, (0, 255, 0), -1)
            
        if "right_eye" in result and result["right_eye"] is not None:
            right_center = tuple(map(int, result["right_eye"]["center"]))
            cv2.circle(annotated_frame, right_center, 5, (0, 255, 0), -1)
            
        # Draw gaze point
        if show_gaze and "gaze_point" in result and result["gaze_point"] is not None:
            gaze_point = tuple(map(int, result["gaze_point"]))
            cv2.circle(annotated_frame, gaze_point, 10, (0, 0, 255), -1)
            
            # Draw gaze line if both eyes are detected
            if "left_eye" in result and "right_eye" in result:
                left_center = tuple(map(int, result["left_eye"]["center"]))
                right_center = tuple(map(int, result["right_eye"]["center"]))
                eye_center = ((left_center[0] + right_center[0]) // 2, 
                              (left_center[1] + right_center[1]) // 2)
                cv2.line(annotated_frame, eye_center, gaze_point, (255, 0, 0), 2)
                
            # Draw quality metrics if available
            if "quality_metrics" in result:
                metrics = result["quality_metrics"]
                cv2.putText(
                    annotated_frame,
                    f"Quality: {metrics['confidence_score']:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if metrics['confidence_score'] > 0.7 else (0, 165, 255),
                    2
                )
        
        return annotated_frame
    
    def landmarks_to_proto(self, landmarks: np.ndarray, width: int, height: int) -> Any:
        """
        Convert numpy landmarks to MediaPipe's protocol buffer format for visualization.
        
        Args:
            landmarks: Numpy array of landmarks
            width: Frame width
            height: Frame height
            
        Returns:
            MediaPipe landmark list proto
        """
        from mediapipe.framework.formats import landmark_pb2
    
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        
        for landmark in landmarks:
            x, y, z = landmark
            landmark_proto = landmark_list.landmark.add()
            landmark_proto.x = x / width
            landmark_proto.y = y / height
            landmark_proto.z = z / width
            
        return landmark_list
    
    def __del__(self):
        """
        Clean up resources.
        """
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.face_mesh.close()
            logger.info("Eye tracker resources released")
        except Exception as e:
            logger.warning(f"Error cleaning up eye tracker resources: {e}")