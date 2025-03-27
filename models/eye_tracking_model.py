"""
Eye tracking model using MediaPipe face mesh to detect and track eye movements.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Upper and lower eyelid indices for vertical gaze tracking
    LEFT_UPPER_EYELID = [386, 374, 373, 390, 388, 387]
    LEFT_LOWER_EYELID = [263, 249, 390, 373, 374, 380]
    RIGHT_UPPER_EYELID = [159, 145, 144, 163, 157, 158]
    RIGHT_LOWER_EYELID = [133, 155, 154, 153, 145, 144]
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 max_num_faces: int = 1,
                 invert_x_gaze: bool = False,
                 use_vertical_ratio: bool = True,
                 use_3d_pose: bool = True):
        """
        Initialize the eye tracker.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (default: 0.5)
            min_tracking_confidence: Minimum confidence for face tracking (default: 0.5)
            max_num_faces: Maximum number of faces to track (default: 1)
            invert_x_gaze: Whether to invert horizontal gaze direction (default: False)
            use_vertical_ratio: Whether to use iris-to-eyelid ratio for vertical gaze (default: True)
            use_3d_pose: Whether to use 3D eye pose estimation (default: True)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_faces = max_num_faces
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.last_gaze_point = None
        self.last_detection_time = 0
        self.frame_count = 0
        self.successful_detections = 0
        self.invert_x_gaze = invert_x_gaze
        self.use_vertical_ratio = use_vertical_ratio
        self.use_3d_pose = use_3d_pose
        
        # Calibration data
        self.vertical_calibration = {
            "points": [],
            "ratios": [],
            "is_calibrated": False
        }
        
        # Default vertical adjustment parameters (will be updated during calibration)
        self.vertical_scale = 1.5  # Higher value for more vertical movement
        self.vertical_center = 0.5  # Center point for vertical ratio (0.5 = middle)
        
        logger.info("Enhanced eye tracker initialized")
        
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
            
        self.frame_count += 1
        frame_height, frame_width, _ = frame.shape
        
        # Convert the image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        start_time = time.time()
        results = self.face_mesh.process(rgb_frame)
        process_time = time.time() - start_time
        
        # Prepare output dictionary
        output = {
            "success": False,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "process_time_ms": process_time * 1000,
            "face_detected": False,
            "landmarks": None,
            "left_eye": None,
            "right_eye": None,
            "gaze_point": None,
            "iris_landmarks": None,
            "vertical_ratio": None,
            "3d_gaze_vector": None
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
                [lm.x * frame_width, lm.y * frame_height, lm.z * frame_width]
                for lm in face_landmarks.landmark
            ])
            
            output["landmarks"] = landmarks_array
            
            # Extract eye landmarks
            left_eye_landmarks = landmarks_array[self.LEFT_EYE_INDICES]
            right_eye_landmarks = landmarks_array[self.RIGHT_EYE_INDICES]
            
            # Extract iris landmarks if available
            left_iris = landmarks_array[self.LEFT_IRIS_INDICES]
            right_iris = landmarks_array[self.RIGHT_IRIS_INDICES]
            
            # Calculate eye centers
            left_eye_center = np.mean(left_eye_landmarks[:, :2], axis=0)
            right_eye_center = np.mean(right_eye_landmarks[:, :2], axis=0)
            
            # Calculate iris centers
            left_iris_center = np.mean(left_iris[:, :2], axis=0)
            right_iris_center = np.mean(right_iris[:, :2], axis=0)
            
            # ENHANCEMENT 1: Calculate iris-to-eyelid ratio for vertical gaze
            vertical_ratio = None
            if self.use_vertical_ratio:
                vertical_ratio = self.calculate_vertical_gaze_ratio(landmarks_array, left_iris_center, right_iris_center)
                output["vertical_ratio"] = vertical_ratio
            
            # ENHANCEMENT 2: Calculate 3D eye pose for improved gaze estimation
            gaze_vector_3d = None
            if self.use_3d_pose:
                gaze_vector_3d = self.estimate_3d_gaze(landmarks_array)
                output["3d_gaze_vector"] = gaze_vector_3d.tolist()
            
            # Use iris position to estimate gaze direction
            left_eye_vector = left_iris_center - left_eye_center
            right_eye_vector = right_iris_center - right_eye_center
            
            # Average the two eye vectors for the final gaze vector
            gaze_vector = (left_eye_vector + right_eye_vector) / 2
            
            # Normalize and scale the gaze vector
            gaze_magnitude = np.linalg.norm(gaze_vector)
            if gaze_magnitude > 0:
                normalized_gaze = gaze_vector / gaze_magnitude
            else:
                normalized_gaze = gaze_vector
            
            # Apply horizontal gaze inversion if configured
            if self.invert_x_gaze:
                normalized_gaze[0] = -normalized_gaze[0]
                
            # Adjust vertical gaze component if using vertical ratio
            if vertical_ratio is not None:
                # Scale vertical gaze based on the iris-to-eyelid ratio
                vertical_offset = (vertical_ratio - self.vertical_center) * self.vertical_scale
                normalized_gaze[1] = vertical_offset
            
            # If using 3D pose estimation, incorporate it into the gaze vector
            if gaze_vector_3d is not None:
                # Scale to match 2D vector and blend them
                normalized_gaze = 0.7 * normalized_gaze + 0.3 * gaze_vector_3d[:2]
                # Re-normalize
                gaze_magnitude = np.linalg.norm(normalized_gaze)
                if gaze_magnitude > 0:
                    normalized_gaze = normalized_gaze / gaze_magnitude
            
            # Estimate gaze point on the frame
            gaze_scale = 200  # Arbitrary scale factor, would be calibrated in real use
            center_point = np.array([frame_width/2, frame_height/2])
            gaze_point = center_point + normalized_gaze * gaze_scale
            
            # Ensure gaze point is within frame boundaries
            gaze_point[0] = max(0, min(frame_width, gaze_point[0]))
            gaze_point[1] = max(0, min(frame_height, gaze_point[1]))
            
            # Apply some smoothing with previous gaze points
            if self.last_gaze_point is not None:
                # Simple exponential smoothing
                alpha = 0.3  # Smoothing factor (0 = no smoothing, 1 = no filtering)
                gaze_point = alpha * gaze_point + (1 - alpha) * self.last_gaze_point
                
            self.last_gaze_point = gaze_point
            
            # Store eye and gaze information in output dictionary
            output["left_eye"] = {
                "center": left_eye_center.tolist(),
                "landmarks": left_eye_landmarks.tolist()
            }
            
            output["right_eye"] = {
                "center": right_eye_center.tolist(),
                "landmarks": right_eye_landmarks.tolist()
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
                
        # Calculate detection rate
        output["detection_rate"] = self.successful_detections / self.frame_count if self.frame_count > 0 else 0
        
        return output
    
    def calculate_vertical_gaze_ratio(self, landmarks_array, left_iris_center, right_iris_center):
        """
        Calculate the vertical gaze ratio based on iris position relative to eyelids.
        
        Args:
            landmarks_array: All face landmarks
            left_iris_center: Center coordinates of left iris
            right_iris_center: Center coordinates of right iris
            
        Returns:
            Vertical ratio value (0-1 range where 0.5 is center)
        """
        try:
            # Get upper and lower eyelid y-positions for left eye
            left_upper_points = landmarks_array[self.LEFT_UPPER_EYELID]
            left_lower_points = landmarks_array[self.LEFT_LOWER_EYELID]
            
            left_upper_y = np.mean(left_upper_points[:, 1])
            left_lower_y = np.mean(left_lower_points[:, 1])
            left_iris_y = left_iris_center[1]
            
            # Calculate normalized position of iris between eyelids for left eye
            left_vertical_ratio = (left_iris_y - left_upper_y) / (left_lower_y - left_upper_y)
            
            # Do the same for right eye
            right_upper_points = landmarks_array[self.RIGHT_UPPER_EYELID]
            right_lower_points = landmarks_array[self.RIGHT_LOWER_EYELID]
            
            right_upper_y = np.mean(right_upper_points[:, 1])
            right_lower_y = np.mean(right_lower_points[:, 1])
            right_iris_y = right_iris_center[1]
            
            right_vertical_ratio = (right_iris_y - right_upper_y) / (right_lower_y - right_upper_y)
            
            # Average the two ratios for more stability
            vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2
            
            # Ensure the ratio is within the 0-1 range
            vertical_ratio = max(0.0, min(1.0, vertical_ratio))
            
            return vertical_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating vertical gaze ratio: {e}")
            return 0.5  # Default to center
    
    def estimate_3d_gaze(self, landmarks_array):
        """
        Estimate 3D gaze direction using PCA on eye landmarks.
        
        Args:
            landmarks_array: All face landmarks
            
        Returns:
            3D normalized gaze vector
        """
        try:
            # Extract 3D eye landmarks
            left_eye_3d = landmarks_array[self.LEFT_EYE_INDICES]
            right_eye_3d = landmarks_array[self.RIGHT_EYE_INDICES]
            
            # Calculate iris centers in 3D
            left_iris_3d = np.mean(landmarks_array[self.LEFT_IRIS_INDICES], axis=0)
            right_iris_3d = np.mean(landmarks_array[self.RIGHT_IRIS_INDICES], axis=0)
            
            # Calculate eye centers in 3D
            left_eye_center_3d = np.mean(left_eye_3d, axis=0)
            right_eye_center_3d = np.mean(right_eye_3d, axis=0)
            
            # Calculate eye normal vectors using PCA
            left_pca = PCA(n_components=3)
            left_pca.fit(left_eye_3d)
            left_normal = left_pca.components_[2]  # Third component is normal to the eye plane
            
            right_pca = PCA(n_components=3)
            right_pca.fit(right_eye_3d)
            right_normal = right_pca.components_[2]
            
            # Ensure normal vectors point forward (towards negative z)
            if left_normal[2] > 0:
                left_normal = -left_normal
            if right_normal[2] > 0:
                right_normal = -right_normal
            
            # Calculate 3D gaze vectors (from eye center to iris)
            left_gaze_3d = left_iris_3d - left_eye_center_3d
            right_gaze_3d = right_iris_3d - right_eye_center_3d
            
            # Project gaze vectors onto the eye plane
            left_gaze_proj = left_gaze_3d - np.dot(left_gaze_3d, left_normal) * left_normal
            right_gaze_proj = right_gaze_3d - np.dot(right_gaze_3d, right_normal) * right_normal
            
            # Average the projected gaze vectors
            gaze_3d = (left_gaze_proj + right_gaze_proj) / 2
            
            # Normalize
            gaze_3d_magnitude = np.linalg.norm(gaze_3d)
            if gaze_3d_magnitude > 0:
                gaze_3d_normalized = gaze_3d / gaze_3d_magnitude
            else:
                gaze_3d_normalized = np.array([0.0, 0.0, -1.0])  # Default to looking forward
                
            return gaze_3d_normalized
            
        except Exception as e:
            logger.warning(f"Error estimating 3D gaze: {e}")
            return np.array([0.0, 0.0, -1.0])  # Default to looking forward
    
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
            landmark_list = self.landmarks_to_proto(result["landmarks"], frame.shape[1], frame.shape[0])
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
        
        # Display vertical gaze ratio if available
        if "vertical_ratio" in result and result["vertical_ratio"] is not None:
            vertical_ratio = result["vertical_ratio"]
            cv2.putText(
                annotated_frame,
                f"V-Ratio: {vertical_ratio:.2f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 165, 0),
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
        
    def perform_vertical_calibration(self, screen_points, vertical_ratios):
        """
        Calibrate vertical gaze tracking with collected data points.
        
        Args:
            screen_points: List of y-coordinates of calibration points on screen
            vertical_ratios: List of measured vertical ratios for each point
            
        Returns:
            True if calibration was successful, False otherwise
        """
        if len(screen_points) < 2 or len(vertical_ratios) < 2:
            logger.warning("Not enough calibration points for vertical calibration")
            return False
            
        try:
            # Find the center point (ratio that corresponds to looking at the center)
            # Assuming screen_points are normalized to 0-1 range
            center_index = np.argmin(np.abs(np.array(screen_points) - 0.5))
            self.vertical_center = vertical_ratios[center_index]
            
            # Calculate vertical scale factor
            # Find points looking at top and bottom
            top_index = np.argmin(screen_points)
            bottom_index = np.argmax(screen_points)
            
            top_ratio = vertical_ratios[top_index]
            bottom_ratio = vertical_ratios[bottom_index]
            
            # Calculate scale factor
            if abs(bottom_ratio - top_ratio) > 0.01:  # Avoid division by near-zero
                # Scale to map the ratio difference to full vertical range
                self.vertical_scale = 1.0 / (bottom_ratio - top_ratio)
            else:
                # Default if the difference is too small
                self.vertical_scale = 2.0
                
            logger.info(f"Vertical calibration complete: center={self.vertical_center:.2f}, scale={self.vertical_scale:.2f}")
            
            # Store calibration data
            self.vertical_calibration["points"] = screen_points
            self.vertical_calibration["ratios"] = vertical_ratios
            self.vertical_calibration["is_calibrated"] = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error during vertical calibration: {e}")
            return False
            
    def save_calibration(self, filepath):
        """
        Save calibration data to a file.
        
        Args:
            filepath: Path to save the calibration data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            calibration_data = {
                "vertical_center": self.vertical_center,
                "vertical_scale": self.vertical_scale,
                "vertical_points": self.vertical_calibration["points"],
                "vertical_ratios": self.vertical_calibration["ratios"],
                "is_calibrated": self.vertical_calibration["is_calibrated"]
            }
            
            np.save(filepath, calibration_data)
            logger.info(f"Calibration data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            return False
            
    def load_calibration(self, filepath):
        """
        Load calibration data from a file.
        
        Args:
            filepath: Path to load the calibration data from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Calibration file not found: {filepath}")
                return False
                
            calibration_data = np.load(filepath, allow_pickle=True).item()
            
            self.vertical_center = calibration_data.get("vertical_center", 0.5)
            self.vertical_scale = calibration_data.get("vertical_scale", 1.5)
            
            self.vertical_calibration["points"] = calibration_data.get("vertical_points", [])
            self.vertical_calibration["ratios"] = calibration_data.get("vertical_ratios", [])
            self.vertical_calibration["is_calibrated"] = calibration_data.get("is_calibrated", False)
            
            logger.info(f"Calibration data loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False
    
    def __del__(self):
        """
        Clean up resources.
        """
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
                logger.info("Face mesh resources released")
        except Exception as e:
            logger.warning(f"Error closing face mesh: {e}")