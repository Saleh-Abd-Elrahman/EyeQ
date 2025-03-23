"""
Eye tracking model using MediaPipe face mesh to detect and track eye movements.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

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
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 max_num_faces: int = 1,
                 invert_x_gaze: bool = False):
        """
        Initialize the eye tracker.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (default: 0.5)
            min_tracking_confidence: Minimum confidence for face tracking (default: 0.5)
            max_num_faces: Maximum number of faces to track (default: 1)
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
        if self.invert_x_gaze:
            normalized_gaze[0] = -normalized_gaze[0]  # Invert X direction
            
            
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
            "iris_landmarks": None
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
            
            # Use iris position to estimate gaze direction
            # This is a simplified approach; more sophisticated methods exist
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
            
            # Estimate gaze point on the frame
            # This is very simplified - for actual eye tracking, calibration is needed
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
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except ValueError as e:
                logger.warning(f"Attempted to close already closed face_mesh: {e}")
        logger.info("Eye tracker resources released")