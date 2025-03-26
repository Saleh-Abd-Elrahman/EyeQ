import cv2
import numpy as np
import mediapipe as mp
import time
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyeTracker:
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS_INDICES = [474, 475, 476, 477]
    RIGHT_IRIS_INDICES = [469, 470, 471, 472]

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.last_gaze_point = None
        self.last_detection_time = 0
        self.frame_count = 0
        self.successful_detections = 0

        self.baseline_aperture = 6.0
        self.aperture_weight = 0.5  # Adjustable weight
        self.gaze_scale = 200       # Adjustable scale for sensitivity

        try:
            self.calibration_matrix = np.load("calibration_matrix.npy")
            logger.info("Loaded calibration matrix.")
        except:
            self.calibration_matrix = None

        logger.info("Eye tracker initialized")

    def compute_aperture(self, eye_landmarks: np.ndarray) -> float:
        top_idx, bottom_idx = 12, 4  # rough top and bottom
        return np.linalg.norm(eye_landmarks[top_idx][:2] - eye_landmarks[bottom_idx][:2])

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None:
            return {"success": False, "error": "Invalid frame input"}

        self.frame_count += 1
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        results = self.face_mesh.process(rgb_frame)
        process_time = time.time() - start_time

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

        if results.multi_face_landmarks:
            self.successful_detections += 1
            self.last_detection_time = time.time()
            output["success"] = True
            output["face_detected"] = True

            face_landmarks = results.multi_face_landmarks[0]
            landmarks_array = np.array([[lm.x * frame_width, lm.y * frame_height, lm.z * frame_width] for lm in face_landmarks.landmark])
            output["landmarks"] = landmarks_array

            left_eye_landmarks = landmarks_array[self.LEFT_EYE_INDICES]
            right_eye_landmarks = landmarks_array[self.RIGHT_EYE_INDICES]
            left_iris = landmarks_array[self.LEFT_IRIS_INDICES]
            right_iris = landmarks_array[self.RIGHT_IRIS_INDICES]

            left_eye_center = np.mean(left_eye_landmarks[:, :2], axis=0)
            right_eye_center = np.mean(right_eye_landmarks[:, :2], axis=0)
            left_iris_center = np.mean(left_iris[:, :2], axis=0)
            right_iris_center = np.mean(right_iris[:, :2], axis=0)

            left_eye_vector = left_iris_center - left_eye_center
            right_eye_vector = right_iris_center - right_eye_center

            gaze_vector = (left_eye_vector + right_eye_vector) / 2

            left_aperture = self.compute_aperture(left_eye_landmarks)
            right_aperture = self.compute_aperture(right_eye_landmarks)
            avg_aperture = (left_aperture + right_aperture) / 2
            aperture_adjustment = self.aperture_weight * (self.baseline_aperture - avg_aperture)
            gaze_vector[1] += aperture_adjustment

            gaze_magnitude = np.linalg.norm(gaze_vector)
            normalized_gaze = gaze_vector / gaze_magnitude if gaze_magnitude > 0 else gaze_vector

            center_point = np.array([frame_width / 2, frame_height / 2])
            gaze_point = center_point + normalized_gaze * self.gaze_scale

            if self.calibration_matrix is not None:
                homog = np.array([gaze_point[0], gaze_point[1], 1.0])
                mapped = self.calibration_matrix @ homog
                mapped /= mapped[2]
                gaze_point = mapped[:2]

            gaze_point = np.clip(gaze_point, [0, 0], [frame_width, frame_height])

            if self.last_gaze_point is not None:
                alpha = 0.3
                gaze_point = alpha * gaze_point + (1 - alpha) * self.last_gaze_point

            self.last_gaze_point = gaze_point

            output["left_eye"] = {"center": left_eye_center.tolist(), "landmarks": left_eye_landmarks.tolist()}
            output["right_eye"] = {"center": right_eye_center.tolist(), "landmarks": right_eye_landmarks.tolist()}
            output["iris_landmarks"] = {"left": left_iris.tolist(), "right": right_iris.tolist()}
            output["gaze_point"] = gaze_point.tolist()

        elif time.time() - self.last_detection_time > 1.0:
            self.last_gaze_point = None

        output["detection_rate"] = self.successful_detections / self.frame_count if self.frame_count > 0 else 0
        return output

    def visualize(self, frame: np.ndarray, result: Dict[str, Any], show_mesh: bool = False, show_gaze: bool = True) -> np.ndarray:
        if not result["success"]:
            return frame

        annotated_frame = frame.copy()

        if show_gaze and result.get("gaze_point"):
            gaze_point = tuple(map(int, result["gaze_point"]))
            cv2.circle(annotated_frame, gaze_point, 10, (0, 0, 255), -1)

        if result.get("left_eye"):
            left_center = tuple(map(int, result["left_eye"]["center"]))
            cv2.circle(annotated_frame, left_center, 5, (0, 255, 0), -1)

        if result.get("right_eye"):
            right_center = tuple(map(int, result["right_eye"]["center"]))
            cv2.circle(annotated_frame, right_center, 5, (0, 255, 0), -1)

        if show_gaze and result.get("left_eye") and result.get("right_eye"):
            left_center = tuple(map(int, result["left_eye"]["center"]))
            right_center = tuple(map(int, result["right_eye"]["center"]))
            eye_center = ((left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2)
            cv2.line(annotated_frame, eye_center, gaze_point, (255, 0, 0), 2)

        return annotated_frame

    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("Eye tracker resources released")
