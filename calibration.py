import cv2
import numpy as np
import time
from models.eye_tracking_model import EyeTracker
from utils.camera import Camera

def perform_calibration():
    eye_tracker = EyeTracker()
    camera = Camera()
    if not camera.start():
        print("Camera initialization failed!")
        return

    screen_points = []
    gaze_points = []

    # Define calibration points on the screen (4 corners)
    calibration_screen_points = [
        (100, 100),  # top-left
        (1820, 100),  # top-right (assuming 1920x1080 screen)
        (1820, 980),  # bottom-right
        (100, 980),  # bottom-left
    ]

    for screen_point in calibration_screen_points:
        print(f"Look at the calibration point: {screen_point}")
        time.sleep(2)  # Wait for user to look at the point

        frames_collected = 0
        gaze_collected = []

        while frames_collected < 30:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            result = eye_tracker.process_frame(frame)
            gaze_point = result.get("gaze_point")
            if gaze_point:
                gaze_collected.append(gaze_point)
                frames_collected += 1

            time.sleep(0.05)

        # Take average gaze point for stability
        avg_gaze = np.mean(gaze_collected, axis=0)
        gaze_points.append(avg_gaze)
        screen_points.append(screen_point)

    camera.stop()
    eye_tracker.__del__()

    # Compute homography
    gaze_points = np.array(gaze_points, dtype=np.float32)
    screen_points = np.array(screen_points, dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(gaze_points, screen_points)

    # Save homography matrix
    np.save('calibration_homography.npy', homography_matrix)
    print("Calibration completed and file saved as 'calibration_homography.npy'")

if __name__ == "__main__":
    perform_calibration()