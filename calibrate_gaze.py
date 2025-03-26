import cv2
import numpy as np
import time
from models.eye_tracking_model import EyeTracker  # import your gaze tracker

# Define 5 calibration points in normalized screen coordinates (x, y)
CALIBRATION_POINTS = [
    (0.1, 0.1),   # Top-left
    (0.9, 0.1),   # Top-right
    (0.5, 0.5),   # Center
    (0.1, 0.9),   # Bottom-left
    (0.9, 0.9)    # Bottom-right
]

SAMPLES_PER_POINT = 40         # Total samples collected per calibration point
STABILIZATION_TIME = 1.5       # Seconds to wait before sampling gaze
SAMPLING_DURATION = 2.0        # Seconds to collect gaze data

def run_calibration():
    cap = cv2.VideoCapture(0)
    eye_tracker = EyeTracker()

    screen_w = 1280
    screen_h = 720

    gaze_data = []
    target_data = []

    for point in CALIBRATION_POINTS:
        screen_x = int(point[0] * screen_w)
        screen_y = int(point[1] * screen_h)

        print(f"\nLook at point: {point}")
        print(f"Hold gaze. Stabilizing...")
        start_time = time.time()

        while time.time() - start_time < STABILIZATION_TIME:
            ret, frame = cap.read()
            if not ret:
                break
            # Show calibration dot
            cv2.circle(frame, (screen_x, screen_y), 15, (255, 255, 255), -1)
            cv2.putText(frame, "Stabilizing...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        print("Recording gaze data...")
        all_gaze = []
        sample_start = time.time()

        while time.time() - sample_start < SAMPLING_DURATION:
            ret, frame = cap.read()
            if not ret:
                break

            result = eye_tracker.process_frame(frame)
            if result["success"] and result["gaze_point"]:
                all_gaze.append(result["gaze_point"])

            # Show red dot during data collection
            cv2.circle(frame, (screen_x, screen_y), 15, (0, 0, 255), -1)
            cv2.putText(frame, "Recording...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        if len(all_gaze) >= 10:
            # Optional: remove outliers using median filtering
            all_gaze = np.array(all_gaze)
            med = np.median(all_gaze, axis=0)
            dists = np.linalg.norm(all_gaze - med, axis=1)
            filtered = all_gaze[dists < np.percentile(dists, 80)]  # keep ~80% closest
            avg_gaze = np.mean(filtered, axis=0)

            gaze_data.append(avg_gaze)
            target_data.append([screen_x, screen_y])
            print(f"✅ Collected {len(filtered)} stable samples.")
        else:
            print("⚠️ Not enough gaze data collected for this point. Skipping.")

    cap.release()
    cv2.destroyAllWindows()

    # Fit linear model or homography
    gaze_data = np.array(gaze_data, dtype=np.float32)
    target_data = np.array(target_data, dtype=np.float32)

    if len(gaze_data) >= 4:
        matrix, _ = cv2.findHomography(gaze_data, target_data, method=0)
        np.save("calibration_matrix.npy", matrix)
        print("\n✅ Calibration completed and saved as calibration_matrix.npy")
    else:
        print("\n❌ Not enough calibration points. Try again.")

if __name__ == "__main__":
    run_calibration()