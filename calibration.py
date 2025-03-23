import cv2
import numpy as np
import time
import os
from models.eye_tracking_model import EyeTracker
from utils.camera import Camera

def create_calibration_window(width=1920, height=1080, window_name="Eye Tracking Calibration"):
    """Create a full-screen calibration window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Create a black background
    background = np.zeros((height, width, 3), dtype=np.uint8)
    return background, window_name

def add_text(image, text, position, font_size=1, color=(255, 255, 255), thickness=2):
    """Add text to the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_size, color, thickness, cv2.LINE_AA)
    return image

def draw_calibration_point(image, point, size=20, color=(0, 255, 0), thickness=-1):
    """Draw a calibration point (circle) on the image."""
    cv2.circle(image, point, size, color, thickness)
    # Draw a smaller white dot in the center for better visibility
    cv2.circle(image, point, size//4, (255, 255, 255), -1)
    return image

def perform_calibration():
    """Perform eye tracking calibration with visual UI."""
    # Get screen resolution - default to 1920x1080 if detection fails
    try:
        from screeninfo import get_monitors
        monitor = get_monitors()[0]
        screen_width, screen_height = monitor.width, monitor.height
    except:
        screen_width, screen_height = 1920, 1080
        print("Warning: Could not detect screen resolution, using default 1920x1080")
    
    # Create calibration window
    background, window_name = create_calibration_window(screen_width, screen_height)
    
    # Create camera preview window
    camera_window = "Camera Preview"
    cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(camera_window, 480, 270)  # Smaller window for camera feed
    
    # Initialize eye tracker and camera
    eye_tracker = EyeTracker()
    camera = Camera(horizontal_flip=True)  # Apply horizontal flip for more intuitive experience
    
    if not camera.start():
        print("Camera initialization failed!")
        cv2.destroyAllWindows()
        return

    # Calibration instructions screen
    instructions = background.copy()
    add_text(instructions, "EYE TRACKING CALIBRATION", (screen_width//2 - 400, screen_height//2 - 200), 1.5, (0, 255, 0), 3)
    add_text(instructions, "Instructions:", (screen_width//2 - 350, screen_height//2 - 100), 1, (255, 255, 255), 2)
    add_text(instructions, "1. Follow the green circle with your eyes only", (screen_width//2 - 350, screen_height//2 - 50), 1, (255, 255, 255), 2)
    add_text(instructions, "2. Keep your head relatively still", (screen_width//2 - 350, screen_height//2), 1, (255, 255, 255), 2)
    add_text(instructions, "3. The circle will turn blue when collecting data", (screen_width//2 - 350, screen_height//2 + 50), 1, (255, 255, 255), 2)
    add_text(instructions, "4. The calibration will proceed automatically", (screen_width//2 - 350, screen_height//2 + 100), 1, (255, 255, 255), 2)
    add_text(instructions, "Press SPACEBAR to begin calibration", (screen_width//2 - 350, screen_height//2 + 200), 1.2, (0, 255, 255), 2)
    
    # Display instruction screen
    cv2.imshow(window_name, instructions)
    
    # Wait for spacebar to start
    key = cv2.waitKey(0)
    if key != 32:  # 32 is spacebar
        print("Calibration cancelled.")
        camera.stop()
        cv2.destroyAllWindows()
        return
    
    # Define calibration points - adjust based on screen resolution
    x_margin = int(screen_width * 0.1)
    y_margin = int(screen_height * 0.1)
    x_positions = [x_margin, screen_width // 2, screen_width - x_margin]
    y_positions = [y_margin, screen_height // 2, screen_height - y_margin]
    
    calibration_screen_points = []
    for y in y_positions:
        for x in x_positions:
            calibration_screen_points.append((x, y))
    
    screen_points = []
    gaze_points = []
    
    # Start calibration process
    for i, screen_point in enumerate(calibration_screen_points):
        # Prepare the frame with only the current calibration point
        calibration_frame = background.copy()
        progress_text = f"Point {i+1}/{len(calibration_screen_points)}"
        add_text(calibration_frame, progress_text, (50, 50), 0.8, (255, 255, 255), 2)
        draw_calibration_point(calibration_frame, screen_point)
        
        # Show the point for 1 second before collecting data
        cv2.imshow(window_name, calibration_frame)
        
        # Show camera preview with detected face for feedback
        start_preview_time = time.time()
        while time.time() - start_preview_time < 1.5:
            success, frame = camera.read()
            if success:
                # Just show raw frame for preview
                cv2.imshow(camera_window, frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("Calibration cancelled.")
                camera.stop()
                cv2.destroyAllWindows()
                return
        
        # Prepare for data collection
        frames_collected = 0
        gaze_collected = []
        
        # Change point color to indicate data collection
        collecting_frame = background.copy()
        add_text(collecting_frame, progress_text, (50, 50), 0.8, (255, 255, 255), 2)
        add_text(collecting_frame, "Collecting data...", (50, 90), 0.7, (0, 255, 255), 2)
        draw_calibration_point(collecting_frame, screen_point, color=(0, 0, 255))  # Blue during collection
        cv2.imshow(window_name, collecting_frame)
        
        collection_start_time = time.time()
        target_frames = 15  # Number of frames to collect
        
        # Main collection loop
        while frames_collected < target_frames:
            success, frame = camera.read()
            if not success or frame is None:
                continue
            
            # Process frame for eye tracking
            result = eye_tracker.process_frame(frame)
            
            # Show the processed frame with eye tracking visualization
            vis_frame = eye_tracker.visualize(frame, result, show_gaze=True)
            
            # Update progress bar
            progress = int(50 * frames_collected / target_frames)
            cv2.rectangle(vis_frame, (10, 30), (10 + progress, 40), (0, 255, 0), -1)
            add_text(vis_frame, f"Collecting: {frames_collected}/{target_frames}", (10, 20), 0.5, (255, 255, 255), 1)
            
            cv2.imshow(camera_window, vis_frame)
            
            # Check if we detected gaze
            gaze_point = result.get("gaze_point")
            if gaze_point:
                gaze_collected.append(gaze_point)
                frames_collected += 1
                
                # Update the collecting frame with progress
                progress_frame = collecting_frame.copy()
                progress_percentage = int(100 * frames_collected / target_frames)
                add_text(progress_frame, f"Progress: {progress_percentage}%", (50, 130), 0.7, (0, 255, 255), 2)
                cv2.imshow(window_name, progress_frame)
            
            # Check for cancel
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("Calibration cancelled.")
                camera.stop()
                cv2.destroyAllWindows()
                return
                
            time.sleep(0.05)
            
            # Timeout if taking too long
            if time.time() - collection_start_time > 10:  # 10 second timeout
                if frames_collected > 5:  # Accept if we have at least 5 frames
                    break
                else:
                    print(f"Warning: Timeout collecting data for point {i+1}, retrying...")
                    collection_start_time = time.time()  # Reset timer and try again
        
        # Show success indication
        success_frame = background.copy()
        add_text(success_frame, progress_text, (50, 50), 0.8, (255, 255, 255), 2)
        add_text(success_frame, "Point completed!", (50, 90), 0.7, (0, 255, 0), 2)
        draw_calibration_point(success_frame, screen_point, color=(0, 255, 0))  # Green again
        cv2.imshow(window_name, success_frame)
        cv2.waitKey(300)  # Short delay to show success
        
        # Calculate average gaze point
        if len(gaze_collected) > 0:
            avg_gaze = np.mean(gaze_collected, axis=0)
            gaze_points.append(avg_gaze)
            screen_points.append(screen_point)
            print(f"Completed point {i+1}: {screen_point}")
        else:
            print(f"Failed to collect data for point {i+1}")
    
    # Final cleanup
    camera.stop()
    eye_tracker.__del__()
    
    # If we haven't collected enough points, show an error
    if len(gaze_points) < 4:
        error_screen = background.copy()
        add_text(error_screen, "CALIBRATION FAILED", (screen_width//2 - 300, screen_height//2 - 50), 1.5, (0, 0, 255), 3)
        add_text(error_screen, "Not enough data points collected.", (screen_width//2 - 300, screen_height//2 + 50), 1, (255, 255, 255), 2)
        add_text(error_screen, "Please try again in better lighting conditions.", (screen_width//2 - 300, screen_height//2 + 100), 1, (255, 255, 255), 2)
        add_text(error_screen, "Press any key to exit", (screen_width//2 - 300, screen_height//2 + 200), 1, (0, 255, 255), 2)
        cv2.imshow(window_name, error_screen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    # Compute homography
    gaze_points = np.array(gaze_points, dtype=np.float32)
    screen_points = np.array(screen_points, dtype=np.float32)
    
    homography_matrix, _ = cv2.findHomography(gaze_points, screen_points)
    
    # Save calibration data
    calibration_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_file = os.path.join(calibration_dir, 'calibration_homography.npy')
    np.save(calibration_file, homography_matrix)
    
    # Show success screen
    success_screen = background.copy()
    add_text(success_screen, "CALIBRATION SUCCESSFUL!", (screen_width//2 - 350, screen_height//2 - 100), 1.5, (0, 255, 0), 3)
    add_text(success_screen, f"Calibration data saved to: {calibration_file}", (screen_width//2 - 350, screen_height//2), 0.8, (255, 255, 255), 2)
    add_text(success_screen, "Your eye tracking should now be more accurate.", (screen_width//2 - 350, screen_height//2 + 50), 1, (255, 255, 255), 2)
    add_text(success_screen, "Press any key to exit", (screen_width//2 - 350, screen_height//2 + 150), 1, (0, 255, 255), 2)
    
    cv2.imshow(window_name, success_screen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nCalibration completed successfully!")
    print(f"Calibration data saved as: {calibration_file}")

if __name__ == "__main__":
    perform_calibration()