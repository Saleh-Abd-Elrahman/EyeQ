import cv2
import numpy as np
import time
import os
import argparse
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

def perform_standard_calibration():
    """Perform standard homography-based calibration for screen mapping."""
    print("\n*** STANDARD CALIBRATION (HOMOGRAPHY) ***")
    print("This calibration maps your gaze to screen coordinates.")
    print("Look at each calibration point when instructed.")
    
    eye_tracker = EyeTracker()
    camera = Camera()
    if not camera.start():
        print("Camera initialization failed!")
        return

    screen_points = []
    gaze_points = []

    # Define calibration points on the screen (4 corners and center)
    calibration_screen_points = [
        (100, 100),       # top-left
        (1820, 100),      # top-right (assuming 1920x1080 screen)
        (1820, 980),      # bottom-right
        (100, 980),       # bottom-left
        (960, 540),       # center
    ]

    # Create a window for calibration
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    calibration_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    for point_idx, screen_point in enumerate(calibration_screen_points):
        # Draw current calibration point
        calibration_image.fill(0)
        cv2.circle(calibration_image, screen_point, 30, (0, 255, 0), -1)
        cv2.putText(calibration_image, f"Point {point_idx+1}/{len(calibration_screen_points)}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Calibration", calibration_image)
        cv2.waitKey(1)
        
        print(f"\nLook at the green dot (Point {point_idx+1})")
        time.sleep(1)  # Give user time to focus
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Collecting in {i}...")
            time.sleep(1)
        
        print("Collecting... Keep looking at the green dot!")

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
                print(f"Collected {frames_collected}/30 samples", end="\r")

            # Show the camera view with tracking visualization
            vis_frame = eye_tracker.visualize(frame, result)
            vis_frame_resized = cv2.resize(vis_frame, (640, 360))
            cv2.imshow("Camera Feed", vis_frame_resized)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                camera.stop()
                return

            time.sleep(0.05)

        # Take average gaze point for stability
        avg_gaze = np.mean(gaze_collected, axis=0)
        gaze_points.append(avg_gaze)
        screen_points.append(screen_point)
        
        print(f"\nPoint {point_idx+1} complete: {avg_gaze}")

    # Close visualization windows
    cv2.destroyAllWindows()
    camera.stop()

    # Compute homography
    gaze_points = np.array(gaze_points, dtype=np.float32)
    screen_points = np.array(screen_points, dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(gaze_points, screen_points)

    # Save homography matrix
    np.save('calibration_homography.npy', homography_matrix)
    print("Calibration completed and file saved as 'calibration_homography.npy'")
    
def perform_vertical_calibration():
    """Perform enhanced vertical calibration to improve up/down gaze tracking."""
    print("\n*** VERTICAL GAZE CALIBRATION ***")
    print("This calibration improves vertical (up/down) gaze tracking.")
    print("Look at each vertical point when instructed.")
    
    eye_tracker = EyeTracker(use_vertical_ratio=True, use_3d_pose=True)
    camera = Camera()
    if not camera.start():
        print("Camera initialization failed!")
        return
    
    screen_height = 1080  # Assuming 1080p screen
    screen_width = 1920
    
    # Define vertical calibration points (at screen center, vary vertically)
    center_x = screen_width // 2
    vertical_points = [
        (center_x, 100),             # Top
        (center_x, screen_height//4), # Upper quarter
        (center_x, screen_height//2), # Middle
        (center_x, 3*screen_height//4), # Lower quarter
        (center_x, screen_height-100) # Bottom
    ]
    
    # Normalize y-coordinates to 0-1 range (0=top, 1=bottom)
    normalized_points = [y/screen_height for _, y in vertical_points]
    
    # Create a window for calibration
    cv2.namedWindow("Vertical Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Vertical Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    calibration_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # For storing calibration data
    vertical_ratios = []
    
    for point_idx, (screen_x, screen_y) in enumerate(vertical_points):
        # Draw current calibration point
        calibration_image.fill(0)
        cv2.circle(calibration_image, (screen_x, screen_y), 30, (0, 255, 0), -1)
        cv2.line(calibration_image, (screen_x - 50, screen_y), (screen_x + 50, screen_y), (0, 255, 0), 2)
        cv2.putText(calibration_image, f"Vertical Point {point_idx+1}/{len(vertical_points)}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Vertical Calibration", calibration_image)
        cv2.waitKey(1)
        
        print(f"\nLook at the green dot (Vertical Point {point_idx+1})")
        time.sleep(1)  # Give user time to focus
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Collecting in {i}...")
            time.sleep(1)
        
        print("Collecting... Keep looking at the green dot!")

        frames_collected = 0
        ratios_collected = []

        while frames_collected < 30:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            result = eye_tracker.process_frame(frame)
            
            if result["success"] and "vertical_ratio" in result and result["vertical_ratio"] is not None:
                ratios_collected.append(result["vertical_ratio"])
                frames_collected += 1
                print(f"Collected {frames_collected}/30 vertical samples", end="\r")

            # Show the camera view with tracking visualization
            vis_frame = eye_tracker.visualize(frame, result)
            vis_frame_resized = cv2.resize(vis_frame, (640, 360))
            cv2.imshow("Camera Feed", vis_frame_resized)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                camera.stop()
                return

            time.sleep(0.05)

        # Take median vertical ratio for stability (median is more robust to outliers)
        median_ratio = np.median(ratios_collected)
        vertical_ratios.append(median_ratio)
        
        print(f"\nPoint {point_idx+1} complete: vertical ratio = {median_ratio:.4f}")

    # Close visualization windows
    cv2.destroyAllWindows()
    camera.stop()
    
    # Perform calibration
    calibration_success = eye_tracker.perform_vertical_calibration(normalized_points, vertical_ratios)
    
    if calibration_success:
        # Save calibration data
        eye_tracker.save_calibration('vertical_calibration.npy')
        print("Vertical calibration completed and saved as 'vertical_calibration.npy'")
    else:
        print("Vertical calibration failed. Please try again.")

def test_calibration():
    """Test the calibration by showing live gaze tracking on screen."""
    print("\n*** TESTING CALIBRATION ***")
    
    homography_file = 'calibration_homography.npy'
    vertical_calibration_file = 'vertical_calibration.npy'
    
    if not os.path.exists(homography_file):
        print(f"Homography calibration file not found: {homography_file}")
        print("Please run standard calibration first.")
        return
        
    # Load homography matrix
    homography = np.load(homography_file, allow_pickle=True)
    
    # Initialize eye tracker with vertical calibration if available
    eye_tracker = EyeTracker(use_vertical_ratio=True, use_3d_pose=True)
    if os.path.exists(vertical_calibration_file):
        print(f"Loading vertical calibration from {vertical_calibration_file}")
        eye_tracker.load_calibration(vertical_calibration_file)
    else:
        print("No vertical calibration file found. Using default values.")
    
    camera = Camera()
    if not camera.start():
        print("Camera initialization failed!")
        return
    
    # Create a visualization window
    cv2.namedWindow("Calibration Test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Create a black screen for visualization
    screen_width, screen_height = 1920, 1080
    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Add targets to track
    targets = [
        (screen_width//2, 100),                # Top center
        (screen_width//2, screen_height-100),  # Bottom center
        (100, screen_height//2),               # Left center 
        (screen_width-100, screen_height//2),  # Right center
        (screen_width//2, screen_height//2),   # Center
    ]
    
    print("Move your gaze around the screen. Press ESC to exit.")
    
    running = True
    while running:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            break
            
        # Process frame with eye tracker
        result = eye_tracker.process_frame(frame)
        
        # Update visualization screen
        screen.fill(0)
        
        # Draw targets
        for i, (x, y) in enumerate(targets):
            cv2.circle(screen, (x, y), 20, (0, 255, 0), -1)
            cv2.putText(screen, str(i+1), (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Transform gaze point using homography
        if result["success"] and "gaze_point" in result and result["gaze_point"] is not None:
            gaze_array = np.array([result["gaze_point"][0], result["gaze_point"][1], 1]).reshape(3, 1)
            screen_point = homography @ gaze_array
            screen_point /= screen_point[2]
            
            screen_x, screen_y = int(screen_point[0].item()), int(screen_point[1].item())
            
            # Ensure point is on screen
            screen_x = max(0, min(screen_width-1, screen_x))
            screen_y = max(0, min(screen_height-1, screen_y))
            
            # Draw gaze point
            cv2.circle(screen, (screen_x, screen_y), 15, (0, 0, 255), -1)
            
            # Add vertical ratio text if available
            if "vertical_ratio" in result and result["vertical_ratio"] is not None:
                vertical_ratio = result["vertical_ratio"]
                cv2.putText(screen, f"V-Ratio: {vertical_ratio:.2f}", 
                           (screen_width-300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display visualization and camera view
        cv2.imshow("Calibration Test", screen)
        
        # Resize camera view and show it
        vis_frame = eye_tracker.visualize(frame, result)
        vis_frame_resized = cv2.resize(vis_frame, (640, 360))
        cv2.imshow("Camera Feed", vis_frame_resized)
        
        # Check for exit
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            running = False
    
    # Cleanup
    cv2.destroyAllWindows()
    camera.stop()
    print("Calibration test completed.")

def main():
    """Main function for calibration utility."""
    parser = argparse.ArgumentParser(description='Eye tracking calibration')
    parser.add_argument('--mode', type=str, default='menu', 
                        choices=['standard', 'vertical', 'test', 'menu'],
                        help='Calibration mode: standard, vertical, test, or menu')
    
    args = parser.parse_args()
    
    if args.mode == 'standard':
        perform_standard_calibration()
    elif args.mode == 'vertical':
        perform_vertical_calibration()
    elif args.mode == 'test':
        test_calibration()
    else:
        # Interactive menu
        while True:
            print("\n=== EYE TRACKING CALIBRATION UTILITY ===")
            print("1. Standard Calibration (Screen Mapping)")
            print("2. Vertical Gaze Calibration")
            print("3. Test Calibration")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                perform_standard_calibration()
            elif choice == '2':
                perform_vertical_calibration()
            elif choice == '3':
                test_calibration()
            elif choice == '4':
                print("Exiting...")
                break
            else:
                print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()