#!/usr/bin/env python3
"""
Eye tracking debug utility to help troubleshoot issues with eye tracking.
This script shows raw camera input and overlays face/eye detection visuals.
"""

import os
import sys
import cv2
import time
import argparse
import logging
import numpy as np

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.camera import Camera
from models.eye_tracking_model import EyeTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('eye_tracking_debug')

def main():
    """Main function to run the eye tracking debug tool."""
    parser = argparse.ArgumentParser(description='Debug eye tracking issues')
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index to use (default: 0)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Camera width (default: 1280)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Camera height (default: 720)'
    )
    
    parser.add_argument(
        '--detection-confidence',
        type=float,
        default=0.5,
        help='Minimum face detection confidence (default: 0.5)'
    )
    
    parser.add_argument(
        '--tracking-confidence',
        type=float,
        default=0.5,
        help='Minimum face tracking confidence (default: 0.5)'
    )
    
    parser.add_argument(
        '--flip',
        action='store_true',
        help='Horizontally flip the camera input (useful for correcting mirrored webcams)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize camera
        camera = Camera(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            horizontal_flip=args.flip
        )
        
        if not camera.start():
            logger.error("Failed to start camera")
            return 1
        
        # Initialize eye tracker with provided confidence values
        eye_tracker = EyeTracker(
            min_detection_confidence=args.detection_confidence,
            min_tracking_confidence=args.tracking_confidence,
            invert_x_gaze=True
        )
        
        # Create display window
        cv2.namedWindow('Eye Tracking Debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Eye Tracking Debug', args.width, args.height)
        
        # Capture and process frames
        frame_count = 0
        last_time = time.time()
        fps = 0
        running = True
        show_mesh = False
        
        print("\n*** EYE TRACKING DEBUG TOOL ***")
        print("Press 'ESC' or 'q' to quit")
        print("Press 'm' to toggle face mesh visualization")
        print("Press 'f' to toggle horizontal flip")
        print("Press 's' to take a screenshot")
        print(f"Camera horizontal flip: {'Enabled' if args.flip else 'Disabled'}\n")
        
        while running:
            # Read frame from camera
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            # Process frame with eye tracker
            result = eye_tracker.process_frame(frame)
            
            # Add debug information to the frame
            debug_frame = frame.copy()
            
            # Display face detection status
            face_detected = result.get("face_detected", False)
            cv2.putText(
                debug_frame,
                f"Face Detected: {face_detected}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if face_detected else (0, 0, 255),
                2
            )
            
            # Display eye detection status
            eyes_detected = "gaze_point" in result and result["gaze_point"] is not None
            cv2.putText(
                debug_frame,
                f"Eyes Tracked: {eyes_detected}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if eyes_detected else (0, 0, 255),
                2
            )
            
            # Display gaze point coordinates if available
            if eyes_detected:
                x, y = result["gaze_point"]
                cv2.putText(
                    debug_frame,
                    f"Gaze: ({int(x)}, {int(y)})",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display detection rate
            detection_rate = result.get("detection_rate", 0)
            cv2.putText(
                debug_frame,
                f"Detection Rate: {detection_rate:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 165, 0),
                2
            )
            
            # Display FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_time = current_time
            
            cv2.putText(
                debug_frame,
                f"FPS: {fps:.2f}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 165, 0),
                2
            )
            
            # Draw face mesh and tracking visualization
            vis_frame = eye_tracker.visualize(debug_frame, result, show_mesh=show_mesh)
            
            # Display the frame
            cv2.imshow('Eye Tracking Debug', vis_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or q to quit
                running = False
            elif key == ord('m'):  # Toggle face mesh
                show_mesh = not show_mesh
                logger.info(f"Face mesh visualization {'enabled' if show_mesh else 'disabled'}")
            elif key == ord('f'):  # Toggle horizontal flip
                camera.horizontal_flip = not camera.horizontal_flip
                logger.info(f"Camera horizontal flip {'enabled' if camera.horizontal_flip else 'disabled'}")
            elif key == ord('s'):  # Take screenshot
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                os.makedirs("screenshots", exist_ok=True)
                filename = f"screenshots/eye_tracking_debug_{timestamp}.jpg"
                cv2.imwrite(filename, vis_frame)
                logger.info(f"Screenshot saved to {filename}")
        
        # Clean up resources
        camera.stop()
        cv2.destroyAllWindows()
        return 0
        
    except Exception as e:
        logger.error(f"Error during eye tracking debug: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 