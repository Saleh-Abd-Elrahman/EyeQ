# Usage Guide

## Getting Started

After completing the installation process outlined in `INSTALLATION.md`, follow these instructions to run the smart eye-tracking system.

## Running the System

### Calibration

Before using the system, it's recommended to run the calibration utility:

```bash
python calibration.py
```

Follow the on-screen instructions to complete the calibration process. This will improve the accuracy of the eye tracking.

### Basic Usage

1. Activate your virtual environment if not already active:
   ```bash
   # For Windows
   venv\Scripts\activate
   
   # For macOS/Linux
   source venv/bin/activate
   ```

2. Run the main application:
   ```bash
   python main.py
   ```

3. The system will:
   - Automatically detect connected cameras
   - Start Chrome browser with an Amazon search page
   - Initialize eye tracking
   - Show a gaze cursor controlled by your eye movements
   - Start analyzing your gaze attention on Amazon products

### Configuration

You can customize the system behavior by modifying the `config.py` file:

- `CAMERA_INDEX`: Specify which camera to use (default: 0)
- `FRAME_WIDTH` and `FRAME_HEIGHT`: Set camera resolution
- `DETECTION_THRESHOLD`: Adjust the confidence threshold for eye detection
- `VISUALIZATION_LEVEL`: Set level of real-time visualization
- `LOG_LEVEL`: Set logging verbosity (INFO, DEBUG, etc.)

### Command Line Options

You can modify the eye tracking debug tool behavior with these options:

```bash
# Use a specific camera
python eye_tracking_debug.py --camera 1

# Set camera dimensions
python eye_tracking_debug.py --width 1280 --height 720

# Adjust detection confidence
python eye_tracking_debug.py --detection-confidence 0.6

# Flip camera horizontally
python eye_tracking_debug.py --flip
```

## Interacting with the System

### Keyboard Controls

- `ESC`: Toggle cursor control on/off - when disabled, you can use your normal mouse
- `Ctrl+C`: Gracefully exit the application
- In debug mode:
  - `m`: Toggle face mesh visualization
  - `f`: Toggle horizontal flip
  - `s`: Take a screenshot
  - `q` or `ESC`: Exit debug tool

### Using the Gaze Cursor

The gaze cursor is enabled by default and allows you to control your mouse pointer with eye movements. This enables a hands-free browsing experience. To temporarily disable it and use your regular mouse, press `ESC`.

## Data Analysis

### Generated Data

The system automatically generates:

1. Attention data for products viewed on Amazon
2. Gaze coordinates over time
3. Product interaction analytics

### Viewing Eye Tracking Data

You can view the calibration data with:

```bash
python calibration_viewer.py
```

This displays the homography matrix used for mapping eye gaze to screen coordinates.

### Running Test Reports

To generate a sample report using existing data:

```bash
python test_report.py
```

## Amazon Integration

The system is configured to work with Amazon product search pages by default. When the main application runs:

1. It automatically opens Chrome browser with Amazon search
2. Extracts product regions from the page using the dynamic grid extractor
3. Tracks your gaze as you browse products
4. Records which products you view and for how long
5. You can scroll the page manually to view more products

## Troubleshooting

- **Poor Tracking Performance**: 
  - Use the eye_tracking_debug.py tool to diagnose issues
  - Ensure adequate lighting without glare
  - Position your face clearly visible to the camera
  
- **Browser Doesn't Open**: 
  - Check that Chrome is installed in the standard location
  - Try updating the chromedriver_path in main.py
  
- **Calibration Fails**:
  - Ensure you are in good lighting conditions
  - Follow the green dots with your eyes during calibration
  - Try to keep your head relatively still during calibration
  
- **Cursor Control Issues**:
  - Press ESC to toggle cursor control on/off
  - Recalibrate the system for better accuracy
  - Check that the calibration_homography.npy file exists

- **System Doesn't Detect Products**:
  - Ensure the Amazon page has loaded completely
  - Try a different product search query
  - Check your internet connection
