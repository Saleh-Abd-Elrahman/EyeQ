"""
Configuration settings for the smart eye-tracking retail system.
"""

import os

# Camera Settings
CAMERA_INDEX = 0  # Default camera (0 = primary webcam)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Eye Tracking Settings
DETECTION_THRESHOLD = 0.7  # Confidence threshold for eye detection
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_FACES = 1  # Number of faces to track simultaneously

# Visualization Settings
VISUALIZATION_LEVEL = 2  # 0=none, 1=basic, 2=detailed
HEATMAP_OPACITY = 0.6
GAZE_POINT_SIZE = 5
GAZE_POINT_COLOR = (0, 255, 0)  # Green
FACE_LANDMARKS_COLOR = (255, 0, 0)  # Red
SHOW_FACE_MESH = False

# Recording and Data Settings
RECORDING_ENABLED = True
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")

# Ensure directories exist
for directory in [DATA_DIR, OUTPUT_DIR, SAMPLE_IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Analytics Settings
ATTENTION_THRESHOLD_SECONDS = 1.0  # Minimum time to consider as "attention"
HEATMAP_RESOLUTION = (50, 50)  # Grid resolution for heatmap
ANALYSIS_WINDOW_SECONDS = 60  # Default time window for analytics

# Dashboard Settings
UI_REFRESH_RATE_MS = 33  # ~30 FPS
DASHBOARD_WIDTH = 1600
DASHBOARD_HEIGHT = 900
DASHBOARD_TITLE = "Smart Eye-Tracking Retail Analytics"

# Self-Checkout Mode Settings
CHECKOUT_UI_WIDTH = 1024
CHECKOUT_UI_HEIGHT = 768
CHECKOUT_AOI_REGIONS = {
    "scan_area": [100, 200, 300, 400],  # [x, y, width, height]
    "payment_options": [500, 200, 200, 100],
    "help_button": [800, 500, 100, 50],
    "total_display": [500, 100, 200, 50]
}

# Debug Settings
DEBUG = False
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL 