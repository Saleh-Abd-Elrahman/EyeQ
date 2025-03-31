# Installation Guide

## System Requirements
- Python 3.12
- Webcam or RGB camera
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Chrome web browser for e-commerce integration
- Modern web browser for viewing HTML reports

## Setting Up the Environment

### Step 1: Clone the Repository
```bash
git clone https://github.com/Saleh-Abd-Elrahman/EyeQ
cd EyeQ
```

### Step 2: Create a Virtual Environment
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

The requirements include:
- opencv-python>=4.8.0.74
- mediapipe>=0.10.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- pandas>=2.0.0
- scikit-learn>=1.0.0
- PyQt5>=5.15.0
- pytest>=7.0.0
- pillow>=10.0.0
- scipy>=1.10.0
- tqdm>=4.60.0
- selenium>=4.18.1
- webdriver-manager>=4.0.1
- beautifulsoup4>=4.12.3
- requests>=2.31.0
- screeninfo>=0.8.0
- pynput>=1.7.0
- pyautogui>=0.9.50

### Step 4: Verify Installation
```bash
python -c "import cv2; import mediapipe; import numpy; import matplotlib; import PyQt5; import pandas; import selenium; import pynput; print('All dependencies successfully installed!')"
```

### Step 5: Setup Chrome WebDriver
The system uses Selenium with Chrome for e-commerce integration. Chrome WebDriver will be automatically installed by the webdriver-manager package when you run the program. Alternatively, you can manually specify the ChromeDriver path in the `main.py` file.

## Running the Calibration

Before running the main application, it's recommended to perform eye tracking calibration:

```bash
python calibration.py
```

Follow the on-screen instructions to complete the calibration process.

## Troubleshooting

### Common Issues

1. **OpenCV Installation Fails**
   ```bash
   pip install --upgrade pip
   pip install opencv-python --force-reinstall
   ```

2. **MediaPipe Compatibility**
   If you encounter issues with MediaPipe, try:
   ```bash
   pip uninstall mediapipe
   pip install mediapipe==0.10.9
   ```

3. **PyQt5 Installation Issues**
   For macOS, you might need to install additional packages:
   ```bash
   brew install pyqt5
   ```
   
   For Ubuntu/Debian:
   ```bash
   sudo apt-get install python3-pyqt5
   ```

4. **Selenium and ChromeDriver Issues**
   If you encounter problems with Chrome launching:
   ```bash
   pip uninstall selenium webdriver-manager
   pip install selenium==4.18.1 webdriver-manager==4.0.1
   ```
   
   You may also need to specify the Chrome binary location in `main.py` if it's installed in a non-standard location.

5. **Cursor Control Issues**
   If the gaze cursor doesn't work properly:
   ```bash
   pip uninstall pynput pyautogui
   pip install pynput==1.7.6 pyautogui==0.9.54
   ```

## Camera Setup

1. Ensure your camera is properly connected and recognized by your system
2. For optimal eye-tracking performance, the camera should be:
   - Positioned at eye level relative to the user
   - Have an unobstructed view of the user's face
   - In an area with consistent lighting without glare on the user's face

## Data Storage Setup

1. The system will automatically create a `data` directory for storing session data and reports
2. Ensure the application has write permissions to this directory
3. Reports are generated as HTML files with supporting images, viewable in any modern web browser

## Using the Debug Tool

If you're experiencing issues with eye tracking detection, try using the debug tool:

```bash
python eye_tracking_debug.py
```

This tool provides a visual interface showing detection success rates and can help identify issues with camera placement, lighting, or face detection.

## Next Steps

After successful installation, refer to the `USAGE.md` file for instructions on how to run the system. 