# Installation Guide

## System Requirements
- Python 3.12
- Webcam or RGB camera
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Modern web browser for viewing HTML reports

## Setting Up the Environment

### Step 1: Clone the Repository
```bash
git clone https://github.com/Saleh-Abd-Elrahman/EyeQ
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

### Step 4: Verify Installation
```bash
python -c "import cv2; import mediapipe; import numpy; import matplotlib; import PyQt5; import pandas; import selenium; print(' All dependencies successfully installed!')"
```

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

4. **Matplotlib or Pandas Issues**
   If you encounter issues with these libraries:
   ```bash
   pip uninstall matplotlib pandas
   pip install matplotlib pandas --force-reinstall
   ```

## Camera Setup

1. Ensure your camera is properly connected and recognized by your system
2. For optimal eye-tracking performance, the camera should be:
   - Positioned at eye level relative to the average customer
   - Have unobstructed view of the shelf and customer
   - In an area with consistent lighting

## Data Storage Setup

1. The system will automatically create a `data` directory for storing session data and reports
2. Ensure the application has write permissions to this directory
3. Reports are generated as HTML files with supporting images, viewable in any modern web browser

## Next Steps

After successful installation, refer to the `USAGE.md` file for instructions on how to run the system. 