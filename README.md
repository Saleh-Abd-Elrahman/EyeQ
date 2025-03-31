# Smart Eye-Tracking Shelves for Retail Optimization

An AI-powered solution for analyzing customer interactions with retail shelves through eye-tracking and providing actionable insights for e-commerce product optimization.

## Features

- **Attention Analysis**: Track which products attract the most attention
- **E-commerce Integration**: Works with Amazon product pages for real-world analytics
- **Gaze Cursor Control**: Control cursor with eye movements for intuitive interaction
- **Calibration System**: Easy setup process for accurate eye tracking
- **Engagement Measurement**: Determine how long customers look at a product
- **Promotional Effectiveness**: Analyze which price tags or promotional labels capture interest
- **Blind Spot Identification**: Identify areas of the page that are ignored
- **Comprehensive Reports**: Generate detailed HTML reports with visualizations and metrics

## System Components

1. **Camera Module**: Handles video input from RGB cameras
2. **Eye-Tracking System**: Detects and tracks customer eye movements using MediaPipe
3. **Dynamic Grid Extractor**: Maps eye coordinates to products on e-commerce pages
4. **Data Collection**: Stores tracking data for future analysis
5. **Analytics Dashboard**: Visualizes insights from the collected data
6. **Report Generator**: Creates comprehensive HTML reports with charts and metrics

## Requirements

- Python 3.12
- OpenCV
- MediaPipe (for eye tracking)
- NumPy
- Matplotlib (for visualizations)
- PyQt5 (for dashboard UI)
- Pandas (for data analysis)
- Selenium (for web browser integration)
- Chrome WebDriver
- Pynput (for cursor control)

## Installation

See the `INSTALLATION.md` file for detailed setup instructions.

## Usage

Refer to `USAGE.md` for instructions on running the system and interpreting the results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.