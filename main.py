#!/usr/bin/env python3
"""
Main application for Smart Eye-Tracking Shelves for Retail Optimization.

This application uses computer vision and eye-tracking to analyze how customers
interact with products on retail shelves.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any

# Import configuration
import config

# Import components
from utils.camera import Camera, VideoFileCamera, list_available_cameras
from models.eye_tracking_model import EyeTracker
from models.shelf_analysis_model import ShelfAnalyzer, Product
from analytics.attention_analytics import AttentionAnalytics
from ui.dashboard import launch_dashboard

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments.
    
    Returns:
        Dict containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Smart Eye-Tracking Shelves for Retail Optimization"
    )
    
    parser.add_argument(
        "--camera", 
        type=int, 
        default=config.CAMERA_INDEX,
        help="Camera index to use (default: {})".format(config.CAMERA_INDEX)
    )
    
    parser.add_argument(
        "--video", 
        type=str, 
        help="Path to video file for processing instead of live camera"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=config.OUTPUT_DIR,
        help="Output directory for analytics data (default: {})".format(config.OUTPUT_DIR)
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["shelf", "checkout"], 
        default="shelf",
        help="Operation mode: shelf or checkout (default: shelf)"
    )
    
    args = parser.parse_args()
    
    return vars(args)


def initialize_components(args):
    """
    Initialize system components based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (camera, eye_tracker, shelf_analyzer, analytics)
    """
    # Set debug mode if requested
    if args["debug"]:
        logging.getLogger().setLevel(logging.DEBUG)
        config.DEBUG = True
        logger.debug("Debug mode enabled")
    
    # Create camera
    if args["video"]:
        logger.info(f"Using video file: {args['video']}")
        camera = VideoFileCamera(args["video"])
    else:
        camera_index = args["camera"]
        logger.info(f"Using camera index: {camera_index}")
        camera = Camera(
            camera_index=camera_index,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            fps=config.FPS
        )
    
    # Create eye tracker
    eye_tracker = EyeTracker(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        max_num_faces=config.MAX_NUM_FACES
    )
    
    # Create shelf analyzer
    shelf_analyzer = ShelfAnalyzer(
        frame_width=config.FRAME_WIDTH,
        frame_height=config.FRAME_HEIGHT
    )
    
    # Set up a demo shelf with products if in shelf mode
    if args["mode"] == "shelf":
        # Create a demo shelf layout
        shelf_analyzer.create_demo_shelf(
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            rows=3,
            cols=4
        )
    else:  # Checkout mode
        # Define checkout UI regions
        for name, region in config.CHECKOUT_AOI_REGIONS.items():
            product = Product(
                id=f"checkout_{name}",
                name=f"Checkout {name.replace('_', ' ').title()}",
                price=0.0,
                position=region
            )
            shelf_analyzer.add_product(product)
    
    # Create analytics
    analytics = AttentionAnalytics()
    
    return camera, eye_tracker, shelf_analyzer, analytics


def main():
    """Main application entry point."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args["output"], exist_ok=True)
    
    # Initialize components
    camera, eye_tracker, shelf_analyzer, analytics = initialize_components(args)
    
    # Launch the dashboard UI
    launch_dashboard(camera, eye_tracker, shelf_analyzer, analytics)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1) 