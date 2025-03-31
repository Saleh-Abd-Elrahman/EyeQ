#!/usr/bin/env python3
"""
Improved main application for Smart Eye-Tracking on Amazon pages.
Tracks gaze, gracefully handles browser closure, and reports only viewed products.
"""

import time
import logging
import numpy as np
import cv2
import config
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from models.eye_tracking_model import EyeTracker
from utils.camera import Camera
from analytics.dynamic_grid_extractor import extract_amazon_product_regions
from analytics.attention_analytics import AttentionAnalytics

# Import the GazeCursor class
from gaze_cursor import GazeCursor
import pyautogui  # Add this import for mouse position fallback

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def setup_browser():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    # Use manually downloaded ChromeDriver if available
    try:
        chromedriver_path = "path/to/your/chromedriver"  # Update this path
        service = Service(chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logger.info("Using manually specified ChromeDriver")
    except:
        logger.info("Attempting to use WebDriver Manager")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    
    return driver

def main():
    driver = setup_browser()
    url = "https://www.amazon.es/s?k=laptop"
    driver.get(url)
    logger.info(f"Opened webpage: {url}")

    # Initialize the gaze cursor with a larger size and brighter color
    gaze_cursor = GazeCursor(size=40, color=(0, 255, 0), alpha=0.9)  # Larger, green, more opaque
    gaze_cursor.start()
    gaze_cursor.show()  # Show the cursor immediately
    
    # Add a timestamp for the last extraction
    last_extraction_time = time.time()

    product_regions = extract_amazon_product_regions(driver)
    logger.info(f"Extracted {len(product_regions)} potential products from Amazon.")

    eye_tracker = EyeTracker()
    camera = Camera()
    if not camera.start():
        logger.error("Failed to start camera. Exiting.")
        gaze_cursor.stop()
        return

    homography = np.load('calibration_homography.npy', allow_pickle=True)
    attention = AttentionAnalytics(product_regions=product_regions)

    logger.info("Tracking started — scroll manually.")
    print("Scroll manually, press Ctrl+C, close browser, or terminal to stop...")

    try:
        # Debug counter for gaze points
        gaze_point_count = 0
        no_gaze_point_count = 0
        
        while True:
            try:
                driver.title  # Check if browser is open
            except:
                logger.info("Browser closed by user.")
                break

            # Re-extract product regions periodically
            current_time = time.time()
            if current_time - last_extraction_time > 5.0:  # Re-extract every 5 seconds
                product_regions = extract_amazon_product_regions(driver)
                last_extraction_time = current_time
                attention.product_regions = product_regions
                logger.info(f"Re-extracted {len(product_regions)} products")

            success, frame = camera.read()
            if not success or frame is None:
                logger.warning("Camera frame not captured.")
                continue

            result = eye_tracker.process_frame(frame)
            gaze_point = result.get("gaze_point")

            if gaze_point:
                gaze_point_count += 1
                gaze_array = np.array([gaze_point[0], gaze_point[1], 1]).reshape(3, 1)
                screen_point = homography @ gaze_array
                screen_point /= screen_point[2]
                screen_x, screen_y = int(screen_point[0].item()), int(screen_point[1].item())
                timestamp = time.time()
                
                logger.debug(f"Gaze point detected: ({screen_x}, {screen_y})")
                
                # Update the gaze cursor position
                gaze_cursor.update_position(screen_x, screen_y)
                gaze_cursor.show()
                
                scroll_y = driver.execute_script("return window.pageYOffset;")
                scroll_x = driver.execute_script("return window.pageXOffset;")

                # Adjust for scroll
                adjusted_y = screen_y + scroll_y
                adjusted_x = screen_x + scroll_x 

                for product in product_regions:
                    left, top, right, bottom = product["bbox"]
                    if left <= adjusted_x <= right and top <= adjusted_y <= bottom:
                        product["total_attention_time"] = product.get("total_attention_time", 0) + 0.1
                        attention.attention_history.append({
                            "timestamp": timestamp,
                            "product_id": product["id"],
                            "product_name": product["name"],
                            "duration": 0.1
                        })
                        break
            else:
                no_gaze_point_count += 1
                # Use mouse position as fallback when no gaze point is detected
                mouse_x, mouse_y = pyautogui.position()
                logger.debug(f"No gaze point detected, using mouse position: ({mouse_x}, {mouse_y})")
                gaze_cursor.update_position(mouse_x, mouse_y)
                gaze_cursor.show()
            
            # Log gaze detection stats every 50 frames
            if (gaze_point_count + no_gaze_point_count) % 50 == 0:
                logger.info(f"Gaze detection stats: {gaze_point_count} detected, {no_gaze_point_count} missed")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Session ended by user.")
    finally:
        # Clean up resources
        logger.info(f"Final gaze detection stats: {gaze_point_count} detected, {no_gaze_point_count} missed")
        gaze_cursor.stop()
        camera.stop()
        driver.quit()
        eye_tracker.__del__()

        logger.info("Filtering products that received attention...")
        viewed_products = [p for p in product_regions if p.get("total_attention_time", 0) > 0]
        attention.product_regions = viewed_products

        logger.info(f"{len(viewed_products)} products viewed by user.")

        output_dir = "analytics_reports"
        attention.export_analytics_report(output_dir=output_dir)
        logger.info("Done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled error in main application.")