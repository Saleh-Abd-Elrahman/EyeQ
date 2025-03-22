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

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def setup_browser():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def main():
    driver = setup_browser()
    url = "https://www.amazon.es/s?k=laptop"
    driver.get(url)
    logger.info(f"Opened webpage: {url}")

    product_regions = extract_amazon_product_regions(driver)
    logger.info(f"Extracted {len(product_regions)} potential products from Amazon.")

    eye_tracker = EyeTracker()
    camera = Camera()
    if not camera.start():
        logger.error("Failed to start camera. Exiting.")
        return

    homography = np.load('calibration_homography.npy', allow_pickle=True)
    attention = AttentionAnalytics(product_regions=product_regions)

    logger.info("Tracking started — scroll manually.")
    print("Scroll manually, press Ctrl+C, close browser, or terminal to stop...")

    try:
        while True:
            try:
                driver.title  # Check if browser is open
            except:
                logger.info("Browser closed by user.")
                break

            success, frame = camera.read()
            if not success or frame is None:
                logger.warning("Camera frame not captured.")
                continue

            result = eye_tracker.process_frame(frame)
            gaze_point = result.get("gaze_point")

            if gaze_point:
                gaze_array = np.array([gaze_point[0], gaze_point[1], 1]).reshape(3, 1)
                screen_point = homography @ gaze_array
                screen_point /= screen_point[2]
                screen_x, screen_y = int(screen_point[0].item()), int(screen_point[1].item())
                timestamp = time.time()

                for product in product_regions:
                    left, top, right, bottom = product["bbox"]
                    if left <= screen_x <= right and top <= screen_y <= bottom:
                        product["total_attention_time"] = product.get("total_attention_time", 0) + 0.1
                        attention.attention_history.append({
                            "timestamp": timestamp,
                            "product_id": product["id"],
                            "product_name": product["name"],
                            "duration": 0.1
                        })
                        break

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Session ended by user.")
    finally:
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