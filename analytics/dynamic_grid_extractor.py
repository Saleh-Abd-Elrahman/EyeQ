import time
import logging
from selenium.webdriver.common.by import By

logger = logging.getLogger(__name__)

def extract_amazon_product_regions(driver):
    """
    Extract product bounding boxes (regions) and categorize products by price from an Amazon search page.

    Args:
        driver: Selenium WebDriver instance with Amazon page loaded.

    Returns:
        List of dicts: [{"id": "product_1", "name": ..., "price": ..., "category": ..., "bbox": [x1, y1, x2, y2]}, ...]
    """
    product_regions = []

    try:
        # Wait for page to load content
        time.sleep(2)

        # Get all product containers
        product_elements = driver.find_elements(By.CSS_SELECTOR, "div.s-main-slot > div[data-component-type='s-search-result']")

        logger.info(f"Found {len(product_elements)} potential products.")

        for idx, el in enumerate(product_elements):
            try:
                location = el.location
                size = el.size

                x1 = int(location['x'])
                y1 = int(location['y'])
                x2 = x1 + int(size['width'])
                y2 = y1 + int(size['height'])

                # Extract name
                try:
                    name = el.find_element(By.CSS_SELECTOR, "h2 a span").text
                except:
                    name = f"Product {idx + 1}"

                # Extract price
                try:
                    whole = el.find_element(By.CSS_SELECTOR, "span.a-price-whole").text.replace(".", "").replace(",", "")
                    fraction = el.find_element(By.CSS_SELECTOR, "span.a-price-fraction").text
                    price = float(f"{whole}.{fraction}")
                except:
                    price = 0.0

                # Categorize based on price
                if price >= 1000:
                    category = "Premium (≥ 1000€)"
                elif price >= 500:
                    category = "Mid-range (500-999€)"
                elif price >= 200:
                    category = "Affordable (200-499€)"
                elif price > 0:
                    category = "Budget (< 200€)"
                else:
                    category = "Price Unavailable"

                product_regions.append({
                    "id": f"product_{idx + 1}",
                    "name": name,
                    "price": price,
                    "category": category,
                    "bbox": [x1, y1, x2, y2]
                })

            except Exception as e:
                logger.warning(f"Skipping product {idx + 1} due to error: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to extract product regions: {e}")

    return product_regions
