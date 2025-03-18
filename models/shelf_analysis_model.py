"""
Shelf analysis model for mapping eye gaze to shelf regions and products.
"""

import cv2
import numpy as np
import time
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Product:
    """Class for storing product information on a shelf."""
    id: str
    name: str
    price: float
    position: List[int]  # [x1, y1, x2, y2] - top-left and bottom-right coordinates
    category: str = ""
    promotion: bool = False
    promotion_type: str = ""
    
    # Attention metrics
    total_attention_time: float = 0.0
    attention_count: int = 0
    last_attention_time: float = 0.0
    is_current_attention: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @property
    def width(self) -> int:
        """Get the width of the product region."""
        return self.position[2] - self.position[0]
    
    @property
    def height(self) -> int:
        """Get the height of the product region."""
        return self.position[3] - self.position[1]
    
    @property
    def area(self) -> int:
        """Get the area of the product region."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center of the product region."""
        return (
            self.position[0] + self.width // 2,
            self.position[1] + self.height // 2
        )
        
    @property
    def average_attention_time(self) -> float:
        """Get the average attention time per view."""
        if self.attention_count == 0:
            return 0.0
        return self.total_attention_time / self.attention_count


class ShelfAnalyzer:
    """
    Shelf analysis model for mapping eye gaze to products and tracking attention.
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize the shelf analyzer.
        
        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.products: List[Product] = []
        self.attention_history: List[Dict[str, Any]] = []
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.heatmap_resolution = (50, 50)  # Default resolution for the heatmap grid
        self.heatmap_grid = np.zeros(self.heatmap_resolution, dtype=np.float32)
        
        # Time tracking
        self.last_update_time = time.time()
        self.session_start_time = time.time()
        
        # Attention thresholds
        self.min_attention_time = 0.3  # Minimum time (seconds) to consider as attention
        self.attention_cooldown = 1.0  # Time (seconds) before counting a new attention to the same product
        
        logger.info(f"Shelf analyzer initialized with frame size: {frame_width}x{frame_height}")
        
    def add_product(self, product: Product) -> None:
        """
        Add a product to the shelf.
        
        Args:
            product: Product object to add
        """
        self.products.append(product)
        logger.info(f"Added product: {product.name} at position {product.position}")
        
    def add_products_from_dict(self, products_dict: List[Dict[str, Any]]) -> None:
        """
        Add multiple products from a dictionary list.
        
        Args:
            products_dict: List of product dictionaries
        """
        for product_data in products_dict:
            product = Product(
                id=product_data.get("id", f"product_{len(self.products)}"),
                name=product_data.get("name", f"Product {len(self.products)}"),
                price=product_data.get("price", 0.0),
                position=product_data.get("position", [0, 0, 100, 100]),
                category=product_data.get("category", ""),
                promotion=product_data.get("promotion", False),
                promotion_type=product_data.get("promotion_type", "")
            )
            self.add_product(product)
            
    def load_products_from_file(self, filepath: str) -> bool:
        """
        Load products from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                products_data = json.load(f)
                
            self.add_products_from_dict(products_data)
            return True
            
        except Exception as e:
            logger.error(f"Error loading products from file: {e}")
            return False
            
    def save_products_to_file(self, filepath: str) -> bool:
        """
        Save products to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            products_data = [product.to_dict() for product in self.products]
            
            with open(filepath, 'w') as f:
                json.dump(products_data, f, indent=4)
                
            logger.info(f"Saved {len(products_data)} products to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving products to file: {e}")
            return False
            
    def process_gaze(self, gaze_point: List[float], timestamp: float = None) -> Dict[str, Any]:
        """
        Process a gaze point and update attention metrics.
        
        Args:
            gaze_point: [x, y] coordinates of the gaze point
            timestamp: Optional timestamp (default: current time)
            
        Returns:
            Dict with processing results
        """
        if timestamp is None:
            timestamp = time.time()
            
        x, y = gaze_point
        current_time = timestamp
        time_delta = current_time - self.last_update_time
        
        # Update all products - first reset current attention flag
        for product in self.products:
            product.is_current_attention = False
        
        # Find the product being looked at
        gazed_product = None
        for product in self.products:
            x1, y1, x2, y2 = product.position
            if x1 <= x <= x2 and y1 <= y <= y2:
                gazed_product = product
                product.is_current_attention = True
                
                # If this is a new attention or after cooldown
                if not product.last_attention_time or \
                   (current_time - product.last_attention_time) > self.attention_cooldown:
                    product.attention_count += 1
                    
                # Update attention time
                product.total_attention_time += time_delta
                product.last_attention_time = current_time
                break
        
        # Update heatmap
        if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
            # Main heatmap (full resolution)
            self.heatmap[int(y), int(x)] += 1
            
            # Downsampled heatmap grid (for efficiency in visualization)
            grid_x = int(x / self.frame_width * self.heatmap_resolution[1])
            grid_y = int(y / self.frame_height * self.heatmap_resolution[0])
            
            if 0 <= grid_x < self.heatmap_resolution[1] and 0 <= grid_y < self.heatmap_resolution[0]:
                self.heatmap_grid[grid_y, grid_x] += 1
            
        # Record in attention history
        attention_record = {
            "timestamp": current_time,
            "gaze_point": gaze_point,
            "product_id": gazed_product.id if gazed_product else None,
            "product_name": gazed_product.name if gazed_product else None,
            "duration": time_delta
        }
        self.attention_history.append(attention_record)
        
        # Update time
        self.last_update_time = current_time
        
        # Prepare result
        result = {
            "gazed_product": gazed_product.to_dict() if gazed_product else None,
            "timestamp": current_time,
            "session_duration": current_time - self.session_start_time
        }
        
        return result
    
    def get_attention_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive attention metrics for all products.
        
        Returns:
            Dict with attention metrics
        """
        if not self.products:
            return {"error": "No products defined"}
        
        # Sort products by total attention time
        sorted_by_time = sorted(
            self.products, 
            key=lambda p: p.total_attention_time, 
            reverse=True
        )
        
        # Sort products by attention count
        sorted_by_count = sorted(
            self.products, 
            key=lambda p: p.attention_count, 
            reverse=True
        )
        
        # Calculate percentage of total attention for each product
        total_attention = sum(p.total_attention_time for p in self.products)
        attention_percentages = {}
        for product in self.products:
            if total_attention > 0:
                percentage = (product.total_attention_time / total_attention) * 100
            else:
                percentage = 0
            attention_percentages[product.id] = percentage
        
        # Identify products that received no attention
        ignored_products = [p.to_dict() for p in self.products if p.attention_count == 0]
        
        # Calculate session duration
        session_duration = time.time() - self.session_start_time
        
        # Prepare metrics dictionary
        metrics = {
            "session_duration": session_duration,
            "total_attention_time": total_attention,
            "product_count": len(self.products),
            "products_by_attention_time": [p.to_dict() for p in sorted_by_time],
            "products_by_attention_count": [p.to_dict() for p in sorted_by_count],
            "attention_percentages": attention_percentages,
            "ignored_products": ignored_products,
            "ignored_percentage": (len(ignored_products) / len(self.products)) * 100 if self.products else 0,
            "timestamp": time.time()
        }
        
        return metrics
    
    def generate_heatmap_visualization(self, frame: np.ndarray, opacity: float = 0.6) -> np.ndarray:
        """
        Generate a visualization of the attention heatmap overlaid on a frame.
        
        Args:
            frame: The frame to overlay the heatmap on
            opacity: Opacity of the heatmap (0-1)
            
        Returns:
            Frame with heatmap overlay
        """
        if self.heatmap_grid.max() == 0:
            return frame  # No data in heatmap
            
        # Normalize heatmap grid
        normalized_grid = self.heatmap_grid / self.heatmap_grid.max()
        
        # Resize to frame dimensions
        resized_heatmap = cv2.resize(
            normalized_grid, 
            (frame.shape[1], frame.shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (resized_heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Create mask for visibility
        mask = (resized_heatmap > 0.05).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Blend with original frame
        blend = frame.copy()
        for c in range(3):  # RGB channels
            blend[:, :, c] = (1 - mask * opacity) * frame[:, :, c] + \
                             (mask * opacity) * heatmap_colored[:, :, c]
        
        return blend.astype(np.uint8)
    
    def draw_product_regions(self, frame: np.ndarray, show_metrics: bool = True) -> np.ndarray:
        """
        Draw product regions on the frame.
        
        Args:
            frame: The frame to draw on
            show_metrics: Whether to show attention metrics
            
        Returns:
            Frame with product regions drawn
        """
        annotated_frame = frame.copy()
        
        for product in self.products:
            x1, y1, x2, y2 = product.position
            
            # Determine color based on attention
            if product.is_current_attention:
                color = (0, 255, 0)  # Green for current attention
                thickness = 3
            elif product.attention_count > 0:
                # Gradient from yellow to red based on attention percentage
                metrics = self.get_attention_metrics()
                if metrics["total_attention_time"] > 0:
                    percentage = product.total_attention_time / metrics["total_attention_time"]
                    # From yellow (0, 255, 255) to red (0, 0, 255)
                    color = (0, int(255 * (1 - percentage)), 255)
                else:
                    color = (0, 255, 255)  # Yellow for some attention
                thickness = 2
            else:
                color = (128, 128, 128)  # Gray for no attention
                thickness = 1
                
            # Draw rectangle around product
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add product name
            cv2.putText(
                annotated_frame, 
                product.name, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1, 
                cv2.LINE_AA
            )
            
            # Add metrics if requested
            if show_metrics and product.attention_count > 0:
                metrics_text = f"Time: {product.total_attention_time:.1f}s | Views: {product.attention_count}"
                cv2.putText(
                    annotated_frame, 
                    metrics_text, 
                    (x1, y2 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    color, 
                    1, 
                    cv2.LINE_AA
                )
                
        return annotated_frame
    
    def reset(self) -> None:
        """
        Reset all attention metrics and heatmap.
        """
        # Reset product metrics
        for product in self.products:
            product.total_attention_time = 0.0
            product.attention_count = 0
            product.last_attention_time = 0.0
            product.is_current_attention = False
            
        # Reset heatmap
        self.heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        self.heatmap_grid = np.zeros(self.heatmap_resolution, dtype=np.float32)
        
        # Reset time tracking
        self.last_update_time = time.time()
        self.session_start_time = time.time()
        
        # Clear attention history
        self.attention_history = []
        
        logger.info("Shelf analyzer reset")
        
    def export_session_data(self, output_dir: str) -> str:
        """
        Export session data to files.
        
        Args:
            output_dir: Directory to save data to
            
        Returns:
            Path to the exported data directory
        """
        try:
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(output_dir, f"session_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)
            
            # Save product data
            products_path = os.path.join(session_dir, "products.json")
            self.save_products_to_file(products_path)
            
            # Save attention history
            history_path = os.path.join(session_dir, "attention_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.attention_history, f, indent=4)
                
            # Save metrics
            metrics_path = os.path.join(session_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.get_attention_metrics(), f, indent=4)
                
            # Save heatmap as numpy array
            heatmap_path = os.path.join(session_dir, "heatmap.npy")
            np.save(heatmap_path, self.heatmap)
            
            # Save heatmap as image
            heatmap_img_path = os.path.join(session_dir, "heatmap.png")
            if self.heatmap.max() > 0:
                normalized = self.heatmap / self.heatmap.max() * 255
                heatmap_img = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(heatmap_img_path, heatmap_img)
                
            logger.info(f"Session data exported to {session_dir}")
            return session_dir
            
        except Exception as e:
            logger.error(f"Error exporting session data: {e}")
            return ""
            
    def create_demo_shelf(self, width: int, height: int, rows: int = 3, cols: int = 4) -> None:
        """
        Create a demo shelf with random products.
        
        Args:
            width: Frame width
            height: Frame height
            rows: Number of rows of products
            cols: Number of columns of products
        """
        # Clear existing products
        self.products = []
        
        # Calculate product dimensions
        margin = 20
        product_width = (width - margin * (cols + 1)) // cols
        product_height = (height - margin * (rows + 1)) // rows
        
        # Product categories
        categories = ["Beverages", "Snacks", "Canned Goods", "Dairy", "Cleaning"]
        
        # Create products
        product_id = 1
        for row in range(rows):
            for col in range(cols):
                x1 = margin + col * (product_width + margin)
                y1 = margin + row * (product_height + margin)
                x2 = x1 + product_width
                y2 = y1 + product_height
                
                # Random price between $1.99 and $9.99
                price = round(1.99 + np.random.random() * 8, 2)
                
                # Random category
                category = categories[np.random.randint(0, len(categories))]
                
                # Random promotion (20% chance)
                promotion = np.random.random() < 0.2
                promotion_type = "SALE" if promotion else ""
                
                product = Product(
                    id=f"P{product_id}",
                    name=f"Product {product_id}",
                    price=price,
                    position=[x1, y1, x2, y2],
                    category=category,
                    promotion=promotion,
                    promotion_type=promotion_type
                )
                
                self.add_product(product)
                product_id += 1
                
        logger.info(f"Created demo shelf with {rows*cols} products")
        
    def __del__(self):
        """Clean up resources."""
        logger.info("Shelf analyzer resources released") 