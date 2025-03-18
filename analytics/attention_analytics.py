"""
Attention analytics module for processing and visualizing eye-tracking data.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionAnalytics:
    """
    Class for analyzing eye-tracking attention data from retail shelves.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the analytics module.
        
        Args:
            data_dir: Directory containing session data files
        """
        self.data_dir = data_dir
        self.products = []
        self.attention_history = []
        self.metrics = {}
        self.heatmap = None
        
        if data_dir and os.path.exists(data_dir):
            self.load_session_data(data_dir)
            
        logger.info("Attention analytics initialized")
        
    def load_session_data(self, session_dir: str) -> bool:
        """
        Load session data from files.
        
        Args:
            session_dir: Directory containing session data files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load products data
            products_path = os.path.join(session_dir, "products.json")
            if os.path.exists(products_path):
                with open(products_path, 'r') as f:
                    self.products = json.load(f)
                    
            # Load attention history
            history_path = os.path.join(session_dir, "attention_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.attention_history = json.load(f)
                    
            # Load metrics
            metrics_path = os.path.join(session_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                    
            # Load heatmap
            heatmap_path = os.path.join(session_dir, "heatmap.npy")
            if os.path.exists(heatmap_path):
                self.heatmap = np.load(heatmap_path)
                
            logger.info(f"Loaded session data from {session_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return False
            
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert attention history to pandas DataFrame for analysis.
        
        Returns:
            DataFrame containing attention data
        """
        if not self.attention_history:
            return pd.DataFrame()
            
        # Convert to pandas DataFrame
        df = pd.DataFrame(self.attention_history)
        
        # Convert timestamps to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Add derived columns
        df['duration_ms'] = df['duration'] * 1000
        
        return df
        
    def generate_product_attention_chart(self, top_n: int = 10, 
                                        save_path: Optional[str] = None,
                                        show_fig: bool = True) -> plt.Figure:
        """
        Generate a bar chart of product attention times.
        
        Args:
            top_n: Number of top products to include
            save_path: Path to save the chart
            show_fig: Whether to show the figure
            
        Returns:
            Matplotlib figure
        """
        # Sort products by attention time
        if not self.metrics or "products_by_attention_time" not in self.metrics:
            logger.error("No metrics data available")
            return None
            
        products = self.metrics["products_by_attention_time"][:top_n]
        
        # Prepare data
        names = [p["name"] for p in products]
        times = [p["total_attention_time"] for p in products]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data
        bars = ax.barh(names, times, color='skyblue')
        
        # Add values to bars
        for i, v in enumerate(times):
            ax.text(v + 0.1, i, f"{v:.1f}s", va='center')
            
        # Add labels and title
        ax.set_xlabel('Attention Time (seconds)')
        ax.set_ylabel('Product')
        ax.set_title('Product Attention Analysis')
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved product attention chart to {save_path}")
            
        # Show if requested
        if show_fig:
            plt.show()
        
        return fig
        
    def generate_attention_timeline(self, 
                                  bin_seconds: int = 30, 
                                  save_path: Optional[str] = None,
                                  show_fig: bool = True) -> plt.Figure:
        """
        Generate a timeline visualization of attention over time.
        
        Args:
            bin_seconds: Time bin size in seconds
            save_path: Path to save the chart
            show_fig: Whether to show the figure
            
        Returns:
            Matplotlib figure
        """
        # Get data as DataFrame
        df = self.to_dataframe()
        
        if df.empty:
            logger.error("No attention data available")
            return None
            
        # Group by time bins and product
        df['time_bin'] = pd.to_datetime(
            (df['timestamp'] // bin_seconds) * bin_seconds, 
            unit='s'
        )
        
        # Sum durations by product and time bin
        timeline_data = df.groupby(['time_bin', 'product_name'])['duration'].sum().reset_index()
        
        # Pivot for plotting
        pivot_data = timeline_data.pivot(
            index='time_bin', 
            columns='product_name', 
            values='duration'
        ).fillna(0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot data
        pivot_data.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Attention Duration (seconds)')
        ax.set_title(f'Attention Timeline (Bins: {bin_seconds}s)')
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved attention timeline to {save_path}")
            
        # Show if requested
        if show_fig:
            plt.show()
        
        return fig
        
    def generate_category_breakdown(self, 
                                  save_path: Optional[str] = None,
                                  show_fig: bool = True) -> plt.Figure:
        """
        Generate a pie chart of attention by product category.
        
        Args:
            save_path: Path to save the chart
            show_fig: Whether to show the figure
            
        Returns:
            Matplotlib figure
        """
        if not self.products:
            logger.error("No product data available")
            return None
            
        # Group by category
        category_data = {}
        
        for product in self.products:
            category = product.get("category", "Unknown")
            attention_time = product.get("total_attention_time", 0)
            
            if category in category_data:
                category_data[category] += attention_time
            else:
                category_data[category] = attention_time
                
        # Prepare data
        categories = list(category_data.keys())
        attention_times = list(category_data.values())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data
        wedges, texts, autotexts = ax.pie(
            attention_times, 
            labels=None, 
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Equal aspect ratio ensures the pie chart is circular
        ax.axis('equal')
        
        # Add legend
        ax.legend(
            wedges, 
            categories, 
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Add title
        ax.set_title('Attention by Product Category')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved category breakdown to {save_path}")
            
        # Show if requested
        if show_fig:
            plt.show()
        
        return fig
        
    def generate_heatmap_visualization(self, 
                                     save_path: Optional[str] = None,
                                     show_fig: bool = True) -> plt.Figure:
        """
        Generate a heatmap visualization of attention.
        
        Args:
            save_path: Path to save the chart
            show_fig: Whether to show the figure
            
        Returns:
            Matplotlib figure
        """
        if self.heatmap is None or np.all(self.heatmap == 0):
            logger.error("No heatmap data available")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Apply Gaussian smoothing for better visualization
        from scipy.ndimage import gaussian_filter
        smoothed_heatmap = gaussian_filter(self.heatmap, sigma=10)
        
        # Plot heatmap
        im = ax.imshow(
            smoothed_heatmap, 
            cmap='jet', 
            interpolation='bilinear',
            origin='upper'
        )
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Attention Intensity')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        ax.set_title('Attention Heatmap')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved heatmap visualization to {save_path}")
            
        # Show if requested
        if show_fig:
            plt.show()
        
        return fig
        
    def export_analytics_report(self, output_dir: str) -> str:
        """
        Generate a comprehensive analytics report with all charts.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the report directory
        """
        try:
            # Create report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(output_dir, f"report_{timestamp}")
            os.makedirs(report_dir, exist_ok=True)
            
            logger.info(f"Created report directory: {report_dir}")
            logger.info(f"Current metrics: {self.metrics}")
            
            # Check if metrics are empty and try to calculate them if we have attention data
            if not self.metrics and self.attention_history:
                logger.info("Metrics dictionary is empty but attention history exists. Calculating metrics...")
                self._calculate_metrics_from_history()
            # If still no metrics, but we have products loaded
            elif not self.metrics and self.products:
                logger.info("No metrics or attention history. Creating placeholder metrics from products...")
                self._create_placeholder_metrics()
            
            # Check if metrics are still empty
            if not self.metrics:
                logger.warning("Metrics dictionary is empty. Charts may not generate properly.")
                # Create minimal metrics to avoid errors
                self.metrics = {
                    "session_duration": 0,
                    "total_attention_time": 0,
                    "product_count": len(self.products) if self.products else 0,
                    "products_by_attention_time": [],
                    "products_by_attention_count": [],
                    "attention_percentages": {},
                    "ignored_products": [],
                    "ignored_percentage": 0,
                    "timestamp": datetime.now().timestamp()
                }
            
            # Generate charts
            try:
                logger.info("Generating product attention chart...")
                self.generate_product_attention_chart(
                    save_path=os.path.join(report_dir, "product_attention.png"),
                    show_fig=False
                )
                logger.info("Product attention chart generated successfully")
            except Exception as chart_err:
                logger.error(f"Failed to generate product attention chart: {chart_err}")
            
            try:
                logger.info("Generating attention timeline...")
                self.generate_attention_timeline(
                    save_path=os.path.join(report_dir, "attention_timeline.png"),
                    show_fig=False
                )
                logger.info("Attention timeline generated successfully")
            except Exception as chart_err:
                logger.error(f"Failed to generate attention timeline: {chart_err}")
            
            try:
                logger.info("Generating category breakdown...")
                self.generate_category_breakdown(
                    save_path=os.path.join(report_dir, "category_breakdown.png"),
                    show_fig=False
                )
                logger.info("Category breakdown generated successfully")
            except Exception as chart_err:
                logger.error(f"Failed to generate category breakdown: {chart_err}")
            
            try:
                logger.info("Generating heatmap visualization...")
                self.generate_heatmap_visualization(
                    save_path=os.path.join(report_dir, "heatmap.png"),
                    show_fig=False
                )
                logger.info("Heatmap visualization generated successfully")
            except Exception as chart_err:
                logger.error(f"Failed to generate heatmap visualization: {chart_err}")
            
            # Export metrics as JSON
            metrics_path = os.path.join(report_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            # Create HTML report
            logger.info("Generating HTML report...")
            html_report = self._generate_html_report(report_dir)
            
            if html_report:
                logger.info(f"HTML report generated successfully at: {html_report}")
            else:
                logger.error("HTML report generation returned empty path")
            
            logger.info(f"Analytics report exported to {report_dir}")
            return report_dir
            
        except Exception as e:
            logger.error(f"Error exporting analytics report: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""
            
    def _calculate_metrics_from_history(self) -> None:
        """
        Calculate metrics from attention history if available.
        This is a fallback method when metrics weren't loaded directly.
        """
        try:
            if not self.attention_history:
                logger.warning("No attention history available to calculate metrics")
                return
                
            logger.info(f"Calculating metrics from {len(self.attention_history)} attention history records")
            
            # First, ensure products are loaded
            if not self.products:
                logger.warning("No products available to calculate metrics")
                return
                
            # Create a dictionary to track product metrics
            product_metrics = {p.id: {"name": p.name, "category": p.category, "price": p.price, 
                                    "total_attention_time": 0, "attention_count": 0,
                                    "last_attention_time": 0} 
                            for p in self.products}
                
            # Calculate session duration
            timestamps = [record.get("timestamp", 0) for record in self.attention_history]
            if timestamps:
                session_start = min(timestamps)
                session_end = max(timestamps)
                session_duration = session_end - session_start
            else:
                session_duration = 0
                
            # Process attention history
            total_attention = 0
            
            for record in self.attention_history:
                product_id = record.get("product_id")
                duration = record.get("duration", 0)
                timestamp = record.get("timestamp", 0)
                
                if product_id and product_id in product_metrics:
                    product_metrics[product_id]["total_attention_time"] += duration
                    product_metrics[product_id]["attention_count"] += 1
                    product_metrics[product_id]["last_attention_time"] = max(
                        timestamp, product_metrics[product_id]["last_attention_time"]
                    )
                    total_attention += duration
            
            # Convert to lists for sorting
            products_list = [
                {"id": pid, **metrics} 
                for pid, metrics in product_metrics.items()
            ]
            
            # Sort by time and count
            sorted_by_time = sorted(
                products_list, 
                key=lambda p: p["total_attention_time"], 
                reverse=True
            )
            
            sorted_by_count = sorted(
                products_list, 
                key=lambda p: p["attention_count"], 
                reverse=True
            )
            
            # Calculate attention percentages
            attention_percentages = {}
            for p in products_list:
                pid = p["id"]
                if total_attention > 0:
                    attention_percentages[pid] = (p["total_attention_time"] / total_attention) * 100
                else:
                    attention_percentages[pid] = 0
            
            # Find ignored products
            ignored_products = [p for p in products_list if p["attention_count"] == 0]
            
            # Create metrics dictionary
            self.metrics = {
                "session_duration": session_duration,
                "total_attention_time": total_attention,
                "product_count": len(self.products),
                "products_by_attention_time": sorted_by_time,
                "products_by_attention_count": sorted_by_count,
                "attention_percentages": attention_percentages,
                "ignored_products": ignored_products,
                "ignored_percentage": (len(ignored_products) / len(self.products)) * 100 if self.products else 0,
                "timestamp": datetime.now().timestamp()
            }
            
            logger.info("Successfully calculated metrics from attention history")
            
        except Exception as e:
            logger.error(f"Error calculating metrics from history: {e}")
            logger.error(traceback.format_exc())
            
    def _create_placeholder_metrics(self) -> None:
        """
        Create placeholder metrics when no attention data is available.
        """
        try:
            if not self.products:
                logger.warning("No products available to create placeholder metrics")
                return
                
            logger.info(f"Creating placeholder metrics for {len(self.products)} products")
            
            # Create empty product metrics
            products_list = [
                {
                    "id": p.id,
                    "name": p.name,
                    "category": p.category,
                    "price": p.price,
                    "total_attention_time": 0,
                    "attention_count": 0,
                    "last_attention_time": 0
                }
                for p in self.products
            ]
            
            # Create metrics dictionary with zeros
            self.metrics = {
                "session_duration": 0,
                "total_attention_time": 0,
                "product_count": len(self.products),
                "products_by_attention_time": products_list,
                "products_by_attention_count": products_list,
                "attention_percentages": {p.id: 0 for p in self.products},
                "ignored_products": products_list,
                "ignored_percentage": 100,  # All products are ignored
                "timestamp": datetime.now().timestamp()
            }
            
            logger.info("Successfully created placeholder metrics")
            
        except Exception as e:
            logger.error(f"Error creating placeholder metrics: {e}")
            logger.error(traceback.format_exc())
        
    def _generate_html_report(self, report_dir: str) -> str:
        """
        Generate an HTML report summarizing the analytics.
        
        Args:
            report_dir: Directory containing report files
            
        Returns:
            Path to the HTML report file
        """
        try:
            logger.info("Starting HTML report generation...")
            # Define HTML template with a raw string to avoid indentation issues
            html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Retail Shelf Attention Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
        }
        .header {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 30px;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f5f5f5;
        }
        .highlight {
            background-color: #FFFFE0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Retail Shelf Attention Analysis Report</h1>
        <p>Generated on: {date}</p>
        <p>Session Duration: {session_duration}</p>
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
        <ul>
            <li>Total products analyzed: {product_count}</li>
            <li>Total attention time: {total_attention:.2f} seconds</li>
            <li>Products receiving no attention: {ignored_count} ({ignored_percentage:.1f}%)</li>
            <li>Most viewed product: {top_product_name} ({top_product_views} views)</li>
            <li>Product with longest attention: {longest_attention_product} ({longest_attention:.2f} seconds)</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Product Attention Analysis</h2>
        <div class="chart-container">
            <img class="chart" src="product_attention.png" alt="Product Attention Chart">
        </div>
        
        <h3>Top 5 Products by Attention Time</h3>
        <table>
            <tr>
                <th>Rank</th>
                <th>Product</th>
                <th>Attention Time (s)</th>
                <th>View Count</th>
                <th>Average Time per View (s)</th>
            </tr>
            {top_products_table}
        </table>
    </div>
    
    <div class="section">
        <h2>Attention Timeline</h2>
        <div class="chart-container">
            <img class="chart" src="attention_timeline.png" alt="Attention Timeline">
        </div>
        <p>This chart shows how attention shifted between products over time during the session.</p>
    </div>
    
    <div class="section">
        <h2>Category Analysis</h2>
        <div class="chart-container">
            <img class="chart" src="category_breakdown.png" alt="Category Breakdown">
        </div>
    </div>
    
    <div class="section">
        <h2>Attention Heatmap</h2>
        <div class="chart-container">
            <img class="chart" src="heatmap.png" alt="Attention Heatmap">
        </div>
        <p>The heatmap visualization shows the areas of the shelf that received the most attention.</p>
    </div>
    
    <div class="section">
        <h2>Products Receiving No Attention</h2>
        <table>
            <tr>
                <th>Product</th>
                <th>Category</th>
                <th>Price</th>
            </tr>
            {ignored_products_table}
        </table>
    </div>
</body>
</html>"""
            
            # Format date
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get key metrics
            logger.info(f"Preparing metrics for HTML report, metrics: {self.metrics}")
            session_duration = self.metrics.get("session_duration", 0)
            session_duration_formatted = f"{session_duration:.2f} seconds"
            if session_duration > 60:
                session_duration_formatted = f"{session_duration/60:.2f} minutes"
                
            product_count = self.metrics.get("product_count", 0)
            total_attention = self.metrics.get("total_attention_time", 0)
            
            ignored_products = self.metrics.get("ignored_products", [])
            ignored_count = len(ignored_products)
            ignored_percentage = self.metrics.get("ignored_percentage", 0)
            
            # Get top products
            products_by_time = self.metrics.get("products_by_attention_time", [])
            products_by_count = self.metrics.get("products_by_attention_count", [])
            
            logger.info(f"products_by_time: {products_by_time}")
            logger.info(f"products_by_count: {products_by_count}")
            
            # Use try-except to handle potential IndexError
            try:
                top_product_name = products_by_count[0]["name"] if products_by_count else "None"
                top_product_views = products_by_count[0]["attention_count"] if products_by_count else 0
            except (IndexError, KeyError) as e:
                logger.error(f"Error getting top product data: {e}")
                top_product_name = "No Data"
                top_product_views = 0
            
            try:
                longest_attention_product = products_by_time[0]["name"] if products_by_time else "None"
                longest_attention = products_by_time[0]["total_attention_time"] if products_by_time else 0
            except (IndexError, KeyError) as e:
                logger.error(f"Error getting longest attention product data: {e}")
                longest_attention_product = "No Data"
                longest_attention = 0
            
            # Generate top products table rows
            top_products_table = ""
            try:
                for i, product in enumerate(products_by_time[:5]):
                    avg_time = product["total_attention_time"] / product["attention_count"] if product["attention_count"] > 0 else 0
                    top_products_table += f"""<tr>
                <td>{i+1}</td>
                <td>{product["name"]}</td>
                <td>{product["total_attention_time"]:.2f}</td>
                <td>{product["attention_count"]}</td>
                <td>{avg_time:.2f}</td>
            </tr>"""
            except Exception as e:
                logger.error(f"Error generating top products table: {e}")
                top_products_table = """<tr>
                <td colspan="5" style="text-align: center;">Error generating product data</td>
            </tr>"""
                
            # Generate ignored products table rows
            ignored_products_table = ""
            try:
                for product in ignored_products:
                    ignored_products_table += f"""<tr>
                <td>{product["name"]}</td>
                <td>{product["category"]}</td>
                <td>${product["price"]:.2f}</td>
            </tr>"""
                    
                if not ignored_products:
                    ignored_products_table = """<tr>
                <td colspan="3" style="text-align: center;">No ignored products</td>
            </tr>"""
            except Exception as e:
                logger.error(f"Error generating ignored products table: {e}")
                ignored_products_table = """<tr>
                <td colspan="3" style="text-align: center;">Error generating ignored products data</td>
            </tr>"""
                
            # Format HTML
            try:
                logger.info("Formatting HTML content...")
                html_content = html_template.format(
                    date=current_date,
                    session_duration=session_duration_formatted,
                    product_count=product_count,
                    total_attention=total_attention,
                    ignored_count=ignored_count,
                    ignored_percentage=ignored_percentage,
                    top_product_name=top_product_name,
                    top_product_views=top_product_views,
                    longest_attention_product=longest_attention_product,
                    longest_attention=longest_attention,
                    top_products_table=top_products_table,
                    ignored_products_table=ignored_products_table
                )
            except Exception as e:
                logger.error(f"Error formatting HTML content: {e}")
                raise
            
            # Write to file
            report_path = os.path.join(report_dir, "report.html")
            logger.info(f"Writing HTML content to {report_path}")
            with open(report_path, 'w') as f:
                f.write(html_content)
                
            logger.info(f"Generated HTML report at {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "" 