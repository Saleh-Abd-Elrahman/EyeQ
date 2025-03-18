"""
Dashboard UI for the smart eye-tracking retail system.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTabWidget, QGroupBox, QComboBox, 
    QCheckBox, QSlider, QFileDialog, QMessageBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoThread(QThread):
    """Thread for processing video frames."""
    
    update_frame = pyqtSignal(np.ndarray)
    update_tracking = pyqtSignal(dict)
    update_analysis = pyqtSignal(dict)
    
    def __init__(self, camera, eye_tracker, shelf_analyzer):
        """
        Initialize the video processing thread.
        
        Args:
            camera: Camera object for video capture
            eye_tracker: Eye tracker object for gaze detection
            shelf_analyzer: Shelf analyzer object for product attention
        """
        super().__init__()
        self.camera = camera
        self.eye_tracker = eye_tracker
        self.shelf_analyzer = shelf_analyzer
        self.running = False
        self.paused = False
        self.show_heatmap = False
        self.show_product_regions = True
        self.show_gaze = True
        self.show_mesh = False
        
    def run(self):
        """Main processing loop."""
        self.running = True
        self.camera.start()
        
        while self.running:
            if not self.paused:
                # Read frame from camera
                success, frame = self.camera.read()
                
                if not success:
                    logger.warning("Failed to read frame")
                    time.sleep(0.05)  # Avoid busy waiting
                    continue
                
                # Process frame with eye tracker
                tracking_result = self.eye_tracker.process_frame(frame)
                
                # Apply visualizations as requested
                if tracking_result["success"]:
                    # Process gaze point with shelf analyzer if available
                    if "gaze_point" in tracking_result and tracking_result["gaze_point"] is not None:
                        analysis_result = self.shelf_analyzer.process_gaze(tracking_result["gaze_point"])
                        self.update_analysis.emit(analysis_result)
                
                # Add visualizations
                annotated_frame = frame.copy()
                
                # Add eye tracking visualization
                if self.show_gaze or self.show_mesh:
                    annotated_frame = self.eye_tracker.visualize(
                        annotated_frame, 
                        tracking_result,
                        show_mesh=self.show_mesh,
                        show_gaze=self.show_gaze
                    )
                
                # Add product regions
                if self.show_product_regions:
                    annotated_frame = self.shelf_analyzer.draw_product_regions(annotated_frame)
                
                # Add heatmap
                if self.show_heatmap:
                    annotated_frame = self.shelf_analyzer.generate_heatmap_visualization(annotated_frame)
                
                # Emit signals
                self.update_frame.emit(annotated_frame)
                self.update_tracking.emit(tracking_result)
            
            # Sleep to control frame rate
            time.sleep(0.01)
            
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()
        if self.camera:
            self.camera.stop()
            
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        return self.paused
        
    def toggle_heatmap(self):
        """Toggle heatmap visualization."""
        self.show_heatmap = not self.show_heatmap
        return self.show_heatmap
        
    def toggle_product_regions(self):
        """Toggle product region visualization."""
        self.show_product_regions = not self.show_product_regions
        return self.show_product_regions
        
    def toggle_gaze(self):
        """Toggle gaze visualization."""
        self.show_gaze = not self.show_gaze
        return self.show_gaze
        
    def toggle_mesh(self):
        """Toggle face mesh visualization."""
        self.show_mesh = not self.show_mesh
        return self.show_mesh


class Dashboard(QMainWindow):
    """Main dashboard window for the eye-tracking retail system."""
    
    def __init__(self, camera, eye_tracker, shelf_analyzer, analytics):
        """
        Initialize the dashboard.
        
        Args:
            camera: Camera object for video capture
            eye_tracker: Eye tracker object for gaze detection
            shelf_analyzer: Shelf analyzer object for product attention
            analytics: Analytics object for data visualization
        """
        super().__init__()
        
        self.camera = camera
        self.eye_tracker = eye_tracker
        self.shelf_analyzer = shelf_analyzer
        self.analytics = analytics
        
        self.video_thread = None
        self.recording = False
        self.session_start_time = time.time()
        
        self.init_ui()
        self.setup_shortcuts()
        self.start_video_thread()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("Smart Eye-Tracking Retail Analytics")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Create left panel for video and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        left_layout.addWidget(self.video_label)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        controls_layout.addWidget(self.pause_button)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_button)
        
        self.heatmap_button = QPushButton("Show Heatmap")
        self.heatmap_button.clicked.connect(self.toggle_heatmap)
        controls_layout.addWidget(self.heatmap_button)
        
        self.regions_button = QPushButton("Hide Regions")
        self.regions_button.clicked.connect(self.toggle_product_regions)
        controls_layout.addWidget(self.regions_button)
        
        self.reset_button = QPushButton("Reset Data")
        self.reset_button.clicked.connect(self.reset_data)
        controls_layout.addWidget(self.reset_button)
        
        self.export_button = QPushButton("Export Data")
        self.export_button.clicked.connect(self.export_data)
        controls_layout.addWidget(self.export_button)
        
        left_layout.addLayout(controls_layout)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Status: Ready")
        status_layout.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: -")
        status_layout.addWidget(self.fps_label)
        
        self.detection_label = QLabel("Detection: -")
        status_layout.addWidget(self.detection_label)
        
        self.session_time_label = QLabel("Session Time: 00:00")
        status_layout.addWidget(self.session_time_label)
        
        left_layout.addLayout(status_layout)
        
        # Right panel with tabs
        right_panel = QTabWidget()
        
        # Product attention tab
        product_tab = QWidget()
        product_layout = QVBoxLayout()
        product_tab.setLayout(product_layout)
        
        # Product table
        self.product_table = QTableWidget(0, 5)
        self.product_table.setHorizontalHeaderLabels([
            "Product", "Attention Time (s)", "Views", "Avg Time/View (s)", "Last Viewed"
        ])
        self.product_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        product_layout.addWidget(self.product_table)
        
        right_panel.addTab(product_tab, "Product Attention")
        
        # Analytics tab
        analytics_tab = QWidget()
        analytics_layout = QVBoxLayout()
        analytics_tab.setLayout(analytics_layout)
        
        # Analytics options
        options_group = QGroupBox("Analytics Options")
        options_layout = QHBoxLayout()
        options_group.setLayout(options_layout)
        
        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "Product Attention", "Attention Timeline", "Category Breakdown", "Attention Heatmap"
        ])
        self.chart_combo.currentIndexChanged.connect(self.update_analytics_chart)
        options_layout.addWidget(QLabel("Chart Type:"))
        options_layout.addWidget(self.chart_combo)
        
        self.generate_report_button = QPushButton("Generate Report")
        self.generate_report_button.clicked.connect(self.generate_analytics_report)
        options_layout.addWidget(self.generate_report_button)
        
        analytics_layout.addWidget(options_group)
        
        # Chart placeholder
        self.chart_label = QLabel("Select an analysis type to generate a chart")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setMinimumHeight(500)
        self.chart_label.setStyleSheet("background-color: white; border: 1px solid #ddd;")
        analytics_layout.addWidget(self.chart_label)
        
        right_panel.addTab(analytics_tab, "Analytics")
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        settings_tab.setLayout(settings_layout)
        
        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)
        
        # Camera selector
        camera_selector_layout = QHBoxLayout()
        camera_selector_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Default Camera (0)")
        camera_selector_layout.addWidget(self.camera_combo)
        
        self.refresh_cameras_button = QPushButton("Refresh")
        self.refresh_cameras_button.clicked.connect(self.refresh_cameras)
        camera_selector_layout.addWidget(self.refresh_cameras_button)
        
        camera_layout.addLayout(camera_selector_layout)
        
        # Visualization options
        viz_group = QGroupBox("Visualization Settings")
        viz_layout = QVBoxLayout()
        viz_group.setLayout(viz_layout)
        
        self.show_gaze_checkbox = QCheckBox("Show Gaze Point")
        self.show_gaze_checkbox.setChecked(True)
        self.show_gaze_checkbox.stateChanged.connect(self.toggle_gaze_viz)
        viz_layout.addWidget(self.show_gaze_checkbox)
        
        self.show_mesh_checkbox = QCheckBox("Show Face Mesh")
        self.show_mesh_checkbox.setChecked(False)
        self.show_mesh_checkbox.stateChanged.connect(self.toggle_mesh_viz)
        viz_layout.addWidget(self.show_mesh_checkbox)
        
        self.show_metrics_checkbox = QCheckBox("Show Metrics on Products")
        self.show_metrics_checkbox.setChecked(True)
        viz_layout.addWidget(self.show_metrics_checkbox)
        
        settings_layout.addWidget(camera_group)
        settings_layout.addWidget(viz_group)
        
        # Add spacer at the bottom
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        settings_layout.addWidget(spacer)
        
        right_panel.addTab(settings_tab, "Settings")
        
        # Set up the main layout with a splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 800])
        main_layout.addWidget(splitter)
        
        # Set up timer for UI updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(1000)  # Update every second
        
    def start_video_thread(self):
        """Start the video processing thread."""
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            
        self.video_thread = VideoThread(self.camera, self.eye_tracker, self.shelf_analyzer)
        self.video_thread.update_frame.connect(self.update_frame)
        self.video_thread.update_tracking.connect(self.update_tracking_info)
        self.video_thread.update_analysis.connect(self.update_analysis_info)
        self.video_thread.start()
        
    def update_frame(self, frame):
        """Update the video display with a new frame."""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        # Convert to QImage
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        # Scale to fit in the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
    def update_tracking_info(self, tracking_result):
        """Update tracking information display."""
        if tracking_result["success"]:
            self.detection_label.setText(f"Detection: Yes")
            self.fps_label.setText(f"FPS: {1000 / tracking_result['process_time_ms']:.1f}")
        else:
            self.detection_label.setText("Detection: No")
            
    def update_analysis_info(self, analysis_result):
        """Update analysis information display."""
        # Update status
        if analysis_result.get("gazed_product"):
            product = analysis_result["gazed_product"]
            self.status_label.setText(f"Looking at: {product['name']} (${product['price']:.2f})")
        else:
            self.status_label.setText("Status: No product in focus")
            
        # Update product table
        self.update_product_table()
        
    def update_product_table(self):
        """Update the product table with current data."""
        products = self.shelf_analyzer.products
        
        # Sort by attention time
        sorted_products = sorted(products, key=lambda p: p.total_attention_time, reverse=True)
        
        # Update table
        self.product_table.setRowCount(len(sorted_products))
        
        for i, product in enumerate(sorted_products):
            # Product name
            self.product_table.setItem(i, 0, QTableWidgetItem(product.name))
            
            # Attention time
            time_item = QTableWidgetItem(f"{product.total_attention_time:.1f}")
            time_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.product_table.setItem(i, 1, time_item)
            
            # View count
            views_item = QTableWidgetItem(str(product.attention_count))
            views_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.product_table.setItem(i, 2, views_item)
            
            # Average time per view
            avg_time = product.total_attention_time / product.attention_count if product.attention_count > 0 else 0
            avg_item = QTableWidgetItem(f"{avg_time:.2f}")
            avg_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.product_table.setItem(i, 3, avg_item)
            
            # Last viewed
            if product.last_attention_time > 0:
                seconds_ago = time.time() - product.last_attention_time
                if seconds_ago < 60:
                    last_viewed = f"{seconds_ago:.0f}s ago"
                else:
                    last_viewed = f"{seconds_ago/60:.1f}m ago"
            else:
                last_viewed = "Never"
                
            self.product_table.setItem(i, 4, QTableWidgetItem(last_viewed))
            
            # Highlight current attention
            if product.is_current_attention:
                for col in range(5):
                    item = self.product_table.item(i, col)
                    item.setBackground(Qt.green)
        
    def update_ui(self):
        """Periodic updates for UI elements."""
        # Update session time
        session_time = time.time() - self.session_start_time
        minutes = int(session_time // 60)
        seconds = int(session_time % 60)
        self.session_time_label.setText(f"Session Time: {minutes:02d}:{seconds:02d}")
        
    def toggle_pause(self):
        """Toggle video processing pause state."""
        if self.video_thread:
            paused = self.video_thread.toggle_pause()
            self.pause_button.setText("Resume" if paused else "Pause")
            
    def toggle_recording(self):
        """Toggle data recording state."""
        self.recording = not self.recording
        
        if self.recording:
            self.record_button.setText("Stop Recording")
            self.status_label.setText("Status: Recording data")
        else:
            self.record_button.setText("Start Recording")
            self.status_label.setText("Status: Recording stopped")
            
    def toggle_heatmap(self):
        """Toggle heatmap visualization."""
        if self.video_thread:
            show_heatmap = self.video_thread.toggle_heatmap()
            self.heatmap_button.setText("Hide Heatmap" if show_heatmap else "Show Heatmap")
            
    def toggle_product_regions(self):
        """Toggle product region visualization."""
        if self.video_thread:
            show_regions = self.video_thread.toggle_product_regions()
            self.regions_button.setText("Show Regions" if not show_regions else "Hide Regions")
            
    def toggle_gaze_viz(self):
        """Toggle gaze visualization."""
        if self.video_thread:
            self.video_thread.show_gaze = self.show_gaze_checkbox.isChecked()
            
    def toggle_mesh_viz(self):
        """Toggle face mesh visualization."""
        if self.video_thread:
            self.video_thread.show_mesh = self.show_mesh_checkbox.isChecked()
            
    def reset_data(self):
        """Reset all tracking and analysis data."""
        reply = QMessageBox.question(
            self, 
            'Reset Data', 
            'Are you sure you want to reset all attention data? This cannot be undone.',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.shelf_analyzer.reset()
            self.session_start_time = time.time()
            self.status_label.setText("Status: Data reset")
            self.update_product_table()
            
    def export_data(self):
        """Export the current session data."""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Output Directory",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        )
        
        if directory:
            try:
                output_dir = self.shelf_analyzer.export_session_data(directory)
                
                if output_dir:
                    QMessageBox.information(
                        self, 
                        "Export Successful", 
                        f"Data exported to:\n{output_dir}"
                    )
                else:
                    QMessageBox.warning(
                        self, 
                        "Export Failed", 
                        "Could not export data. Check console for details."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Export Error", 
                    f"Error exporting data: {str(e)}"
                )
                
    def update_analytics_chart(self):
        """Update the analytics chart based on selected type."""
        chart_type = self.chart_combo.currentText()
        
        try:
            if chart_type == "Product Attention":
                # Generate chart to a temporary file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_file.close()
                
                self.analytics.generate_product_attention_chart(
                    save_path=temp_file.name,
                    show_fig=False
                )
                
                # Load and display the chart
                pixmap = QPixmap(temp_file.name)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                
                # Clean up
                os.unlink(temp_file.name)
                
            elif chart_type == "Attention Timeline":
                # Similar approach for other chart types
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_file.close()
                
                self.analytics.generate_attention_timeline(
                    save_path=temp_file.name,
                    show_fig=False
                )
                
                pixmap = QPixmap(temp_file.name)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                
                os.unlink(temp_file.name)
                
            elif chart_type == "Category Breakdown":
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_file.close()
                
                self.analytics.generate_category_breakdown(
                    save_path=temp_file.name,
                    show_fig=False
                )
                
                pixmap = QPixmap(temp_file.name)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                
                os.unlink(temp_file.name)
                
            elif chart_type == "Attention Heatmap":
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_file.close()
                
                self.analytics.generate_heatmap_visualization(
                    save_path=temp_file.name,
                    show_fig=False
                )
                
                pixmap = QPixmap(temp_file.name)
                self.chart_label.setPixmap(pixmap.scaled(
                    self.chart_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                
                os.unlink(temp_file.name)
                
        except Exception as e:
            self.chart_label.setText(f"Error generating chart: {str(e)}")
            logger.error(f"Error generating chart: {e}")
            
    def generate_analytics_report(self):
        """Generate and open a comprehensive analytics report."""
        try:
            directory = QFileDialog.getExistingDirectory(
                self, 
                "Select Output Directory for Report",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
            )
            
            if directory:
                report_dir = self.analytics.export_analytics_report(directory)
                
                if report_dir:
                    report_path = os.path.join(report_dir, "report.html")
                    
                    reply = QMessageBox.information(
                        self, 
                        "Report Generated", 
                        f"Report generated at:\n{report_path}\n\nOpen report now?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    
                    if reply == QMessageBox.Yes:
                        import webbrowser
                        webbrowser.open(f"file://{report_path}")
                else:
                    QMessageBox.warning(
                        self, 
                        "Report Generation Failed", 
                        "Could not generate report. Check console for details."
                    )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Report Error", 
                f"Error generating report: {str(e)}"
            )
            
    def refresh_cameras(self):
        """Refresh the list of available cameras."""
        from utils.camera import list_available_cameras
        
        self.camera_combo.clear()
        
        try:
            cameras = list_available_cameras()
            
            if cameras:
                for i, cam_idx in enumerate(cameras):
                    self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
            else:
                self.camera_combo.addItem("No cameras detected")
                
        except Exception as e:
            self.camera_combo.addItem("Error detecting cameras")
            logger.error(f"Error refreshing cameras: {e}")
            
    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtWidgets import QShortcut
        
        # Pause/Resume - Space
        pause_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        pause_shortcut.activated.connect(self.toggle_pause)
        
        # Start/Stop Recording - R
        record_shortcut = QShortcut(QKeySequence(Qt.Key_R), self)
        record_shortcut.activated.connect(self.toggle_recording)
        
        # Toggle Heatmap - H
        heatmap_shortcut = QShortcut(QKeySequence(Qt.Key_H), self)
        heatmap_shortcut.activated.connect(self.toggle_heatmap)
        
        # Save Snapshot - S
        snapshot_shortcut = QShortcut(QKeySequence(Qt.Key_S), self)
        snapshot_shortcut.activated.connect(self.export_data)
        
        # Exit - Esc
        exit_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        exit_shortcut.activated.connect(self.close)
        
    def closeEvent(self, event):
        """Handle window close event."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            
        event.accept()


def launch_dashboard(camera, eye_tracker, shelf_analyzer, analytics):
    """
    Launch the dashboard application.
    
    Args:
        camera: Camera object for video capture
        eye_tracker: Eye tracker object for gaze detection
        shelf_analyzer: Shelf analyzer object for product attention
        analytics: Analytics object for data visualization
        
    Returns:
        None
    """
    app = QApplication(sys.argv)
    window = Dashboard(camera, eye_tracker, shelf_analyzer, analytics)
    window.show()
    sys.exit(app.exec_()) 