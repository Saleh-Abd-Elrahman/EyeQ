import threading
import time
import logging
from pynput.mouse import Controller
from pynput.keyboard import Listener, Key

logger = logging.getLogger(__name__)

class GazeCursor:
    def __init__(self, size=20, color=(255, 0, 0), alpha=0.7):
        """
        Initialize a gaze cursor using the system cursor.
        
        Args:
            size: Not used for system cursor
            color: Not used for system cursor
            alpha: Not used for system cursor
        """
        self.position = (100, 100)  # Default position
        self.visible = False
        self.running = False
        self.thread = None
        self.mouse = Controller()
        self.original_position = (0, 0)
        self.control_enabled = True  # Flag to enable/disable cursor control
        
        # Start keyboard listener for the escape key
        self.keyboard_listener = Listener(on_press=self._on_key_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()
        
        logger.info("Gaze cursor initialized using system cursor")
        logger.info("Press ESC to toggle cursor control on/off")
    
    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            if key == Key.esc:
                # Toggle cursor control
                self.control_enabled = not self.control_enabled
                if self.control_enabled:
                    logger.info("Cursor control enabled - eye tracking now controls cursor")
                else:
                    logger.info("Cursor control disabled - you can now use your mouse normally")
                    # Restore original position when disabling
                    if hasattr(self, 'original_position_saved') and self.original_position_saved:
                        self.mouse.position = self.original_position
        except Exception as e:
            logger.error(f"Error handling key press: {e}")
    
    def update_position(self, x, y):
        """Update the cursor position."""
        self.position = (x, y)
        if self.visible and self.control_enabled:
            try:
                # Update system cursor position
                self.mouse.position = (x, y)
            except Exception as e:
                logger.error(f"Failed to update cursor position: {e}")
    
    def show(self):
        """Show the cursor by enabling updates."""
        if not self.visible:
            self.visible = True
            # Store original mouse position when first showing
            if not hasattr(self, 'original_position_saved') or not self.original_position_saved:
                self.original_position = self.mouse.position
                self.original_position_saved = True
            logger.info(f"Showing gaze cursor at position {self.position}")
    
    def hide(self):
        """Hide the cursor (not actually hiding, just stopping updates)."""
        if self.visible:
            self.visible = False
            logger.info("Hiding gaze cursor")
    
    def start(self):
        """Start the cursor update thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Gaze cursor thread started")
    
    def stop(self):
        """Stop the cursor update thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            # Restore original mouse position
            if hasattr(self, 'original_position_saved') and self.original_position_saved:
                self.mouse.position = self.original_position
            # Stop keyboard listener
            if hasattr(self, 'keyboard_listener'):
                self.keyboard_listener.stop()
            logger.info("Gaze cursor stopped")
        except Exception as e:
            logger.error(f"Error stopping cursor: {e}")
    
    def _update_loop(self):
        """Main update loop for the cursor."""
        logger.info("Gaze cursor update loop started")
        while self.running:
            try:
                if self.visible and self.control_enabled:
                    self.mouse.position = self.position
                time.sleep(0.01)  # Small delay to reduce CPU usage
            except Exception as e:
                logger.error(f"Error in cursor update loop: {e}")
                time.sleep(0.1)  # Add a delay to prevent rapid error logging