#!/usr/bin/env python3
"""
Visual Overlay for Dictator
Shows a small, discreet colored circle in the corner of the screen
"""

import tkinter as tk
import threading
from enum import Enum


class OverlayState(Enum):
    """Recording states"""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"  # TTS speaking state


class VisualOverlay:
    """Small visual indicator overlay"""

    def __init__(self, size=15, position="top-right", padding=20):
        """
        Initialize overlay

        Args:
            size: Diameter of the circle in pixels (default: 15)
            position: Corner position - "top-right", "top-left", "bottom-right", "bottom-left"
            padding: Distance from screen edge in pixels (default: 20)
        """
        self.size = size
        self.position = position
        self.padding = padding
        self.current_state = OverlayState.IDLE

        self.root = None
        self.canvas = None
        self.circle = None
        self.running = False

        # Colors for each state
        self.colors = {
            OverlayState.IDLE: None,  # Hidden
            OverlayState.RECORDING: "#FF4444",  # Red
            OverlayState.PROCESSING: "#FFAA00",  # Yellow/Orange
            OverlayState.SPEAKING: "#44FF44"  # Green
        }

    def _create_window(self):
        """Create the overlay window"""
        self.root = tk.Tk()

        # Remove window decorations
        self.root.overrideredirect(True)

        # Make window transparent
        self.root.attributes('-alpha', 0.8)  # 80% opacity
        self.root.attributes('-topmost', True)  # Always on top

        # Set window size
        self.root.geometry(f"{self.size}x{self.size}")

        # Position window
        self._position_window()

        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.size,
            height=self.size,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()

        # Make black transparent (only circle visible)
        self.root.wm_attributes('-transparentcolor', 'black')

        # Create circle (initially hidden)
        radius = self.size // 2
        self.circle = self.canvas.create_oval(
            2, 2, self.size-2, self.size-2,
            fill=self.colors[OverlayState.RECORDING],
            outline=""
        )

        # Start hidden
        self.root.withdraw()

    def _position_window(self):
        """Position window at specified corner"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        if self.position == "top-right":
            x = screen_width - self.size - self.padding
            y = self.padding
        elif self.position == "top-left":
            x = self.padding
            y = self.padding
        elif self.position == "bottom-right":
            x = screen_width - self.size - self.padding
            y = screen_height - self.size - self.padding
        elif self.position == "bottom-left":
            x = self.padding
            y = screen_height - self.size - self.padding
        else:
            # Default to top-right
            x = screen_width - self.size - self.padding
            y = self.padding

        self.root.geometry(f"+{x}+{y}")

    def _run(self):
        """Run the overlay window (blocks)"""
        self._create_window()
        self.running = True
        self.root.mainloop()

    def start(self):
        """Start the overlay in a background thread"""
        if self.running:
            return

        # Run in background thread
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

        # Wait a bit for window to be created
        import time
        time.sleep(0.2)

    def set_state(self, state: OverlayState):
        """
        Update overlay state

        Args:
            state: New state (IDLE, RECORDING, PROCESSING)
        """
        if not self.running or not self.root:
            return

        self.current_state = state

        def update():
            if state == OverlayState.IDLE:
                # Hide overlay
                self.root.withdraw()
            else:
                # Show overlay with appropriate color
                color = self.colors[state]
                self.canvas.itemconfig(self.circle, fill=color)
                self.root.deiconify()

        # Schedule update in main thread
        if self.root:
            self.root.after(0, update)

    def stop(self):
        """Stop and close the overlay"""
        if not self.running or not self.root:
            return

        def close():
            self.root.quit()
            self.root.destroy()

        self.root.after(0, close)
        self.running = False


# Convenience functions
_global_overlay = None


def init_overlay(size=15, position="top-right", padding=20):
    """Initialize global overlay instance"""
    global _global_overlay
    _global_overlay = VisualOverlay(size=size, position=position, padding=padding)
    _global_overlay.start()
    return _global_overlay


def set_state(state: OverlayState):
    """Set global overlay state"""
    if _global_overlay:
        _global_overlay.set_state(state)


def stop_overlay():
    """Stop global overlay"""
    if _global_overlay:
        _global_overlay.stop()
