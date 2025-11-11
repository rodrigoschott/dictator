#!/usr/bin/env python3
"""
Dictator System Tray Icon
Provides visual control for the Dictator service
"""

import sys
import threading
from pathlib import Path

import yaml
import pystray
from PIL import Image, ImageDraw
from pystray import MenuItem as item

from dictator.service import DictatorService
from dictator.overlay import OverlayState, init_overlay, set_state as set_overlay_state


class DictatorTray:
    """System tray icon for Dictator"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize tray icon"""
        self.config_path = config_path
        self.service = DictatorService(config_path)
        self.icon = None
        self.running = False
        self.current_state = "idle"
        self.overlay = None

        # Register state callback
        self.service.register_state_callback(self.on_state_change)

    def save_config(self):
        """Save current config to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.service.config, f, default_flow_style=False, allow_unicode=True)
            self.service.logger.info("💾 Configuration saved")
        except Exception as e:
            self.service.logger.error(f"❌ Failed to save config: {e}")

    def create_icon_image(self, color: str = "white") -> Image.Image:
        """Create microphone icon with specified color"""
        # Create a simple microphone icon
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)

        # Draw microphone
        # Body
        draw.ellipse([20, 15, 44, 35], fill=color, outline=color)
        draw.rectangle([20, 25, 44, 40], fill=color, outline=color)

        # Stand
        draw.rectangle([30, 40, 34, 50], fill=color, outline=color)
        draw.ellipse([25, 48, 39, 52], fill=color, outline=color)

        return image

    def on_state_change(self, state: str):
        """Handle state changes from service"""
        self.current_state = state

        # Update tray icon color
        color_map = {
            "idle": "white",
            "recording": "#FF4444",  # Red
            "processing": "#FFAA00",  # Yellow/Orange
            "tts_speaking": "#44FF44",  # Green (TTS)
            "tts_idle": "white",
            "tts_stopping": "white"
        }

        if self.icon and state in color_map:
            new_icon = self.create_icon_image(color_map[state])
            self.icon.icon = new_icon

        # Update overlay
        overlay_state_map = {
            "idle": OverlayState.IDLE,
            "recording": OverlayState.RECORDING,
            "processing": OverlayState.PROCESSING,
            "tts_speaking": OverlayState.SPEAKING,
            "tts_idle": OverlayState.IDLE,
            "tts_stopping": OverlayState.IDLE
        }

        if state in overlay_state_map:
            set_overlay_state(overlay_state_map[state])

    def toggle_vad(self):
        """Toggle VAD (Voice Activity Detection)"""
        current = self.service.config['hotkey'].get('vad_enabled', False)
        self.service.config['hotkey']['vad_enabled'] = not current
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu
        self.service.logger.info(f"VAD: {'Enabled' if not current else 'Disabled'}")

    def toggle_push_to_talk(self):
        """Toggle between push-to-talk and toggle mode"""
        current = self.service.config['hotkey'].get('mode', 'toggle')
        new_mode = 'push_to_talk' if current == 'toggle' else 'toggle'
        self.service.config['hotkey']['mode'] = new_mode
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu
        self.service.logger.info(f"Mode: {new_mode}")

    def set_mouse_button(self, button):
        """Set mouse button trigger"""
        self.service.config['hotkey']['type'] = 'mouse'
        self.service.config['hotkey']['mouse_button'] = button
        self.save_config()
        self.restart_service()

    def set_keyboard(self):
        """Set keyboard trigger"""
        self.service.config['hotkey']['type'] = 'keyboard'
        self.save_config()
        self.restart_service()

    def create_menu(self):
        """Create tray menu"""
        # Get current settings
        trigger_type = self.service.config['hotkey'].get('type', 'keyboard')
        mouse_button = self.service.config['hotkey'].get('mouse_button', 'side1')
        vad_enabled = self.service.config['hotkey'].get('vad_enabled', False)
        mode = self.service.config['hotkey'].get('mode', 'toggle')
        is_push_to_talk = (mode == 'push_to_talk')

        # Trigger info text
        if trigger_type == 'mouse':
            trigger_text = f"Trigger: Mouse {mouse_button}"
        else:
            trigger_text = f"Trigger: {self.service.config['hotkey'].get('keyboard_trigger', 'ctrl+alt+v')}"

        return pystray.Menu(
            item(
                'Dictator Service',
                lambda: None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            item(
                trigger_text,
                lambda: None,
                enabled=False
            ),
            item(
                f"Model: {self.service.config['whisper']['model']}",
                lambda: None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            # Hotkey selection submenu
            item(
                'Change Trigger',
                pystray.Menu(
                    item(
                        'Mouse Side 1 (Back)',
                        lambda: self.set_mouse_button('side1'),
                        checked=lambda _: trigger_type == 'mouse' and mouse_button == 'side1'
                    ),
                    item(
                        'Mouse Side 2 (Forward)',
                        lambda: self.set_mouse_button('side2'),
                        checked=lambda _: trigger_type == 'mouse' and mouse_button == 'side2'
                    ),
                    item(
                        'Mouse Middle (Scroll)',
                        lambda: self.set_mouse_button('middle'),
                        checked=lambda _: trigger_type == 'mouse' and mouse_button == 'middle'
                    ),
                    pystray.Menu.SEPARATOR,
                    item(
                        'Keyboard (Ctrl+Alt+V)',
                        self.set_keyboard,
                        checked=lambda _: trigger_type == 'keyboard'
                    )
                )
            ),
            pystray.Menu.SEPARATOR,
            # Recording options
            item(
                'Push-to-Talk Mode',
                self.toggle_push_to_talk,
                checked=lambda _: is_push_to_talk
            ),
            item(
                'Auto-Stop (VAD)',
                self.toggle_vad,
                checked=lambda _: vad_enabled
            ),
            pystray.Menu.SEPARATOR,
            item(
                'Open Config',
                self.open_config
            ),
            item(
                'Open Logs',
                self.open_logs
            ),
            pystray.Menu.SEPARATOR,
            item(
                'Test TTS',
                self.test_tts_from_menu,
                enabled=lambda _: self.service.tts_engine is not None
            ),
            pystray.Menu.SEPARATOR,
            item(
                'Restart Service',
                self.restart_service
            ),
            item(
                'Exit',
                self.exit_app
            )
        )

    def open_config(self):
        """Open config file"""
        import subprocess
        config_file = Path(self.config_path).absolute()
        subprocess.Popen(['notepad', str(config_file)])

    def open_logs(self):
        """Open log file"""
        import subprocess
        log_file = Path('logs/dictator.log').absolute()
        if log_file.exists():
            subprocess.Popen(['notepad', str(log_file)])

    def test_tts_from_menu(self):
        """Test TTS functionality from tray menu"""
        if self.service.tts_engine:
            self.service.logger.info("🧪 Testing TTS from menu...")
            self.service.test_tts()
        else:
            self.service.logger.warning("⚠️ TTS not available")

    def restart_service(self):
        """Restart the service"""
        self.service.logger.info("🔄 Restarting service...")
        self.service.stop()
        self.service = DictatorService(self.config_path)
        threading.Thread(target=self.service.start, daemon=True).start()

    def exit_app(self):
        """Exit application"""
        self.service.logger.info("Exiting...")
        self.service.stop()
        self.icon.stop()
        sys.exit(0)

    def start(self):
        """Start tray icon and service"""
        # Initialize overlay
        overlay_config = self.service.config.get('overlay', {})
        overlay_size = overlay_config.get('size', 15)
        overlay_position = overlay_config.get('position', 'top-right')
        overlay_padding = overlay_config.get('padding', 20)

        self.overlay = init_overlay(
            size=overlay_size,
            position=overlay_position,
            padding=overlay_padding
        )

        # Create icon
        image = self.create_icon_image()
        tooltip = self.service.config['tray']['tooltip']

        self.icon = pystray.Icon(
            name='dictator',
            icon=image,
            title=tooltip,
            menu=self.create_menu()
        )

        # Start service in background thread
        service_thread = threading.Thread(target=self.service.start, daemon=True)
        service_thread.start()

        # Run tray icon (blocks until exit)
        self.icon.run()


def main():
    """Main entry point for tray app"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    tray = DictatorTray(config_path)
    tray.start()


if __name__ == "__main__":
    main()
