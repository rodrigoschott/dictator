#!/usr/bin/env python3
"""
Dictator System Tray Icon
Provides visual control for the Dictator service
"""

import sys
import threading
import logging
from pathlib import Path

import yaml
import pystray
from PIL import Image, ImageDraw
from pystray import MenuItem as item
import requests

from dictator.service import DictatorService
from dictator.overlay import OverlayState, init_overlay, set_state as set_overlay_state

logger = logging.getLogger(__name__)


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
        self.capture_mode = False
        self.mouse_capture_listener = None
        self.keyboard_capture_listener = None
        self.captured_keys = set()
        self.keys_currently_pressed = set()

        # Register state callback
        self.service.register_state_callback(self.on_state_change)

    def save_config(self):
        """Save current config to file"""
        try:
            # Create a clean copy of config with only serializable values
            clean_config = {}
            for key, value in self.service.config.items():
                # Skip non-serializable objects
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    clean_config[key] = value
                elif hasattr(value, '__dict__'):
                    self.service.logger.warning(f"Skipping non-serializable config key: {key}")
                    
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(clean_config, f, default_flow_style=False, allow_unicode=True)
            self.service.logger.info("Configuration saved")
        except Exception as e:
            self.service.logger.error(f"Failed to save config: {e}")

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

        # Update tooltip with health status (NEW)
        self.update_tooltip()

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

    def update_tooltip(self):
        """Update tray tooltip with health status (NEW)"""
        if not self.icon:
            return

        try:
            base_text = self.service.config.get('tray', {}).get('tooltip', 'Dictator')

            # Get health status if available
            if hasattr(self.service, 'health_report') and self.service.health_report:
                tooltip = self.service.health_report.get_tray_tooltip(base_text)
            else:
                tooltip = base_text

            self.icon.title = tooltip

        except Exception as e:
            self.service.logger.warning(f"Failed to update tooltip: {e}")

    def get_ollama_models(self):
        """Fetch available models from Ollama API"""
        try:
            base_url = self.service.config['voice']['llm']['ollama']['base_url']
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            response.raise_for_status()

            data = response.json()
            models = [model['name'] for model in data.get('models', [])]

            return sorted(models) if models else []
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            # Return empty list if Ollama is not reachable
            return []

    def get_whisper_models(self):
        """Get available Whisper models"""
        return [
            'tiny',
            'base',
            'small',
            'medium',
            'large-v3'
        ]

    def set_whisper_model(self, model: str):
        """Set Whisper model and reload"""
        self.service.logger.info(f"Changing Whisper model to: {model}")

        # Update config
        self.service.config['whisper']['model'] = model
        self.save_config()

        # Refresh menu
        self.icon.menu = self.create_menu()

        # Reload Whisper model
        self.service.logger.info("Reloading Whisper model...")
        if self.service.update_whisper_model():
            self.service.logger.info(f"✓ Whisper model changed to: {model}")
        else:
            self.service.logger.error(f"✗ Failed to change Whisper model to: {model}")

    def _create_ollama_model_items(self, voice_config):
        """Create menu items for Ollama models"""
        from pystray import MenuItem as item

        def make_callback(model_name):
            """Create callback for specific model"""
            def callback(icon, item_obj):
                self.set_ollama_model(model_name)
            return callback

        current_model = voice_config.get('llm', {}).get('ollama', {}).get('model', 'llama3.2:latest')

        return tuple(
            item(
                model_name,
                make_callback(model_name),
                checked=lambda _, m=model_name: m == current_model
            )
            for model_name in self.get_ollama_models()
        )

    def _create_whisper_model_items(self):
        """Create menu items for Whisper models"""
        from pystray import MenuItem as item

        def make_callback(model_name):
            """Create callback for specific model"""
            def callback(icon, item_obj):
                self.set_whisper_model(model_name)
            return callback

        current_model = self.service.config['whisper'].get('model', 'large-v3')

        # Model descriptions
        model_info = {
            'tiny': 'Tiny (~75 MB, fastest)',
            'base': 'Base (~145 MB)',
            'small': 'Small (~466 MB)',
            'medium': 'Medium (~1.5 GB)',
            'large-v3': 'Large-v3 (~3 GB, best)'
        }

        return tuple(
            item(
                model_info.get(model_name, model_name),
                make_callback(model_name),
                checked=lambda _, m=model_name: m == current_model
            )
            for model_name in self.get_whisper_models()
        )

    def toggle_vad(self):
        """Toggle VAD (Voice Activity Detection)"""
        current = self.service.config['hotkey'].get('vad_enabled', False)
        self.service.config['hotkey']['vad_enabled'] = not current
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu

        status = 'Enabled' if not current else 'Disabled'
        self.service.logger.info(f"VAD: {status}")
        # VAD config is read dynamically - no restart needed

    def toggle_push_to_talk(self):
        """Toggle between push-to-talk and toggle mode"""
        current = self.service.config['hotkey'].get('mode', 'toggle')
        new_mode = 'push_to_talk' if current == 'toggle' else 'toggle'
        self.service.config['hotkey']['mode'] = new_mode
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu
        self.service.logger.info(f"Mode: {new_mode}")

    def toggle_claude_mode(self):
        """Toggle LLM Mode (Voice Assistant)"""
        voice_config = self.service.config.get('voice', {})
        current = voice_config.get('claude_mode', False)
        self.service.config.setdefault('voice', {})['claude_mode'] = not current
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu
        status = 'Enabled' if not current else 'Disabled'
        self.service.logger.info(f"LLM Mode: {status}")
        # Restart service to apply changes
        self.service.logger.info("Restarting service to apply LLM Mode changes...")
        self.restart_service()

    def set_llm_provider(self, provider: str):
        """Set LLM provider"""
        self.service.config.setdefault('voice', {}).setdefault('llm', {})['provider'] = provider
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu
        self.service.logger.info(f"LLM Provider changed to: {provider}")
        # Update LLM caller dynamically without restart
        if self.service.use_event_driven_mode:
            self.service.update_llm_caller()

    def set_ollama_model(self, model: str):
        """Set Ollama model"""
        self.service.config.setdefault('voice', {}).setdefault('llm', {}).setdefault('ollama', {})['model'] = model
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu
        self.service.logger.info(f"Ollama model changed to: {model}")
        # Update LLM caller dynamically without restart
        if self.service.use_event_driven_mode:
            self.service.update_llm_caller()

    def set_mouse_button(self, button):
        """Set mouse button trigger"""
        self.service.config['hotkey']['type'] = 'mouse'
        self.service.config['hotkey']['mouse_button'] = button
        self.save_config()
        # Update listener dynamically without restart
        self.service.update_hotkey_listener()

    def set_keyboard(self):
        """Set keyboard trigger"""
        self.service.config['hotkey']['type'] = 'keyboard'
        self.save_config()
        # Update listener dynamically without restart
        self.service.update_hotkey_listener()

    def start_capture_custom_hotkey(self):
        """Start capturing custom hotkey (mouse or keyboard)"""
        if self.capture_mode:
            return  # Already in capture mode

        self.capture_mode = True
        self.captured_keys = set()  # Track pressed keys for combinations
        self.keys_currently_pressed = set()  # Track which keys are currently held down
        self.service.logger.info("=== CAPTURE MODE ACTIVATED ===")
        self.service.logger.info("Press any key combination or click any mouse button")
        self.service.logger.info("Timeout: 15 seconds")

        # Update tooltip to show capture is active
        if self.icon:
            self.icon.title = "CAPTURING - Press hotkey or click button (15s)..."
            # Update icon color to indicate capture mode
            self.icon.icon = self.create_icon_image("#00FFFF")  # Cyan for capture mode

        # Auto-cancel capture after 15 seconds
        def cancel_capture_timeout():
            import time
            time.sleep(15)
            if self.capture_mode:
                self.service.logger.warning("Capture timeout - no input received")
                self._finish_capture()
                self.update_tooltip()

        timeout_thread = threading.Thread(target=cancel_capture_timeout, daemon=True)
        timeout_thread.start()

        # Start temporary capture listeners (both mouse and keyboard)
        from pynput import mouse, keyboard

        def on_click_capture(x, y, button, pressed):
            """Capture mouse button and exit capture mode"""
            if not pressed:  # Only capture on button press
                return

            self._finish_capture()

            # Map button to string representation
            button_name = self._button_to_string(button)

            self.service.logger.info(f"Captured mouse button: {button_name}")

            # Update config for mouse
            self.service.config['hotkey']['type'] = 'mouse'
            self.service.config['hotkey']['mouse_button'] = button_name
            self.save_config()

            # Update listener and UI
            self._apply_captured_hotkey()

            return False  # Stop listener

        def on_press_capture(key):
            """Capture keyboard key press - track all pressed keys"""
            try:
                # Add key to captured set
                if hasattr(key, 'char') and key.char:
                    key_str = key.char
                else:
                    key_str = str(key).replace('Key.', '')

                self.service.logger.info(f"[CAPTURE] Key pressed: {key_str}")
                self.captured_keys.add(key_str)
                self.keys_currently_pressed.add(key_str)

                # Update tooltip to show current combo
                combo_display = '+'.join(sorted(self.keys_currently_pressed))
                if self.icon:
                    self.icon.title = f"CAPTURING: {combo_display}"

                self.service.logger.info(f"[CAPTURE] Current combo: {combo_display}")

            except AttributeError as e:
                self.service.logger.error(f"[CAPTURE] Error capturing key: {e}")

        def on_release_capture(key):
            """Capture keyboard key combination when all keys released"""
            try:
                # Remove from currently pressed
                if hasattr(key, 'char') and key.char:
                    key_str = key.char
                else:
                    key_str = str(key).replace('Key.', '')

                self.service.logger.info(f"[CAPTURE] Key released: {key_str}")

                if key_str in self.keys_currently_pressed:
                    self.keys_currently_pressed.remove(key_str)

                self.service.logger.info(f"[CAPTURE] Keys still pressed: {self.keys_currently_pressed}")
                self.service.logger.info(f"[CAPTURE] All captured keys: {self.captured_keys}")

                # Only finalize when all keys are released AND we captured something
                if not self.keys_currently_pressed and self.captured_keys:
                    self.service.logger.info("[CAPTURE] All keys released, finalizing...")
                    self._finish_capture()

                    # Build hotkey string from captured keys
                    modifiers = []
                    regular_keys = []

                    for k in self.captured_keys:
                        k_lower = str(k).lower()
                        if 'ctrl' in k_lower or 'control' in k_lower or 'ctrl_l' in k_lower or 'ctrl_r' in k_lower:
                            modifiers.append('ctrl')
                        elif 'alt' in k_lower or 'alt_l' in k_lower or 'alt_r' in k_lower:
                            modifiers.append('alt')
                        elif 'shift' in k_lower or 'shift_l' in k_lower or 'shift_r' in k_lower:
                            modifiers.append('shift')
                        elif 'cmd' in k_lower or 'super' in k_lower:
                            modifiers.append('cmd')
                        else:
                            regular_keys.append(k)

                    # Build hotkey string (e.g., "ctrl+alt+v")
                    hotkey_parts = sorted(set(modifiers)) + sorted(regular_keys)
                    hotkey_string = '+'.join(hotkey_parts)

                    if not hotkey_string:
                        self.service.logger.warning("No valid key captured")
                        self.update_tooltip()
                        return False

                    self.service.logger.info(f"Captured keyboard hotkey: {hotkey_string}")

                    # Update config for keyboard
                    self.service.config['hotkey']['type'] = 'keyboard'
                    self.service.config['hotkey']['keyboard_trigger'] = hotkey_string
                    self.save_config()

                    # Update listener and UI
                    self._apply_captured_hotkey()

                    return False  # Stop listener

            except AttributeError:
                pass

        # Start both listeners
        self.mouse_capture_listener = mouse.Listener(on_click=on_click_capture)
        self.keyboard_capture_listener = keyboard.Listener(
            on_press=on_press_capture,
            on_release=on_release_capture
        )

        self.mouse_capture_listener.start()
        self.keyboard_capture_listener.start()

    def _finish_capture(self):
        """Stop all capture listeners and exit capture mode"""
        if hasattr(self, 'mouse_capture_listener') and self.mouse_capture_listener:
            self.mouse_capture_listener.stop()
            self.mouse_capture_listener = None

        if hasattr(self, 'keyboard_capture_listener') and self.keyboard_capture_listener:
            self.keyboard_capture_listener.stop()
            self.keyboard_capture_listener = None

        self.capture_mode = False
        self.captured_keys = set()

        # Restore icon to normal color based on current state
        if self.icon:
            color_map = {
                "idle": "white",
                "recording": "#FF4444",
                "processing": "#FFAA00",
                "tts_speaking": "#44FF44",
                "tts_idle": "white",
                "tts_stopping": "white"
            }
            normal_color = color_map.get(self.current_state, "white")
            self.icon.icon = self.create_icon_image(normal_color)

    def _apply_captured_hotkey(self):
        """Apply captured hotkey and update UI"""
        # Update listener dynamically without restart
        self.service.update_hotkey_listener()

        # Refresh menu to show new selection
        self.icon.menu = self.create_menu()

        # Restore normal tooltip
        self.update_tooltip()

    def _button_to_string(self, button) -> str:
        """Convert pynput button to string representation"""
        from pynput import mouse

        button_map = {
            mouse.Button.left: 'left',
            mouse.Button.right: 'right',
            mouse.Button.middle: 'middle',
        }

        # Handle Button.x1 and Button.x2 (side buttons)
        button_str = str(button)
        if 'x1' in button_str.lower() or button == mouse.Button.x1:
            return 'side1'
        elif 'x2' in button_str.lower() or button == mouse.Button.x2:
            return 'side2'

        # Return mapped button or default to string representation
        return button_map.get(button, str(button).replace('Button.', ''))

    def _create_trigger_menu(self):
        """Create trigger selection submenu with custom hotkey display"""
        from pystray import MenuItem as item

        trigger_type = self.service.config['hotkey'].get('type', 'keyboard')
        mouse_button = self.service.config['hotkey'].get('mouse_button', 'middle')
        keyboard_trigger = self.service.config['hotkey'].get('keyboard_trigger', 'ctrl+alt+v')

        # Build menu items list
        menu_items = []

        # Default mouse middle option
        is_middle_active = trigger_type == 'mouse' and mouse_button == 'middle'
        menu_items.append(
            item(
                'Mouse Middle (Default)',
                lambda icon, item: self.set_mouse_button('middle'),
                checked=lambda _: self.service.config['hotkey'].get('type') == 'mouse' and self.service.config['hotkey'].get('mouse_button') == 'middle'
            )
        )

        # Default keyboard option
        is_keyboard_default_active = trigger_type == 'keyboard' and keyboard_trigger == 'ctrl+alt+v'
        menu_items.append(
            item(
                'Keyboard (Ctrl+Alt+V)',
                self.set_keyboard,
                checked=lambda _: self.service.config['hotkey'].get('type') == 'keyboard' and self.service.config['hotkey'].get('keyboard_trigger', 'ctrl+alt+v') == 'ctrl+alt+v'
            )
        )

        # Show current custom option ONLY if it's active
        is_custom_mouse = trigger_type == 'mouse' and mouse_button != 'middle'
        is_custom_keyboard = trigger_type == 'keyboard' and keyboard_trigger != 'ctrl+alt+v'

        if is_custom_mouse or is_custom_keyboard:
            menu_items.append(pystray.Menu.SEPARATOR)

            if is_custom_mouse:
                custom_label = f"Custom: Mouse {mouse_button.title()}"
                # Create closure that captures the current button value
                def make_mouse_callback(button_name):
                    return lambda icon, item: self.set_mouse_button(button_name)

                menu_items.append(
                    item(
                        custom_label,
                        make_mouse_callback(mouse_button),
                        checked=lambda _: True  # Always checked because it's the active option
                    )
                )
            else:  # is_custom_keyboard
                custom_label = f"Custom: {keyboard_trigger}"
                menu_items.append(
                    item(
                        custom_label,
                        lambda icon, item: None,
                        checked=lambda _: True,  # Always checked because it's the active option
                        enabled=False  # Can't be changed directly, use capture
                    )
                )

        # Separator and capture option
        menu_items.append(pystray.Menu.SEPARATOR)
        menu_items.append(
            item(
                'Capture Custom Hotkey...',
                lambda icon, item: self.start_capture_custom_hotkey()
            )
        )

        return pystray.Menu(*menu_items)

    def create_menu(self):
        """Create tray menu"""
        # Get current settings for display only (not for lambdas - they read config directly)
        trigger_type = self.service.config['hotkey'].get('type', 'keyboard')
        mouse_button = self.service.config['hotkey'].get('mouse_button', 'side1')

        # Trigger info text with friendly names
        if trigger_type == 'mouse':
            button_display = {
                'middle': 'Middle',
                'side1': 'Side 1 (Back)',
                'side2': 'Side 2 (Forward)',
                'left': 'Left',
                'right': 'Right'
            }.get(mouse_button, mouse_button.title())
            trigger_text = f"Trigger: Mouse {button_display}"
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
                self._create_trigger_menu()
            ),
            pystray.Menu.SEPARATOR,
            # Recording options
            item(
                'Push-to-Talk Mode',
                self.toggle_push_to_talk,
                checked=lambda _: self.service.config['hotkey'].get('mode', 'toggle') == 'push_to_talk'
            ),
            item(
                'Auto-Stop (VAD)',
                self.toggle_vad,
                checked=lambda _: self.service.config['hotkey'].get('vad_enabled', False)
            ),
            pystray.Menu.SEPARATOR,
            # Whisper model selection
            item(
                'Whisper Model',
                pystray.Menu(
                    *self._create_whisper_model_items()
                )
            ),
            pystray.Menu.SEPARATOR,
            # LLM integration
            item(
                'LLM Mode (Voice Assistant)',
                self.toggle_claude_mode,
                checked=lambda _: self.service.config.get('voice', {}).get('claude_mode', False)
            ),
            pystray.Menu.SEPARATOR,
            # LLM Provider selection
            item(
                'LLM Provider',
                pystray.Menu(
                    item(
                        'Claude CLI (Local)',
                        lambda icon, item: self.set_llm_provider('claude-cli'),
                        checked=lambda _: self.service.config.get('voice', {}).get('llm', {}).get('provider', 'claude-cli') == 'claude-cli'
                    ),
                    item(
                        'Claude Direct (API)',
                        lambda icon, item: self.set_llm_provider('claude-direct'),
                        checked=lambda _: self.service.config.get('voice', {}).get('llm', {}).get('provider', 'claude-cli') == 'claude-direct'
                    ),
                    item(
                        'Ollama (Local)',
                        lambda icon, item: self.set_llm_provider('ollama'),
                        checked=lambda _: self.service.config.get('voice', {}).get('llm', {}).get('provider', 'claude-cli') == 'ollama'
                    ),
                    item(
                        'N8N Tool-Calling',
                        lambda icon, item: self.set_llm_provider('n8n_toolcalling'),
                        checked=lambda _: self.service.config.get('voice', {}).get('llm', {}).get('provider', 'claude-cli') == 'n8n_toolcalling'
                    )
                )
            ),
            # Ollama model selection (only shown when Ollama is active) - Static list
            item(
                'Ollama Model',
                pystray.Menu(
                    *self._create_ollama_model_items(self.service.config.get('voice', {}))
                ),
                visible=lambda _: self.service.config.get('voice', {}).get('llm', {}).get('provider', 'claude-cli') == 'ollama'
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
        run_dir = getattr(self.service, 'run_dir', None)
        log_name = self.service.config.get('service', {}).get('log_file') or 'dictator.log'
        log_path = None

        if run_dir:
            candidate = Path(run_dir) / Path(log_name).name
            if candidate.exists():
                log_path = candidate

        if log_path is None:
            legacy_candidate = Path('logs') / Path(log_name).name
            if legacy_candidate.exists():
                log_path = legacy_candidate

        if log_path is None:
            return

        subprocess.Popen(['notepad', str(log_path.absolute())])

    def test_tts_from_menu(self):
        """Test TTS functionality from tray menu"""
        if self.service.tts_engine:
            self.service.logger.info("Testing TTS from menu...")
            self.service.test_tts()
        else:
            self.service.logger.warning("TTS not available")

    def restart_service(self):
        """Restart the service"""
        self.service.logger.info("Restarting service...")
        self.service.stop()
        self.service = DictatorService(self.config_path)
        threading.Thread(target=self.service.start, daemon=True).start()

    def exit_app(self):
        """Exit application"""
        self.service.logger.info("Exiting...")

        # 0. Stop capture listeners if active
        self._finish_capture()

        # 1. Stop service (cleanup listeners, threads)
        self.service.stop()

        # 2. Stop overlay
        if self.overlay:
            from dictator.overlay import stop_overlay
            stop_overlay()

        # 3. Stop tray icon - causes icon.run() to return naturally
        # Do NOT call sys.exit() - let Python exit when main thread completes
        self.icon.stop()

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
    import tempfile
    import os
    from pathlib import Path

    # Global lock file handle (must keep reference to maintain lock)
    global _lock_file_handle
    _lock_file_handle = None

    try:
        # Single instance check using lock file
        lock_file_path = Path(tempfile.gettempdir()) / "dictator_instance.lock"

        try:
            # Try to open file in exclusive mode
            # This will fail if another instance already has it open
            import msvcrt

            # Open or create the lock file
            _lock_file_handle = open(lock_file_path, 'w')

            # Try to lock it exclusively (non-blocking)
            try:
                msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
            except IOError:
                # Lock failed - another instance is running
                _lock_file_handle.close()

                import tkinter as tk
                from tkinter import messagebox

                root = tk.Tk()
                root.withdraw()

                messagebox.showerror(
                    "Dictator Already Running",
                    "Dictator is already running!\n\n"
                    "Only one instance can run at a time.\n"
                    "Check your system tray for the existing instance."
                )

                root.destroy()
                sys.exit(1)

            # Lock acquired - write PID to file
            _lock_file_handle.write(str(os.getpid()))
            _lock_file_handle.flush()

        except Exception as e:
            # Failed to create/lock file
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()

            messagebox.showerror(
                "Dictator Error",
                f"Failed to acquire instance lock:\n{str(e)}"
            )

            root.destroy()
            sys.exit(1)

        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        tray = DictatorTray(config_path)
        tray.start()

    except Exception as e:
        # Catch all exceptions and show error dialog
        import traceback
        import tkinter as tk
        from tkinter import messagebox

        error_msg = f"Dictator failed to start:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"

        # Try to log error
        try:
            logger.error(f"Fatal error in main: {error_msg}")
        except:
            pass

        # Show error dialog
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Dictator Error", error_msg)
            root.destroy()
        except:
            # If GUI fails, print to console
            print(error_msg)

        sys.exit(1)


if __name__ == "__main__":
    main()
