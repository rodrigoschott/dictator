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
import requests

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
            # Create a clean copy of config with only serializable values
            clean_config = {}
            for key, value in self.service.config.items():
                # Skip non-serializable objects
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    clean_config[key] = value
                elif hasattr(value, '__dict__'):
                    self.service.logger.warning(f"⚠️ Skipping non-serializable config key: {key}")
                    
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(clean_config, f, default_flow_style=False, allow_unicode=True)
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

        if state in overlay_state_map:
            set_overlay_state(overlay_state_map[state])

    def get_ollama_models(self):
        """Fetch available models from Ollama API"""
        try:
            base_url = self.service.config['voice']['llm']['ollama']['base_url']
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            if not models:
                # Fallback to defaults if API returns empty
                return ['llama3.2:latest', 'llama3.1:latest', 'mistral:latest', 'codellama:latest']
            
            return sorted(models)
        except Exception as e:
            print(f"⚠️ Failed to fetch Ollama models: {e}")
            # Fallback to default models
            return ['llama3.2:latest', 'llama3.1:latest', 'mistral:latest', 'codellama:latest']

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

    def toggle_vad(self):
        """Toggle VAD (Voice Activity Detection)"""
        current = self.service.config['hotkey'].get('vad_enabled', False)
        self.service.config['hotkey']['vad_enabled'] = not current
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu

        status = 'Enabled' if not current else 'Disabled'
        self.service.logger.info(f"VAD: {status}")

        # If LLM Mode is active, restart to apply VAD changes
        if self.service.use_event_driven_mode:
            self.service.logger.info("⚠️ Restarting service to apply VAD changes in LLM Mode")
            self.restart_service()

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
        self.service.logger.info(f"🤖 LLM Mode: {status}")
        # Restart service to apply changes
        self.service.logger.info("🔄 Restarting service to apply LLM Mode changes...")
        self.restart_service()

    def set_llm_provider(self, provider: str):
        """Set LLM provider"""
        self.service.config.setdefault('voice', {}).setdefault('llm', {})['provider'] = provider
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu
        self.service.logger.info(f"🔄 LLM Provider changed to: {provider}")
        # Restart service to apply changes
        if self.service.use_event_driven_mode:
            self.service.logger.info("🔄 Restarting service to apply provider changes...")
            self.restart_service()

    def set_ollama_model(self, model: str):
        """Set Ollama model"""
        self.service.config.setdefault('voice', {}).setdefault('llm', {}).setdefault('ollama', {})['model'] = model
        self.save_config()
        self.icon.menu = self.create_menu()  # Refresh menu
        self.service.logger.info(f"🦙 Ollama model changed to: {model}")
        # Restart service to apply changes
        if self.service.use_event_driven_mode:
            self.service.logger.info("🔄 Restarting service to apply model changes...")
            self.restart_service()

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

        # Voice/Claude settings
        voice_config = self.service.config.get('voice', {})
        claude_mode = voice_config.get('claude_mode', False)

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
                        lambda icon, item: self.set_mouse_button('side1'),
                        checked=lambda _: trigger_type == 'mouse' and mouse_button == 'side1'
                    ),
                    item(
                        'Mouse Side 2 (Forward)',
                        lambda icon, item: self.set_mouse_button('side2'),
                        checked=lambda _: trigger_type == 'mouse' and mouse_button == 'side2'
                    ),
                    item(
                        'Mouse Middle (Scroll)',
                        lambda icon, item: self.set_mouse_button('middle'),
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
            # LLM integration
            item(
                'LLM Mode (Voice Assistant)',
                self.toggle_claude_mode,
                checked=lambda _: claude_mode
            ),
            pystray.Menu.SEPARATOR,
            # LLM Provider selection
            item(
                'LLM Provider',
                pystray.Menu(
                    item(
                        'Claude CLI (Local)',
                        lambda icon, item: self.set_llm_provider('claude-cli'),
                        checked=lambda _: voice_config.get('llm', {}).get('provider', 'claude-cli') == 'claude-cli'
                    ),
                    item(
                        'Claude Direct (API)',
                        lambda icon, item: self.set_llm_provider('claude-direct'),
                        checked=lambda _: voice_config.get('llm', {}).get('provider', 'claude-cli') == 'claude-direct'
                    ),
                    item(
                        'Ollama (Local)',
                        lambda icon, item: self.set_llm_provider('ollama'),
                        checked=lambda _: voice_config.get('llm', {}).get('provider', 'claude-cli') == 'ollama'
                    ),
                    item(
                        'N8N Tool-Calling',
                        lambda icon, item: self.set_llm_provider('n8n_toolcalling'),
                        checked=lambda _: voice_config.get('llm', {}).get('provider', 'claude-cli') == 'n8n_toolcalling'
                    )
                )
            ),
            # Ollama model selection (only shown when Ollama is active) - Dynamic discovery
            item(
                'Ollama Model',
                pystray.Menu(
                    lambda: self._create_ollama_model_items(voice_config)
                ),
                visible=lambda _: voice_config.get('llm', {}).get('provider', 'claude-cli') == 'ollama'
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
