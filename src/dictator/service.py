#!/usr/bin/env python3
"""
Dictator Service - Background voice-to-text service
Listens for global hotkey and transcribes speech to text
"""

import os
import sys
import time
import tempfile
import threading
import logging
from pathlib import Path
from typing import Optional

import yaml
from faster_whisper import WhisperModel
import sounddevice as sd
import soundfile as sf
import pyperclip
import pyautogui
import numpy as np
from pynput import keyboard, mouse
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button

try:
    from .tts_engine import KokoroTTSEngine
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class DictatorService:
    """Main service class for Dictator"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the service"""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_logging()

        self.logger.info("🎙️ Dictator Service initializing...")

        # State
        self.is_recording = False
        self.recording_data = []
        self.model: Optional[WhisperModel] = None
        self.tts_engine: Optional[KokoroTTSEngine] = None
        self.hotkey_listener: Optional[keyboard.GlobalHotKeys] = None
        self.mouse_listener: Optional[mouse.Listener] = None
        self.running = False

        # State change callbacks
        self.state_callbacks = []

        # Load Whisper model
        self.load_model()

        # Load TTS engine if enabled
        self.load_tts()

    def register_state_callback(self, callback):
        """Register a callback for state changes: callback(state: str)"""
        self.state_callbacks.append(callback)

    def _emit_state(self, state: str):
        """Emit state change to all callbacks"""
        for callback in self.state_callbacks:
            try:
                callback(state)
            except Exception as e:
                self.logger.error(f"Error in state callback: {e}")

    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)

        if not config_file.exists():
            # Use default config
            return self.get_default_config()

        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_default_config(self) -> dict:
        """Return default configuration"""
        return {
            'whisper': {
                'model': 'medium',
                'language': 'pt',
                'device': 'cuda'
            },
            'hotkey': {
                'type': 'mouse',
                'keyboard_trigger': 'ctrl+alt+v',
                'mouse_button': 'side1',
                'mode': 'toggle',
                'vad_enabled': False,
                'vad_threshold': 0.01,
                'auto_stop_silence': 2.0,
                'max_duration': 60
            },
            'audio': {
                'sample_rate': 16000,
                'channels': 1
            },
            'paste': {
                'delay': 0.5,
                'auto_paste': True
            },
            'service': {
                'auto_start': True,
                'notifications': True,
                'log_level': 'INFO',
                'log_file': ''
            },
            'tray': {
                'enabled': True,
                'tooltip': 'Dictator - Voice to Text (Mouse Side Button)'
            },
            'overlay': {
                'enabled': True,
                'size': 15,
                'position': 'top-right',
                'padding': 20
            }
        }

    def setup_logging(self):
        """Setup logging"""
        log_level = getattr(logging, self.config['service']['log_level'], logging.INFO)

        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        log_file = self.config['service'].get('log_file') or log_dir / 'dictator.log'

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger('DictatorService')

    def load_model(self):
        """Load Whisper model using faster-whisper"""
        model_name = self.config['whisper']['model']
        device = self.config['whisper']['device']

        # Determine compute type based on device
        if device == 'cuda':
            compute_type = "float16"  # Best for GPU
        else:
            compute_type = "int8"  # Best for CPU

        self.logger.info(f"📦 Loading Whisper model '{model_name}' on {device}...")
        self.logger.info(f"   Using faster-whisper with compute_type={compute_type}")

        try:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type
            )
            self.logger.info(f"✅ Model '{model_name}' loaded successfully on {device}!")

            # Try to get GPU info if available
            if device == 'cuda':
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        self.logger.info(f"🎮 Using GPU: {gpu_name} ({vram_total:.1f} GB VRAM)")
                except:
                    self.logger.info(f"🎮 Using CUDA device")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

    def load_tts(self):
        """Load TTS engine if enabled"""
        tts_config = self.config.get('tts', {})
        tts_enabled = tts_config.get('enabled', False)

        if not tts_enabled:
            self.logger.info("🔇 TTS disabled in config")
            return

        if not TTS_AVAILABLE:
            self.logger.warning("⚠️ TTS requested but kokoro-onnx not installed")
            return

        try:
            self.logger.info("🎤 Loading TTS engine...")
            self.tts_engine = KokoroTTSEngine(self.config)

            # Register TTS state callback to emit to service callbacks
            self.tts_engine.register_state_callback(lambda state: self._emit_state(f"tts_{state}"))

            self.logger.info("✅ TTS engine loaded successfully!")
        except Exception as e:
            self.logger.error(f"❌ Failed to load TTS engine: {e}", exc_info=True)
            self.tts_engine = None

    def parse_hotkey(self, hotkey_str: str) -> str:
        """Parse hotkey string to pynput format"""
        # Convert "ctrl+alt+v" to "<ctrl>+<alt>+v"
        parts = hotkey_str.lower().split('+')
        parsed = []

        for part in parts:
            part = part.strip()
            if part in ['ctrl', 'control']:
                parsed.append('<ctrl>')
            elif part in ['alt']:
                parsed.append('<alt>')
            elif part in ['shift']:
                parsed.append('<shift>')
            elif part in ['cmd', 'win', 'super']:
                parsed.append('<cmd>')
            else:
                parsed.append(part)

        return '+'.join(parsed)

    def get_mouse_button(self, button_name: str) -> Button:
        """Map button name to Button enum"""
        button_map = {
            'side1': Button.x1,  # Back button
            'side2': Button.x2,  # Forward button
            'middle': Button.middle  # Scroll wheel click
        }
        return button_map.get(button_name.lower(), Button.x1)

    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse button events"""
        target_button = self.get_mouse_button(self.config['hotkey']['mouse_button'])

        # Only react to the configured button
        if button == target_button:
            mode = self.config['hotkey'].get('mode', 'toggle')

            if mode == 'push_to_talk':
                # Push-to-talk: Record while button is pressed
                if pressed:
                    self.start_recording()
                else:
                    self.stop_recording()
            else:
                # Toggle mode: Click to start/stop
                if pressed:
                    self.toggle_recording()

    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            self.logger.warning("Already recording!")
            return

        vad_enabled = self.config['hotkey'].get('vad_enabled', False)
        vad_mode = " (VAD)" if vad_enabled else ""
        self.logger.info(f"🔴 Recording started{vad_mode}...")

        self.is_recording = True
        self.recording_data = []
        self.last_sound_time = time.time()  # Track last time we heard sound

        # Emit state change
        self._emit_state("recording")

        # Start recording in background thread
        threading.Thread(target=self._record_audio, daemon=True).start()

    def _record_audio(self):
        """Record audio in background"""
        sample_rate = self.config['audio']['sample_rate']
        channels = self.config['audio']['channels']
        vad_enabled = self.config['hotkey'].get('vad_enabled', False)
        silence_duration = self.config['hotkey'].get('auto_stop_silence', 2.0)
        silence_threshold = self.config['hotkey'].get('vad_threshold', 0.01)

        # Track RMS values for debugging and adaptive threshold
        rms_samples = []
        max_rms_seen = [0.0]  # Track maximum RMS seen during recording

        def callback(indata, frames, time_info, status):
            if status:
                self.logger.warning(f"Audio status: {status}")
            if self.is_recording:
                self.recording_data.append(indata.copy())

                # VAD: Check if audio contains sound (not silence)
                if vad_enabled:
                    # Calculate RMS (root mean square) amplitude
                    rms = np.sqrt(np.mean(indata**2))
                    rms_samples.append(rms)

                    # Track maximum RMS
                    if rms > max_rms_seen[0]:
                        max_rms_seen[0] = rms

                    # Adaptive threshold: use either fixed threshold or 20% of max RMS
                    # whichever is lower (more sensitive)
                    adaptive_threshold = min(silence_threshold, max_rms_seen[0] * 0.2)

                    # If amplitude is above adaptive threshold, reset silence timer
                    if rms > adaptive_threshold:
                        self.last_sound_time = time.time()

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                callback=callback
            ):
                start_time = time.time()

                # Record until stopped
                while self.is_recording:
                    time.sleep(0.1)

                    # VAD auto-stop: check if silence duration exceeded
                    if vad_enabled:
                        silence_time = time.time() - self.last_sound_time

                        # Only check after minimum recording time (0.5s)
                        if time.time() - start_time > 0.5 and silence_time >= silence_duration:
                            # Log RMS statistics for calibration
                            if rms_samples:
                                avg_rms = np.mean(rms_samples)
                                max_rms = np.max(rms_samples)
                                self.logger.info(f"🔇 Silence detected ({silence_time:.1f}s)")
                                self.logger.info(f"   RMS stats: avg={avg_rms:.5f}, max={max_rms:.5f}, threshold={silence_threshold:.5f}")
                            self.stop_recording()
                            break

                    # Max duration check
                    max_duration = self.config['hotkey'].get('max_duration', 60)
                    if time.time() - start_time >= max_duration:
                        self.logger.warning(f"⏰ Max duration ({max_duration}s) reached, stopping...")
                        self.stop_recording()
                        break

        except Exception as e:
            self.logger.error(f"❌ Recording error: {e}")
            self.is_recording = False

    def stop_recording(self):
        """Stop recording and transcribe"""
        if not self.is_recording:
            return

        self.logger.info("⏹️ Recording stopped")
        self.is_recording = False

        if not self.recording_data:
            self.logger.warning("No audio recorded!")
            self._emit_state("idle")
            return

        # Emit processing state
        self._emit_state("processing")

        # Process in background thread
        threading.Thread(target=self._process_recording, daemon=True).start()

    def _process_recording(self):
        """Process recorded audio"""
        try:
            # Combine audio chunks
            audio_data = np.concatenate(self.recording_data, axis=0)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(
                    temp_path,
                    audio_data,
                    self.config['audio']['sample_rate']
                )

            self.logger.info("🤖 Transcribing...")

            # Transcribe with faster-whisper
            # Note: VAD is applied during recording, not transcription
            segments, info = self.model.transcribe(
                temp_path,
                language=self.config['whisper']['language']
            )

            # Collect all text from segments
            text = "".join([segment.text for segment in segments]).strip()
            self.logger.info(f"📝 Transcribed: {text}")

            # Clean up temp file
            os.unlink(temp_path)

            # Paste text
            if text:
                self._paste_text(text)

            # Return to idle state
            self._emit_state("idle")

        except Exception as e:
            self.logger.error(f"❌ Processing error: {e}", exc_info=True)
            self._emit_state("idle")

    def _paste_text(self, text: str):
        """Paste text to active window"""
        try:
            # Copy to clipboard
            pyperclip.copy(text)
            self.logger.info("📋 Copied to clipboard")

            # Auto-paste if enabled
            if self.config['paste']['auto_paste']:
                delay = self.config['paste']['delay']
                time.sleep(delay)

                # Paste (Ctrl+V)
                pyautogui.hotkey('ctrl', 'v')
                self.logger.info("✅ Text pasted!")

        except Exception as e:
            self.logger.error(f"❌ Paste error: {e}")

    def speak_text(self, text: str):
        """Speak text using TTS engine"""
        if not self.tts_engine:
            self.logger.warning("⚠️ TTS engine not available")
            return

        try:
            self.logger.info(f"🔊 Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
            self.tts_engine.speak(text, blocking=False)
        except Exception as e:
            self.logger.error(f"❌ TTS error: {e}", exc_info=True)

    def test_tts(self):
        """Test TTS functionality (for debugging)"""
        if not self.tts_engine:
            self.logger.error("❌ TTS engine not initialized!")
            return

        test_text = "Olá! Este é um teste do sistema de síntese de voz."
        self.logger.info(f"🧪 Testing TTS: {test_text}")
        self.speak_text(test_text)

    def start(self):
        """Start the service"""
        self.logger.info("🚀 Dictator Service started")
        self.running = True

        # Setup trigger based on type
        trigger_type = self.config['hotkey'].get('type', 'keyboard')

        if trigger_type == 'mouse':
            # Mouse button trigger
            button_name = self.config['hotkey'].get('mouse_button', 'side1')
            self.logger.info(f"🖱️  Listening for mouse button: {button_name}")

            # Create mouse listener
            self.mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
            self.mouse_listener.start()

        else:
            # Keyboard trigger
            hotkey_str = self.config['hotkey'].get('keyboard_trigger', 'ctrl+alt+v')
            parsed_hotkey = self.parse_hotkey(hotkey_str)

            self.logger.info(f"⌨️  Listening for hotkey: {hotkey_str}")

            # Create hotkey listener
            self.hotkey_listener = keyboard.GlobalHotKeys({
                parsed_hotkey: self.toggle_recording
            })
            self.hotkey_listener.start()

        # Keep running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            self.stop()

    def stop(self):
        """Stop the service"""
        self.logger.info("🛑 Dictator Service stopping...")
        self.running = False

        if self.is_recording:
            self.stop_recording()

        if self.hotkey_listener:
            self.hotkey_listener.stop()

        if self.mouse_listener:
            self.mouse_listener.stop()

        self.logger.info("👋 Dictator Service stopped")


def main():
    """Main entry point"""
    # Get config path from args or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    # Create service
    service = DictatorService(config_path)

    # Start service
    service.start()


if __name__ == "__main__":
    main()
