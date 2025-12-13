#!/usr/bin/env python3
"""
Text-to-Speech Engine for Dictator
Uses kokoro-onnx for high-quality local TTS with GPU acceleration
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
import soundfile as sf
from kokoro_onnx import Kokoro


class TTSState:
    """TTS engine states"""
    IDLE = "idle"
    SPEAKING = "speaking"
    STOPPING = "stopping"


class KokoroTTSEngine:
    """
    High-quality TTS engine using Kokoro-ONNX
    Supports Portuguese with GPU acceleration
    """

    def __init__(self, config: dict):
        """
        Initialize the TTS engine

        Args:
            config: TTS configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('KokoroTTS')

        # Thread synchronization (ADDED - Phase 1.2)
        self._state_lock = threading.Lock()

        # State (protected by _state_lock)
        self.state = TTSState.IDLE
        self.current_audio = None
        self.playback_thread = None
        self.state_callbacks = []

        # Config
        tts_config = config.get('tts', {})
        kokoro_config = tts_config.get('kokoro', {})

        self.language = kokoro_config.get('language', 'pt-br')  # Portuguese Brazil
        self.voice = kokoro_config.get('voice', 'af_sarah')
        self.speed = kokoro_config.get('speed', 1.0)
        self.volume = tts_config.get('volume', 0.8)
        self.sample_rate = 24000  # Kokoro default

        # Model paths
        self.model_path = kokoro_config.get('model_path', 'kokoro-v1.0.onnx')
        self.voices_path = kokoro_config.get('voices_path', 'voices-v1.0.bin')

        # Initialize Kokoro
        self.logger.info(f"[MIC] Loading Kokoro TTS (model={self.model_path})...")
        try:
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            self.logger.info("Kokoro TTS loaded successfully!")
        except Exception as e:
            self.logger.error(f"Failed to load Kokoro: {e}")
            raise

    def register_state_callback(self, callback: Callable[[str], None]):
        """
        Register a callback for state changes

        Args:
            callback: Function to call on state change, receives state string
        """
        self.state_callbacks.append(callback)

    def _emit_state(self, state: str):
        """Emit state change to all callbacks"""
        with self._state_lock:
            self.state = state

        for callback in self.state_callbacks:
            try:
                callback(state)
            except Exception as e:
                self.logger.error(f"Error in TTS state callback: {e}")

    def speak(self, text: str, voice: Optional[str] = None, blocking: bool = False):
        """
        Convert text to speech and play

        Args:
            text: Text to speak
            voice: Voice to use (optional, uses default if not specified)
            blocking: If True, blocks until speech is complete
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided to TTS")
            return

        # Stop any current playback
        if self.is_speaking():
            self.stop()
            time.sleep(0.1)  # Brief pause

        self.logger.info(f" Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")

        # Use specified voice or default
        voice_to_use = voice or self.voice

        try:
            # Generate audio
            self._emit_state(TTSState.SPEAKING)
            start_time = time.time()

            # Generate with kokoro-onnx
            audio_data, sample_rate = self.kokoro.create(
                text,
                voice=voice_to_use,
                speed=self.speed,
                lang=self.language
            )

            generation_time = time.time() - start_time
            self.logger.info(f" Audio generated in {generation_time:.2f}s")

            # Apply volume adjustment
            if self.volume != 1.0:
                audio_data = audio_data * self.volume

            # Store current audio
            self.current_audio = audio_data
            self.sample_rate = sample_rate

            # Play audio
            if blocking:
                self._play_audio_blocking(audio_data, sample_rate)
            else:
                self._play_audio_async(audio_data, sample_rate)

        except Exception as e:
            self.logger.error(f"TTS error: {e}", exc_info=True)
            self._emit_state(TTSState.IDLE)

    def _play_audio_blocking(self, audio_data: np.ndarray, sample_rate: int):
        """Play audio in blocking mode"""
        try:
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()  # Wait until playback is finished
            self._emit_state(TTSState.IDLE)
            self.logger.info("Speech completed")
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            self._emit_state(TTSState.IDLE)

    def _play_audio_async(self, audio_data: np.ndarray, sample_rate: int):
        """Play audio in async mode (non-blocking)"""
        def playback():
            try:
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()
                self._emit_state(TTSState.IDLE)
                self.logger.info("Speech completed")
            except Exception as e:
                self.logger.error(f"Playback error: {e}")
                self._emit_state(TTSState.IDLE)

        self.playback_thread = threading.Thread(target=playback, daemon=True)
        self.playback_thread.start()

    def stop(self):
        """Stop current speech playback"""
        if not self.is_speaking():
            return

        self.logger.info("Stopping TTS playback...")
        self._emit_state(TTSState.STOPPING)

        try:
            # Stop sounddevice playback
            sd.stop()

            # Wait for thread to finish
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=1.0)

            self._emit_state(TTSState.IDLE)
            self.logger.info("TTS stopped")

        except Exception as e:
            self.logger.error(f"Error stopping TTS: {e}")
            self._emit_state(TTSState.IDLE)

    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        with self._state_lock:
            return self.state == TTSState.SPEAKING

    def get_available_voices(self) -> list[str]:
        """Get list of available voices for current language"""
        # Kokoro voices (as of v1.0)
        return [
            'af_sarah',    # American Female - Sarah
            'af_nicole',   # American Female - Nicole
            'af_sky',      # American Female - Sky
            'af',          # American Female
            'am',          # American Male
            'bf',          # British Female
            'bm',          # British Male
        ]

    def save_to_file(self, text: str, output_path: str, voice: Optional[str] = None):
        """
        Generate speech and save to file instead of playing

        Args:
            text: Text to convert
            output_path: Path to save audio file
            voice: Voice to use (optional)
        """
        voice_to_use = voice or self.voice

        try:
            self.logger.info(f" Saving TTS to file: {output_path}")

            # Generate audio
            audio_data, sample_rate = self.kokoro.create(
                text,
                voice=voice_to_use,
                speed=self.speed,
                lang=self.language
            )

            # Apply volume
            if self.volume != 1.0:
                audio_data = audio_data * self.volume

            # Save to file
            sf.write(output_path, audio_data, sample_rate)
            self.logger.info(f"Audio saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving audio: {e}", exc_info=True)


def main():
    """Test the TTS engine standalone"""
    import yaml

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Default config for testing
        config = {
            'tts': {
                'enabled': True,
                'engine': 'kokoro-onnx',
                'kokoro': {
                    'language': 'pt-br',  # Portuguese Brazil
                    'voice': 'af_sarah',
                    'speed': 1.0,
                    'model_path': 'kokoro-v1.0.onnx',
                    'voices_path': 'voices-v1.0.bin'
                },
                'volume': 0.8
            }
        }

    # Create engine
    print("Creating TTS engine...")
    engine = KokoroTTSEngine(config)

    # Test Portuguese
    print("\nTesting Portuguese TTS...")
    test_text = "Olá! Este é um teste do sistema de síntese de voz em português."
    engine.speak(test_text, blocking=True)

    print("\nTesting English TTS...")
    # Create new config for English
    config_en = {
        'tts': {
            'enabled': True,
            'engine': 'kokoro-onnx',
            'kokoro': {
                'language': 'en-us',  # American English
                'voice': 'af_sarah',
                'speed': 1.0,
                'model_path': 'kokoro-v1.0.onnx',
                'voices_path': 'voices-v1.0.bin'
            },
            'volume': 0.8
        }
    }
    engine_en = KokoroTTSEngine(config_en)
    test_text_en = "Hello! This is a test of the text to speech system."
    engine_en.speak(test_text_en, blocking=True)

    print("\nTTS engine test completed!")


if __name__ == "__main__":
    main()
