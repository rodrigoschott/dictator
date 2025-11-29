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
from enum import Enum
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

from . import logging_setup
from .monitoring.thread_monitor import ThreadMonitor
from .audio_processor import AudioProcessor, VoiceSessionAudioBridge  # ADDED - Phase 5.1
try:
    from .tts_engine import KokoroTTSEngine
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from .voice import (
        VoiceSessionManager,
        VADConfig,
        LLMCaller,
        DirectLLMCaller,
        OllamaLLMCaller,
        N8NToolCallingLLMCaller
    )
    VOICE_SESSION_AVAILABLE = True
except ImportError:
    VOICE_SESSION_AVAILABLE = False


class ServiceState(str, Enum):
    """
    Service states for voice interaction

    State transitions:
    - IDLE → RECORDING (user presses hotkey)
    - RECORDING → PROCESSING (user stops or VAD detects silence)
    - PROCESSING → SPEAKING (LLM response ready)
    - PROCESSING → IDLE (dictation mode, no TTS)
    - SPEAKING → IDLE (TTS finished)
    - ANY → INTERRUPTED (user presses hotkey during processing/speaking)
    - INTERRUPTED → RECORDING (new recording starts)
    """
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"  # Transcription + LLM
    SPEAKING = "speaking"      # TTS playback
    INTERRUPTED = "interrupted"  # User interrupted during processing/speaking


class DictatorService:
    """Main service class for Dictator"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the service"""
        self.config_path = config_path
        self.config = self.load_config()
        self.ensure_config_defaults()
        self.thread_monitor: Optional[ThreadMonitor] = None
        self.logging_state: Optional[logging_setup.LoggingState] = None
        self.run_dir: Optional[Path] = None
        self.setup_logging()

        self.logger.info("🎙️ Dictator Service initializing...")

        # Thread synchronization (ADDED - Phase 1.1)
        self._state_lock = threading.RLock()  # Recursive lock for state
        self._recording_data_lock = threading.Lock()  # Lock for recording_data list

        # State (protected by _state_lock)
        self._state = ServiceState.IDLE  # CHANGED - Phase 2: State machine
        self.is_recording = False  # DEPRECATED - kept for backward compat, will be removed
        self.recording_data = []
        self._session_counter = 0  # ADDED - Phase 4.3: Track recording sessions to discard stale TTS
        self._current_tts_session = 0  # ADDED - Phase 4.3: Session number for current/pending TTS
        self.model: Optional[WhisperModel] = None
        self.tts_engine: Optional[KokoroTTSEngine] = None
        self.hotkey_listener: Optional[keyboard.GlobalHotKeys] = None
        self.mouse_listener: Optional[mouse.Listener] = None
        self.running = False

        # Event-driven voice session (NEW - replaces old polling system)
        self.voice_session: Optional[VoiceSessionManager] = None
        self.voice_session_thread: Optional[threading.Thread] = None
        self.voice_session_loop: Optional['asyncio.AbstractEventLoop'] = None
        self.use_event_driven_mode = False
        self.audio_chunk_timestamp_ms = 0  # Track timestamp for audio chunks (DEPRECATED - Phase 5.1)

        # Audio processing (ADDED - Phase 5.1)
        self.audio_processor: Optional[AudioProcessor] = None
        self.voice_session_bridge: Optional[VoiceSessionAudioBridge] = None

        # State change callbacks
        self.state_callbacks = []

        # Health check and dependency validation (NEW)
        from .health_check import DependencyValidator
        validator = DependencyValidator(self.config)
        self.health_report = validator.run_full_check(quick=True)

        # Auto-disable features based on health (NEW)
        self._apply_health_degradation()

        # Write status file (NEW)
        self._write_status_file()

        # Load Whisper model
        self.load_model()

        # Load TTS engine if enabled AND available (MODIFIED)
        if self._tts_available:
            self.load_tts()

        # Load event-driven voice session if enabled AND available (MODIFIED)
        if self._voice_available:
            self.load_voice_session()

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

    def _transition_state(self, from_states: list[ServiceState], to_state: ServiceState) -> bool:
        """
        Attempt state transition with validation (ADDED - Phase 2.3)

        Args:
            from_states: Allowed current states
            to_state: Target state

        Returns:
            True if transition succeeded, False if invalid
        """
        with self._state_lock:
            if self._state not in from_states:
                self.logger.warning(
                    f"Invalid state transition: {self._state} -> {to_state} "
                    f"(expected one of {from_states})"
                )
                return False

            old_state = self._state
            self._state = to_state

            # Update deprecated flag for backward compat
            self.is_recording = (to_state == ServiceState.RECORDING)

            self.logger.debug(f"State transition: {old_state} -> {to_state}")

        # Emit state change to UI (outside lock)
        self._emit_state(to_state.value)
        return True

    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)

        if not config_file.exists():
            # Use default config
            return self.get_default_config()

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # Return default if file is empty or invalid
            if not config:
                return self.get_default_config()
            return config

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
            'logging': {
                'level': 'INFO',
                'structured': False,
                'run_retention': 5,
                'trace_main_loop': False,
                'trace_threads': False,
            },
            'profiling': {
                'freeze_timeout': 10,
                'thread_monitor_interval': 30,
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
            },
            'tts': {
                'enabled': True,
                'model': 'kokoro-v1.0.onnx',
                'voice': 'af_sarah'
            },
            'voice': {
                'mode': 'event_driven',
                'claude_mode': False,
                'llm': {
                    'provider': 'ollama',
                    'ollama': {
                        'base_url': 'http://localhost:11434',
                        'model': 'llama3.2:latest'
                    },
                    'claude_direct': {
                        'api_key': '',
                        'model': 'claude-3-5-sonnet-20241022'
                    }
                }
            }
        }

    def ensure_config_defaults(self) -> None:
        """Ensure new logging/profiling defaults exist without overwriting user values."""

        logging_defaults = {
            'level': self.config.get('service', {}).get('log_level', 'INFO'),
            'structured': False,
            'run_retention': 5,
            'trace_main_loop': False,
            'trace_threads': False,
        }
        logging_cfg = self.config.setdefault('logging', {})
        for key, value in logging_defaults.items():
            logging_cfg.setdefault(key, value)

        profiling_defaults = {
            'freeze_timeout': 10,
            'thread_monitor_interval': 30,
        }
        profiling_cfg = self.config.setdefault('profiling', {})
        for key, value in profiling_defaults.items():
            profiling_cfg.setdefault(key, value)

    def setup_logging(self) -> None:
        """Bootstrap logging configuration and optional monitors."""

        if self.thread_monitor:
            self.thread_monitor.stop()
            self.thread_monitor = None

        state = logging_setup.bootstrap_logging(self.config)
        self.logging_state = state
        self.run_dir = state.run_dir

        self.logger = logging.getLogger('DictatorService')
        self.logger.info(
            "Logging initialized (run_dir=%s, structured=%s)",
            state.run_dir,
            state.structured,
        )

        if state.trace_threads:
            monitor_logger = logging.getLogger('DictatorService.ThreadMonitor')
            interval = self._resolve_thread_monitor_interval()
            self.thread_monitor = ThreadMonitor(
                interval_seconds=interval,
                logger=monitor_logger,
                run_dir=state.run_dir,
            )
            monitor_logger.debug("Thread monitor enabled (interval=%.1fs)", interval)
            self.thread_monitor.start()

    def _resolve_thread_monitor_interval(self) -> float:
        """Read and sanitize thread monitor interval from config."""

        profiling_cfg = self.config.get('profiling', {}) or {}
        raw_value = profiling_cfg.get('thread_monitor_interval', 30)
        try:
            interval = float(raw_value)
        except (TypeError, ValueError):
            interval = 30.0
        return max(interval, 1.0)

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

    def load_voice_session(self):
        """
        Load event-driven voice session (NEW)

        Replaces polling-based conversation manager with zero-token event system.
        Based on Speaches.ai architecture.
        """
        voice_config = self.config.get('voice', {})

        # Check BOTH flags: mode must be event_driven AND claude_mode must be enabled
        is_event_driven = voice_config.get('mode') == 'event_driven'
        is_claude_mode = voice_config.get('claude_mode', False)

        self.use_event_driven_mode = is_event_driven and is_claude_mode

        if not is_event_driven:
            self.logger.info("📢 Event-driven mode disabled (using legacy mode)")
            return

        if not is_claude_mode:
            self.logger.info("📢 Claude Mode disabled - using Dictation Mode (text only)")
            return

        if not VOICE_SESSION_AVAILABLE:
            self.logger.warning("⚠️ Voice session requested but module not available")
            return

        if not self.tts_engine:
            self.logger.warning("⚠️ Voice session requires TTS engine")
            return

        try:
            self.logger.info("🎯 Loading event-driven voice session...")

            # Get VAD enabled flag from hotkey config (same as normal dictation mode)
            vad_enabled = self.config['hotkey'].get('vad_enabled', False)
            self.logger.info(f"🎤 VAD enabled: {vad_enabled}")

            # Create VAD config
            vad_config_dict = voice_config.get('vad', {})
            vad_config = VADConfig(
                threshold=vad_config_dict.get('threshold', 0.5),
                min_silence_duration_ms=vad_config_dict.get('silence_duration_ms', 500),
                sample_rate=16000
            )

            # Create LLM caller based on provider
            llm_config = voice_config.get('llm', {})
            provider = llm_config.get('provider', 'claude-cli')

            if provider == 'ollama':
                # Ollama provider
                ollama_config = llm_config.get('ollama', {})
                base_url = ollama_config.get('base_url', 'http://localhost:11434')
                model = ollama_config.get('model', 'llama3.2:latest')
                
                self.logger.info(f"🦙 Using Ollama provider: {model} @ {base_url}")
                llm_caller = OllamaLLMCaller(
                    pubsub=None,  # Will be set by session manager
                    base_url=base_url,
                    model=model
                )
                
            elif provider == 'claude-direct':
                # Direct Anthropic API
                direct_config = llm_config.get('claude_direct', {})
                api_key = direct_config.get('api_key')
                model = direct_config.get('model', 'claude-sonnet-4-20250514')

                self.logger.info(f"🤖 Using Claude Direct API: {model}")
                llm_caller = DirectLLMCaller(
                    pubsub=None,  # Will be set by session manager
                    api_key=api_key,
                    model=model
                )

            elif provider == 'n8n_toolcalling':
                # N8N Tool-Calling provider
                n8n_config = llm_config.get('n8n_toolcalling', {})
                webhook_url = n8n_config.get('webhook_url', 'http://localhost:15678/webhook/dictator-llm')
                timeout = n8n_config.get('timeout', 120)

                self.logger.info(f"🔗 Using N8N Tool-Calling provider: {webhook_url}")
                llm_caller = N8NToolCallingLLMCaller(
                    pubsub=None,  # Will be set by session manager
                    webhook_url=webhook_url,
                    timeout=timeout
                )

            else:
                # Default: Claude CLI
                mcp_request_file = Path("temp/mcp_voice_request.json")
                mcp_response_file = Path("temp/mcp_voice_response.json")
                
                self.logger.info("🤖 Using Claude CLI provider")
                llm_caller = LLMCaller(
                    pubsub=None,  # Will be set by session manager
                    mcp_request_file=mcp_request_file,
                    mcp_response_file=mcp_response_file
                )

            # Create voice session manager
            self.voice_session = VoiceSessionManager(
                stt_callback=self.transcribe_audio,
                tts_callback=self.speak_text,
                llm_caller=llm_caller,
                vad_config=vad_config,
                vad_enabled=vad_enabled,
                error_callback=self._emit_state,  # Reset UI state on errors
                vad_stop_callback=self._on_vad_stop,  # Reset recording state when VAD detects stop
                tts_engine=self.tts_engine  # For TTS interrupt support
            )

            # Start session in background thread with asyncio loop
            import asyncio

            def run_voice_session():
                """
                Run voice session with error recovery

                Runs in dedicated asyncio loop with retry logic.
                """
                retry_count = 0
                max_retries = 3

                while retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self.voice_session_loop = loop  # Store reference

                    # Initialize VoiceSessionBridge (ADDED - Phase 5.1)
                    self.voice_session_bridge = VoiceSessionAudioBridge(
                        voice_session=self.voice_session,
                        event_loop=loop
                    )

                    try:
                        self.logger.info(f"🎤 Starting voice session (attempt {retry_count + 1}/{max_retries})")
                        loop.run_until_complete(self.voice_session.start())
                        self.logger.info("Voice session completed normally")
                        break  # Completed successfully

                    except asyncio.CancelledError:
                        self.logger.info("Voice session cancelled")
                        break

                    except Exception as e:
                        retry_count += 1
                        self.logger.error(f"❌ Voice session crashed: {e}", exc_info=True)

                        if retry_count < max_retries:
                            self.logger.info(f"🔄 Retrying in 2s... ({retry_count}/{max_retries})")
                            time.sleep(2)
                        else:
                            self.logger.error(f"❌ Max retries ({max_retries}) reached, giving up on voice session")
                            self.use_event_driven_mode = False  # Fallback to dictation

                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass
                        self.voice_session_loop = None

            self.voice_session_thread = threading.Thread(
                target=run_voice_session,
                daemon=True,
                name="VoiceSessionThread"
            )
            self.voice_session_thread.start()

            # Wait a moment for session to start
            time.sleep(0.1)

            self.logger.info("✅ Event-driven voice session loaded - ZERO polling!")

        except Exception as e:
            self.logger.error(f"❌ Failed to load voice session: {e}", exc_info=True)
            self.voice_session = None

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
        with self._state_lock:
            current_state = self._state

        if current_state == ServiceState.RECORDING:
            self.stop_recording()
        elif current_state in (ServiceState.IDLE, ServiceState.INTERRUPTED):
            self.start_recording()
        else:
            self.logger.warning(f"Cannot toggle recording in state: {current_state}")

    def _on_vad_stop(self):
        """
        Callback when VAD detects speech stopped

        Resets recording state so that next hotkey press starts a new recording
        instead of trying to stop an already-stopped recording.
        """
        self.logger.info("🎯 VAD detected speech stop, resetting recording state")
        # Transition to PROCESSING (transcription will happen)
        self._transition_state([ServiceState.RECORDING], ServiceState.PROCESSING)

    def start_recording(self):
        """Start recording audio"""
        # Transition to RECORDING state
        # Allow interrupting from SPEAKING (TTS) or PROCESSING (LLM) states
        if not self._transition_state(
            [ServiceState.IDLE, ServiceState.INTERRUPTED, ServiceState.SPEAKING, ServiceState.PROCESSING],
            ServiceState.RECORDING
        ):
            return

        vad_enabled = self.config['hotkey'].get('vad_enabled', False)
        vad_mode = " (VAD)" if vad_enabled else ""

        # Log interrupt if applicable
        with self._state_lock:
            old_state = self._state
            # ADDED - Phase 4.3: Increment session counter (invalidates pending TTS from previous session)
            self._session_counter += 1
            current_session = self._session_counter

        if old_state in [ServiceState.SPEAKING, ServiceState.PROCESSING]:
            self.logger.info(f"🚨 Interrupting {old_state.value} (session {current_session}) - user pressed hotkey to speak")

        self.logger.info(f"🔴 Recording started (session {current_session}){vad_mode}...")

        # Interrupt TTS if playing (user wants to speak)
        if self.tts_engine and self.tts_engine.is_speaking():
            self.logger.info("⏹️ Stopping TTS playback...")
            self.tts_engine.stop()
            time.sleep(0.1)  # Brief wait for TTS to fully stop

        # ADDED - Phase 4: Cancel LLM call if in PROCESSING state
        if old_state == ServiceState.PROCESSING and self.use_event_driven_mode and self.voice_session:
            self.logger.info("🛑 Cancelling LLM call...")
            if hasattr(self.voice_session.llm_caller, 'cancel_current_call'):
                self.voice_session.llm_caller.cancel_current_call()

        with self._recording_data_lock:
            self.recording_data = []
        self.last_sound_time = time.time()  # Track last time we heard sound
        self.audio_chunk_timestamp_ms = 0  # Reset timestamp for voice session

        # Clear voice session buffer if in event-driven mode
        if self.use_event_driven_mode and self.voice_session:
            self.voice_session.current_audio_buffer = np.array([], dtype=np.float32)
            self.logger.info("🧹 Cleared audio buffer")

        # Start recording in background thread
        threading.Thread(target=self._record_audio, daemon=True).start()

    def _record_audio(self):
        """Record audio in background (REFACTORED - Phase 5.1)"""
        sample_rate = self.config['audio']['sample_rate']
        channels = self.config['audio']['channels']
        vad_enabled = self.config['hotkey'].get('vad_enabled', False)
        silence_duration = self.config['hotkey'].get('auto_stop_silence', 2.0)
        silence_threshold = self.config['hotkey'].get('vad_threshold', 0.01)

        # Initialize AudioProcessor for this recording session
        def on_audio_chunk(chunk: np.ndarray):
            """Callback for each audio chunk - forwards to voice session if enabled"""
            if self.use_event_driven_mode and self.voice_session_bridge:
                timestamp_ms = self.audio_processor.get_timestamp_ms()
                self.voice_session_bridge.process_audio_chunk(chunk, timestamp_ms)

        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            on_audio_chunk=on_audio_chunk if self.use_event_driven_mode else None,
            vad_enabled=vad_enabled and not self.use_event_driven_mode,  # Only use VAD in legacy mode
            silence_threshold=silence_threshold
        )
        self.audio_processor.start()

        # Simplified sounddevice callback
        def callback(indata, frames, time_info, status):
            """Audio callback - delegates to AudioProcessor (SIMPLIFIED - Phase 5.1)"""
            if status:
                self.logger.warning(f"Audio status: {status}")

            # Thread-safe check of recording state
            with self._state_lock:
                is_currently_recording = self.is_recording

            if is_currently_recording:
                # Store in recording_data for backward compatibility
                with self._recording_data_lock:
                    self.recording_data.append(indata.copy())

                # Process chunk through AudioProcessor (FIXED - Phase 6: race condition protection)
                if self.audio_processor is not None:
                    self.audio_processor.process_chunk(indata)

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                callback=callback
            ):
                start_time = time.time()

                # Record until stopped
                while True:
                    # Thread-safe check
                    with self._state_lock:
                        if not self.is_recording:
                            break

                    time.sleep(0.1)

                    # VAD auto-stop: check if silence duration exceeded (REFACTORED - Phase 5.1)
                    # Skip in event-driven mode - VoiceSessionManager handles VAD
                    if vad_enabled and not self.use_event_driven_mode and self.audio_processor:
                        silence_time = self.audio_processor.get_vad_silence_duration()

                        # Only check after minimum recording time (0.5s)
                        if time.time() - start_time > 0.5 and silence_time >= silence_duration:
                            # Log RMS statistics for calibration
                            vad_stats = self.audio_processor.get_vad_stats()
                            if vad_stats['samples'] > 0:
                                self.logger.info(f"🔇 Silence detected ({silence_time:.1f}s)")
                                self.logger.info(
                                    f"   RMS stats: avg={vad_stats['avg_rms']:.5f}, "
                                    f"max={vad_stats['max_rms']:.5f}, "
                                    f"threshold={vad_stats['threshold']:.5f}"
                                )
                            self.stop_recording()
                            break

                    # Max duration check
                    max_duration = self.config['hotkey'].get('max_duration', 60)
                    if time.time() - start_time >= max_duration:
                        self.logger.warning(f"⏰ Max duration ({max_duration}s) reached, stopping...")
                        self.stop_recording()
                        break

                # Cleanup AudioProcessor BEFORE exiting InputStream context (FIXED - Phase 6)
                # This ensures no callbacks arrive after AudioProcessor is None
                if self.audio_processor:
                    self.audio_processor.stop()
                    self.audio_processor = None

        except Exception as e:
            self.logger.error(f"❌ Recording error: {e}")
            # Reset to IDLE on error
            self._transition_state([ServiceState.RECORDING], ServiceState.IDLE)
            # Cleanup AudioProcessor on error
            if self.audio_processor:
                self.audio_processor.stop()
                self.audio_processor = None

    def stop_recording(self):
        """Stop recording and transcribe"""
        # Transition to PROCESSING state
        if not self._transition_state([ServiceState.RECORDING], ServiceState.PROCESSING):
            return

        # ADDED - Phase 4.3: Capture session number for this processing cycle
        with self._state_lock:
            self._current_tts_session = self._session_counter

        self.logger.info(f"⏹️ Recording stopped (will process for session {self._current_tts_session})")

        # Event-driven mode with VAD disabled: emit SPEECH_STOPPED manually
        if self.use_event_driven_mode and self.voice_session:
            vad_enabled = self.config['hotkey'].get('vad_enabled', False)
            if not vad_enabled:
                # User stopped manually, trigger speech processing
                from .voice import Event, EventType
                self.logger.info("🎤 VAD disabled: emitting manual SPEECH_STOPPED event")
                
                # Interrupt TTS if playing (user wants to speak)
                if self.tts_engine and self.tts_engine.is_speaking():
                    self.logger.info("🚨 Interrupting TTS - user wants to speak")
                    self.tts_engine.stop()
                
                # Emit processing state BEFORE event (voice session will handle transcription)
                self._emit_state("processing")
                
                # Emit SPEECH_STOPPED event to voice session
                try:
                    subscriber_count = self.voice_session.pubsub.subscriber_count
                    self.logger.info(f"📢 Publishing SPEECH_STOPPED event to {subscriber_count} subscribers")
                    
                    self.voice_session.pubsub.publish_nowait(Event(
                        type=EventType.SPEECH_STOPPED,
                        data={},
                        session_id=self.voice_session.session_id
                    ))
                    
                    if subscriber_count == 0:
                        self.logger.error("❌ No subscribers! Event processor may have crashed")
                        self._emit_state("idle")
                        return
                    
                    self.logger.info("✅ SPEECH_STOPPED event published to queue")
                except Exception as e:
                    self.logger.error(f"❌ Failed to publish SPEECH_STOPPED event: {e}")
                    self._emit_state("idle")
                    return
                
                # In event-driven mode, voice session handles everything
                self.logger.info("🎯 Event-driven mode: Voice session will handle transcription via events")
                return

        if not self.recording_data:
            self.logger.warning("No audio recorded!")
            self._emit_state("idle")
            return

        # Emit processing state
        self._emit_state("processing")

        # Process in background thread
        threading.Thread(target=self._process_recording, daemon=True).start()

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data using Whisper

        Args:
            audio_data: Audio samples (float32, 16kHz)

        Returns:
            Transcribed text
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(
                temp_path,
                audio_data,
                self.config['audio']['sample_rate']
            )

        try:
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                temp_path,
                language=self.config['whisper']['language']
            )

            # Collect all text from segments
            text = "".join([segment.text for segment in segments]).strip()
            return text

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _process_recording(self):
        """Process recorded audio"""
        try:
            # Event-driven mode: voice session already handled everything during recording
            if self.use_event_driven_mode and self.voice_session:
                self.logger.info("🎯 Event-driven mode: Voice session processed speech automatically via events")
                self._emit_state("idle")
                return

            # Legacy/Dictation mode: process manually
            # Combine audio chunks (thread-safe copy)
            with self._recording_data_lock:
                recording_data_copy = self.recording_data.copy()

            audio_data = np.concatenate(recording_data_copy, axis=0)

            self.logger.info("🤖 Transcribing...")

            # Transcribe audio
            text = self.transcribe_audio(audio_data)
            self.logger.info(f"📝 Transcribed: {text}")

            # Paste to clipboard
            if text:
                self._paste_text(text)

            # Return to idle state
            self._transition_state([ServiceState.PROCESSING], ServiceState.IDLE)

        except Exception as e:
            self.logger.error(f"❌ Processing error: {e}", exc_info=True)
            self._transition_state([ServiceState.PROCESSING], ServiceState.IDLE)

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
            # No TTS, go straight to IDLE
            self._transition_state([ServiceState.PROCESSING, ServiceState.IDLE], ServiceState.IDLE)
            return

        # ADDED - Phase 4.3: Check if this TTS is for the current session
        with self._state_lock:
            current = self._state
            tts_session = self._current_tts_session
            active_session = self._session_counter

        if tts_session < active_session:
            # This TTS is from an old session - user already started new recording
            self.logger.info(f"🔇 Skipping stale TTS from session {tts_session} (current session: {active_session})")
            return

        if current == ServiceState.RECORDING:
            # User already started recording again (race condition) - skip TTS
            self.logger.info(f"🔇 Skipping TTS - user already recording new input (session {active_session})")
            return

        # Transition to SPEAKING
        # Allow from PROCESSING (normal flow) or IDLE (if LLM completed before TTS event arrived)
        if not self._transition_state([ServiceState.PROCESSING, ServiceState.IDLE], ServiceState.SPEAKING):
            self.logger.warning(f"⚠️ Cannot transition to SPEAKING from {current}, skipping TTS")
            return

        try:
            self.logger.info(f"🔊 Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
            # Use blocking=True to ensure sentence completes before next one starts
            self.tts_engine.speak(text, blocking=True)

            # After speaking, return to IDLE (if not already interrupted)
            # MODIFIED - Phase 4.1: Check current state - if interrupted, don't change
            with self._state_lock:
                current = self._state

            if current == ServiceState.SPEAKING:
                # Normal completion - return to IDLE
                self._transition_state([ServiceState.SPEAKING], ServiceState.IDLE)
            else:
                # Was interrupted - keep current state (RECORDING or PROCESSING)
                self.logger.info(f"🔄 TTS completed but state already changed to {current.value}, keeping it")

        except Exception as e:
            self.logger.error(f"❌ TTS error: {e}", exc_info=True)
            # On error, only transition if still in SPEAKING
            with self._state_lock:
                current = self._state
            if current == ServiceState.SPEAKING:
                self._transition_state([ServiceState.SPEAKING], ServiceState.IDLE)

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

    def _apply_health_degradation(self):
        """
        Apply feature degradation based on health report (NEW)

        Disables optional features when dependencies are unavailable,
        ensuring core dictation functionality remains operational.
        """
        self._tts_available = True
        self._voice_available = True

        for component in self.health_report.components:
            # Critical failures: abort startup
            if component.status == "critical" and component.required:
                self.logger.error(f"❌ CRITICAL: {component.message}")
                if component.fix_hint:
                    self.logger.error(f"   Fix: {component.fix_hint}")
                raise RuntimeError(f"Critical dependency missing: {component.name}")

            # Optional failures: disable features
            if component.status in ["unavailable", "critical"]:
                if component.name in ["Git LFS & Models", "TTS Model Files"]:
                    self._tts_available = False
                    self.logger.warning(f"⚠️ TTS disabled: {component.message}")
                    if component.fix_hint:
                        self.logger.warning(f"   Fix: {component.fix_hint}")

                elif component.name in ["Ollama", "N8N"]:
                    self._voice_available = False
                    self.logger.warning(f"⚠️ Voice assistant disabled: {component.message}")
                    if component.fix_hint:
                        self.logger.warning(f"   Fix: {component.fix_hint}")

                elif component.name == "GPU/CUDA" and component.status == "critical":
                    # Force CPU mode
                    self.config['whisper']['device'] = 'cpu'
                    self.logger.warning(f"⚠️ Fallback to CPU mode: {component.message}")
                    if component.fix_hint:
                        self.logger.warning(f"   Fix: {component.fix_hint}")

        # Log final status
        if self.health_report.overall_status == "healthy":
            self.logger.info("✅ All systems healthy")
        elif self.health_report.overall_status == "degraded":
            degraded_list = ', '.join(self.health_report.degraded_features) if self.health_report.degraded_features else 'features'
            self.logger.warning(f"⚠️ Running in degraded mode: {degraded_list} disabled")
            self.logger.info("ℹ️  Core dictation functionality remains available")

    def _write_status_file(self):
        """Write health status to JSON file (NEW)"""
        if not self.run_dir:
            return

        try:
            import json
            status_file = self.run_dir / "status.json"

            with open(status_file, 'w') as f:
                json.dump(self.health_report.to_dict(), f, indent=2)

            self.logger.debug(f"Status file written: {status_file}")

        except Exception as e:
            self.logger.warning(f"Failed to write status file: {e}")

    def stop(self):
        """Stop the service"""
        self.logger.info("🛑 Dictator Service stopping...")
        self.running = False

        if self.is_recording:
            self.stop_recording()

        # Stop voice session if running (event-driven mode)
        if self.voice_session:
            try:
                import asyncio
                if self.voice_session_loop and self.voice_session_loop.is_running():
                    # Schedule stop on the voice session's loop
                    asyncio.run_coroutine_threadsafe(
                        self.voice_session.stop(),
                        self.voice_session_loop
                    )
                self.logger.info("Voice session stopped")
            except Exception as e:
                self.logger.error(f"Error stopping voice session: {e}")

        if self.hotkey_listener:
            self.hotkey_listener.stop()

        if self.mouse_listener:
            self.mouse_listener.stop()

        if self.thread_monitor:
            self.thread_monitor.stop()
            self.thread_monitor = None

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
