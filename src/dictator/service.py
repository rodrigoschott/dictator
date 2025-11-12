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

try:
    from .voice import (
        VoiceSessionManager,
        VADConfig,
        LLMCaller,
        DirectLLMCaller,
        OllamaLLMCaller
    )
    VOICE_SESSION_AVAILABLE = True
except ImportError:
    VOICE_SESSION_AVAILABLE = False


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

        # Event-driven voice session (NEW - replaces old polling system)
        self.voice_session: Optional[VoiceSessionManager] = None
        self.voice_session_thread: Optional[threading.Thread] = None
        self.voice_session_loop: Optional['asyncio.AbstractEventLoop'] = None
        self.use_event_driven_mode = False
        self.audio_chunk_timestamp_ms = 0  # Track timestamp for audio chunks

        # State change callbacks
        self.state_callbacks = []

        # Load Whisper model
        self.load_model()

        # Load TTS engine if enabled
        self.load_tts()

        # Load event-driven voice session if enabled
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
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def _on_vad_stop(self):
        """
        Callback when VAD detects speech stopped
        
        Resets recording state so that next hotkey press starts a new recording
        instead of trying to stop an already-stopped recording.
        """
        if self.is_recording:
            self.logger.info("🎯 VAD detected speech stop, resetting recording state")
            self.is_recording = False

    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            self.logger.warning("Already recording!")
            return

        vad_enabled = self.config['hotkey'].get('vad_enabled', False)
        vad_mode = " (VAD)" if vad_enabled else ""
        self.logger.info(f"🔴 Recording started{vad_mode}...")

        # Interrupt TTS if playing (user wants to speak)
        if self.tts_engine and self.tts_engine.is_speaking():
            self.logger.info("🚨 Interrupting TTS - user pressed hotkey to speak")
            self.tts_engine.stop()
            time.sleep(0.1)  # Brief wait for TTS to fully stop

        self.is_recording = True
        self.recording_data = []
        self.last_sound_time = time.time()  # Track last time we heard sound
        self.audio_chunk_timestamp_ms = 0  # Reset timestamp for voice session

        # Clear voice session buffer if in event-driven mode
        if self.use_event_driven_mode and self.voice_session:
            self.voice_session.current_audio_buffer = np.array([], dtype=np.float32)
            self.logger.info("🧹 Cleared audio buffer")

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
                audio_chunk = indata.copy()
                self.recording_data.append(audio_chunk)

                # Pass chunk to event-driven voice session if enabled
                if self.use_event_driven_mode and self.voice_session:
                    # Convert to float32 if needed
                    if audio_chunk.dtype != np.float32:
                        audio_chunk = audio_chunk.astype(np.float32)

                    # Flatten to 1D if stereo
                    if len(audio_chunk.shape) > 1:
                        audio_chunk = audio_chunk.mean(axis=1)

                    # Pass to voice session (synchronous wrapper for async call)
                    import asyncio
                    try:
                        # Use the voice session's event loop
                        if self.voice_session_loop:
                            # Log first chunk for health check
                            if self.audio_chunk_timestamp_ms == 0:
                                self.logger.debug("🎵 Voice session started receiving audio chunks")

                            asyncio.run_coroutine_threadsafe(
                                self.voice_session.process_audio_chunk(
                                    audio_chunk,
                                    self.audio_chunk_timestamp_ms
                                ),
                                self.voice_session_loop
                            )

                            # Log after 1 second to confirm session is alive
                            if self.audio_chunk_timestamp_ms == 0:
                                self.last_health_check_time = time.time()
                            elif self.audio_chunk_timestamp_ms >= 1000 and hasattr(self, 'last_health_check_time'):
                                if time.time() - self.last_health_check_time < 2:
                                    self.logger.info("✅ Voice session healthy, processing audio")
                                    delattr(self, 'last_health_check_time')  # Log only once

                            # Increment timestamp (assume chunks at 16kHz)
                            self.audio_chunk_timestamp_ms += int(len(audio_chunk) * 1000 / sample_rate)
                    except Exception as e:
                        self.logger.warning(f"Failed to pass audio chunk to voice session: {e}")

                # VAD: Check if audio contains sound (not silence)
                if vad_enabled and not self.use_event_driven_mode:
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
                    # Skip in event-driven mode - VoiceSessionManager handles VAD
                    if vad_enabled and not self.use_event_driven_mode:
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
            # Combine audio chunks
            audio_data = np.concatenate(self.recording_data, axis=0)

            self.logger.info("🤖 Transcribing...")

            # Transcribe audio
            text = self.transcribe_audio(audio_data)
            self.logger.info(f"📝 Transcribed: {text}")

            # Paste to clipboard
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
            # Use blocking=True to ensure sentence completes before next one starts
            self.tts_engine.speak(text, blocking=True)
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
