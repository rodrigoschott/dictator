"""
Voice Activity Detection (VAD) Processor

Uses Silero VAD v5 via faster-whisper for real-time speech detection.
Runs locally on GPU/CPU with zero network latency.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray

try:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

from .events import Event, EventType, EventPubSub


@dataclass
class VADConfig:
    """VAD configuration (based on Speaches.ai)"""
    threshold: float = 0.5  # Speech probability threshold
    min_speech_duration_ms: int = 250  # Minimum speech duration
    max_speech_duration_ms: int = 30000  # Maximum speech duration (30s)
    min_silence_duration_ms: int = 500  # Silence to trigger stop (faster than Speaches: 2000ms)
    speech_pad_ms: int = 400  # Padding around speech
    sample_rate: int = 16000  # VAD expects 16kHz


@dataclass
class VADResult:
    """Result from VAD processing"""
    speech_active: bool
    speech_started: bool = False
    speech_stopped: bool = False
    speech_probability: float = 0.0
    timestamp_ms: int = 0


class VADProcessor:
    """
    Voice Activity Detection Processor

    Uses Silero VAD v5 for real-time speech detection.
    Emits events via EventPubSub when speech starts/stops.

    Based on Speaches.ai implementation:
    - Processes audio chunks in real-time
    - Uses GPU/CPU (no network calls)
    - Emits events instantly via publish_nowait()
    - Configurable thresholds and timeouts
    """

    def __init__(
        self,
        config: VADConfig,
        pubsub: EventPubSub,
        session_id: str | None = None
    ):
        if not VAD_AVAILABLE:
            raise ImportError(
                "faster-whisper not installed. "
                "Install with: pip install faster-whisper"
            )

        self.config = config
        self.pubsub = pubsub
        self.session_id = session_id

        # VAD options for faster-whisper
        # Note: max_speech_duration is in SECONDS, others in milliseconds
        self.vad_options = VadOptions(
            threshold=config.threshold,
            min_speech_duration_ms=config.min_speech_duration_ms,
            max_speech_duration_s=config.max_speech_duration_ms / 1000.0,  # Convert ms to seconds
            min_silence_duration_ms=config.min_silence_duration_ms,
            speech_pad_ms=config.speech_pad_ms,
        )

        # State tracking
        self.speech_active = False
        self.audio_buffer: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.buffer_start_timestamp_ms = 0

        # Debouncing for silence detection (prevent oscillation)
        self.last_speech_timestamp_ms = 0
        self.silence_start_timestamp_ms: Optional[int] = None

        # Silero VAD model (loaded lazily)
        self._vad_model: Optional[torch.nn.Module] = None

    @property
    def vad_model(self) -> torch.nn.Module:
        """Lazy load VAD model"""
        if self._vad_model is None:
            try:
                # Load Silero VAD v5 model
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False  # Use PyTorch version
                )
                self._vad_model = model
            except Exception as e:
                raise RuntimeError(f"Failed to load Silero VAD model: {e}")

        return self._vad_model

    def process_audio_chunk(
        self,
        audio_chunk: NDArray[np.float32],
        timestamp_ms: int
    ) -> VADResult:
        """
        Process audio chunk for speech detection

        Args:
            audio_chunk: Audio data (float32, 16kHz)
            timestamp_ms: Timestamp of this chunk

        Returns:
            VADResult with detection info
        """
        # Append to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

        # Keep only last 30 seconds (max speech duration)
        max_samples = (self.config.max_speech_duration_ms * self.config.sample_rate) // 1000
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
            self.buffer_start_timestamp_ms = timestamp_ms - self.config.max_speech_duration_ms

        # Run VAD on buffer
        result = self._detect_speech(timestamp_ms)

        # Emit events on state changes
        if result.speech_started:
            self.pubsub.publish_nowait(Event(
                type=EventType.SPEECH_STARTED,
                data={
                    "timestamp_ms": timestamp_ms,
                    "probability": result.speech_probability
                },
                session_id=self.session_id
            ))

        if result.speech_stopped:
            self.pubsub.publish_nowait(Event(
                type=EventType.SPEECH_STOPPED,
                data={
                    "timestamp_ms": timestamp_ms,
                    "duration_ms": timestamp_ms - self.buffer_start_timestamp_ms,
                    "audio_length_samples": len(self.audio_buffer)
                },
                session_id=self.session_id
            ))

        return result

    def _detect_speech(self, current_timestamp_ms: int) -> VADResult:
        """
        Detect speech in current buffer using Silero VAD

        Uses a sliding window approach to detect recent speech,
        not just historical speech in the buffer.

        Returns:
            VADResult with detection info
        """
        # Need at least 512 samples for VAD
        if len(self.audio_buffer) < 512:
            return VADResult(
                speech_active=self.speech_active,
                speech_probability=0.0,
                timestamp_ms=current_timestamp_ms
            )

        # Use sliding window: analyze only recent audio
        # Window = 2x silence_duration to ensure we detect silence properly
        # This prevents old speech from keeping speech_detected = True
        window_duration_ms = self.config.min_silence_duration_ms * 2
        window_samples = int((window_duration_ms * self.config.sample_rate) / 1000)

        # Get recent window from buffer
        if len(self.audio_buffer) > window_samples:
            recent_audio = self.audio_buffer[-window_samples:]
        else:
            recent_audio = self.audio_buffer

        # Get speech timestamps using faster-whisper VAD on recent window
        try:
            timestamps = get_speech_timestamps(
                audio=recent_audio,
                vad_options=self.vad_options,
                sampling_rate=self.config.sample_rate
            )

            # Check if speech is currently active in RECENT audio
            speech_detected = len(timestamps) > 0

            # Debug logging
            import logging
            logger = logging.getLogger("DictatorService")

            # Determine state changes with debouncing
            speech_started = False
            speech_stopped = False

            if speech_detected:
                # Speech is happening NOW
                self.last_speech_timestamp_ms = current_timestamp_ms
                self.silence_start_timestamp_ms = None  # Reset silence timer

                if not self.speech_active:
                    # Speech just started
                    speech_started = True
                    self.speech_active = True
                    self.buffer_start_timestamp_ms = current_timestamp_ms
                    logger.debug(f"VAD: Speech STARTED (window={window_duration_ms}ms)")

            else:
                # No speech detected in recent window
                if self.speech_active:
                    # We were speaking, now checking for silence

                    if self.silence_start_timestamp_ms is None:
                        # First silence detection - start timer
                        self.silence_start_timestamp_ms = current_timestamp_ms
                        logger.debug(f"VAD: Silence detected, starting debounce timer")
                    else:
                        # Ongoing silence - check if duration exceeded
                        silence_duration_ms = current_timestamp_ms - self.silence_start_timestamp_ms

                        if silence_duration_ms >= self.config.min_silence_duration_ms:
                            # Confirmed silence for min_silence_duration_ms
                            speech_stopped = True
                            self.speech_active = False
                            self.silence_start_timestamp_ms = None
                            logger.debug(f"VAD: Speech STOPPED after {silence_duration_ms}ms of silence")

            # Calculate average probability (simplified)
            probability = 0.6 if speech_detected else 0.3

            return VADResult(
                speech_active=self.speech_active,
                speech_started=speech_started,
                speech_stopped=speech_stopped,
                speech_probability=probability,
                timestamp_ms=current_timestamp_ms
            )

        except Exception as e:
            import logging
            logging.getLogger("DictatorService").error(f"VAD detection error: {e}")
            return VADResult(
                speech_active=self.speech_active,
                timestamp_ms=current_timestamp_ms
            )

    def get_speech_audio(self) -> NDArray[np.float32]:
        """
        Get buffered speech audio

        Returns:
            Audio buffer (float32, 16kHz)
        """
        return self.audio_buffer.copy()

    def clear_buffer(self) -> None:
        """Clear audio buffer"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_start_timestamp_ms = 0
        self.speech_active = False

    def unload(self) -> None:
        """Unload VAD model to free memory"""
        if self._vad_model is not None:
            del self._vad_model
            self._vad_model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
