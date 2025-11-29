#!/usr/bin/env python3
"""
Audio Processing Module

Encapsulates audio callback logic with proper thread safety and separation of concerns.
"""

import logging
import threading
import time
from typing import Optional, Callable
import asyncio

import numpy as np


class AudioProcessor:
    """
    Thread-safe audio processor for real-time audio callbacks

    Responsibilities:
    - Collect audio chunks from sounddevice callback
    - Forward chunks to voice session (event-driven mode)
    - Calculate VAD metrics (legacy mode)
    - Maintain thread-safe state
    """

    def __init__(
        self,
        sample_rate: int,
        on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None,
        vad_enabled: bool = False,
        silence_threshold: float = 0.01
    ):
        """
        Initialize audio processor

        Args:
            sample_rate: Audio sample rate (Hz)
            on_audio_chunk: Optional callback for each audio chunk
            vad_enabled: Enable VAD (Voice Activity Detection)
            silence_threshold: RMS threshold for VAD
        """
        self.logger = logging.getLogger('AudioProcessor')
        self.sample_rate = sample_rate
        self.on_audio_chunk = on_audio_chunk
        self.vad_enabled = vad_enabled
        self.silence_threshold = silence_threshold

        # Thread-safe state
        self._lock = threading.RLock()
        self._is_active = False
        self._audio_chunks = []

        # VAD state (protected by lock)
        self._last_sound_time = 0.0
        self._rms_samples = []
        self._max_rms = 0.0

        # Timestamp tracking (for voice session)
        self._timestamp_ms = 0

        # Health check (one-time log)
        self._health_check_logged = False

    def start(self):
        """Start audio processing"""
        with self._lock:
            self._is_active = True
            self._audio_chunks = []
            self._timestamp_ms = 0
            self._last_sound_time = time.time()
            self._rms_samples = []
            self._max_rms = 0.0
            self._health_check_logged = False

        self.logger.debug("Audio processor started")

    def stop(self):
        """Stop audio processing"""
        with self._lock:
            self._is_active = False

        self.logger.debug("Audio processor stopped")

    def is_active(self) -> bool:
        """Check if processor is active"""
        with self._lock:
            return self._is_active

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Process single audio chunk (called from sounddevice callback)

        Args:
            audio_chunk: Audio data from sounddevice

        Returns:
            True if chunk was processed, False if processor is inactive
        """
        with self._lock:
            if not self._is_active:
                return False

            # Store chunk
            chunk_copy = audio_chunk.copy()
            self._audio_chunks.append(chunk_copy)

            # Update VAD metrics
            if self.vad_enabled:
                self._update_vad_metrics(audio_chunk)

            # Increment timestamp
            chunk_duration_ms = int(len(audio_chunk) * 1000 / self.sample_rate)
            self._timestamp_ms += chunk_duration_ms

            # Log health check after 1 second
            if not self._health_check_logged and self._timestamp_ms >= 1000:
                self.logger.info("✅ Audio processor healthy, receiving chunks")
                self._health_check_logged = True

        # Call external callback (outside lock to avoid deadlock)
        if self.on_audio_chunk:
            try:
                self.on_audio_chunk(chunk_copy)
            except Exception as e:
                self.logger.warning(f"Audio chunk callback failed: {e}")

        return True

    def _update_vad_metrics(self, audio_chunk: np.ndarray):
        """Update VAD (Voice Activity Detection) metrics (lock must be held)"""
        # Calculate RMS (root mean square) amplitude
        rms = np.sqrt(np.mean(audio_chunk**2))
        self._rms_samples.append(rms)

        # Track maximum RMS
        if rms > self._max_rms:
            self._max_rms = rms

        # Adaptive threshold: use either fixed threshold or 20% of max RMS
        adaptive_threshold = min(self.silence_threshold, self._max_rms * 0.2)

        # If amplitude is above threshold, reset silence timer
        if rms > adaptive_threshold:
            self._last_sound_time = time.time()

    def get_vad_silence_duration(self) -> float:
        """Get duration of silence since last sound (seconds)"""
        with self._lock:
            return time.time() - self._last_sound_time

    def get_vad_stats(self) -> dict:
        """Get VAD statistics for debugging"""
        with self._lock:
            if not self._rms_samples:
                return {
                    'avg_rms': 0.0,
                    'max_rms': 0.0,
                    'threshold': self.silence_threshold,
                    'samples': 0
                }

            return {
                'avg_rms': float(np.mean(self._rms_samples)),
                'max_rms': float(np.max(self._rms_samples)),
                'threshold': self.silence_threshold,
                'samples': len(self._rms_samples)
            }

    def get_audio_data(self) -> np.ndarray:
        """Get all collected audio chunks as single array"""
        with self._lock:
            if not self._audio_chunks:
                return np.array([], dtype=np.float32)

            return np.concatenate(self._audio_chunks)

    def get_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds"""
        with self._lock:
            return self._timestamp_ms

    def clear_chunks(self):
        """Clear collected audio chunks"""
        with self._lock:
            self._audio_chunks = []
            self._timestamp_ms = 0


class VoiceSessionAudioBridge:
    """
    Bridge between AudioProcessor and VoiceSessionManager

    Handles async communication with voice session from sync audio callback.
    """

    def __init__(self, voice_session, event_loop: asyncio.AbstractEventLoop):
        """
        Initialize bridge

        Args:
            voice_session: VoiceSessionManager instance
            event_loop: Event loop running in voice session thread
        """
        self.logger = logging.getLogger('VoiceSessionBridge')
        self.voice_session = voice_session
        self.event_loop = event_loop

        # Track first chunk for logging
        self._first_chunk_logged = False

    def process_audio_chunk(self, audio_chunk: np.ndarray, timestamp_ms: int):
        """
        Send audio chunk to voice session (called from audio callback thread)

        Args:
            audio_chunk: Audio data (float32, mono)
            timestamp_ms: Timestamp in milliseconds
        """
        # Convert to float32 mono if needed
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Flatten to 1D if stereo
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        # Log first chunk
        if not self._first_chunk_logged:
            self.logger.debug("🎵 Voice session started receiving audio chunks")
            self._first_chunk_logged = True

        # Schedule async call in voice session's event loop
        try:
            asyncio.run_coroutine_threadsafe(
                self.voice_session.process_audio_chunk(audio_chunk, timestamp_ms),
                self.event_loop
            )
        except Exception as e:
            self.logger.warning(f"Failed to send chunk to voice session: {e}")

    def reset(self):
        """Reset bridge state"""
        self._first_chunk_logged = False
