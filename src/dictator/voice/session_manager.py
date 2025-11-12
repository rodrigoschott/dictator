"""
Voice Session Manager

Coordinates all voice processing components in an event-driven architecture.
Runs 100% locally - NO continuous polling, NO wasted tokens.

Based on Speaches.ai pattern:
- Parallel async tasks
- Event-driven communication (PubSub)
- Single LLM call per utterance
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from .events import Event, EventType, EventPubSub
from .vad_processor import VADProcessor, VADConfig
from .llm_caller import LLMCaller
from .sentence_chunker import SentenceChunker


class VoiceSessionManager:
    """
    Manages a voice interaction session

    Coordinates:
    - Audio input buffering
    - VAD (speech detection)
    - STT (transcription)
    - LLM (conversation)
    - TTS (speech synthesis)

    All in an event-driven, zero-polling architecture.

    Usage:
        manager = VoiceSessionManager(
            stt_callback=whisper.transcribe,
            tts_callback=kokoro.synthesize,
            ...
        )

        # Start session
        await manager.start()

        # Feed audio chunks
        await manager.process_audio_chunk(audio_data, timestamp_ms)

        # Stop session
        await manager.stop()
    """

    def __init__(
        self,
        stt_callback: Callable[[NDArray[np.float32]], str],
        tts_callback: Callable[[str], None],
        llm_caller: LLMCaller,
        vad_config: Optional[VADConfig] = None,
        vad_enabled: bool = True,
        session_id: Optional[str] = None,
        error_callback: Optional[Callable[[str], None]] = None,
        vad_stop_callback: Optional[Callable[[], None]] = None,
        tts_engine = None
    ):
        """
        Initialize voice session

        Args:
            stt_callback: Function to transcribe audio (Whisper)
            tts_callback: Function to synthesize speech (Kokoro)
            llm_caller: LLM caller for conversation
            vad_config: VAD configuration
            vad_enabled: Whether VAD is enabled (for auto-stop on silence)
            session_id: Session identifier
            error_callback: Optional callback for errors (to update UI state)
            vad_stop_callback: Optional callback when VAD detects speech stopped (to reset recording state)
            tts_engine: Optional TTS engine for interrupt support
        """
        self.stt_callback = stt_callback
        self.tts_callback = tts_callback
        self.llm_caller = llm_caller
        self.session_id = session_id or f"session_{int(time.time())}"
        self.error_callback = error_callback
        self.vad_stop_callback = vad_stop_callback
        self.tts_engine = tts_engine  # For TTS interrupt support

        # Event system
        self.pubsub = EventPubSub()

        # Connect pubsub and session_id to LLM caller (it was passed with pubsub=None)
        self.llm_caller.pubsub = self.pubsub
        self.llm_caller.session_id = self.session_id

        # VAD processor
        self.vad_config = vad_config or VADConfig()
        self.vad_enabled = vad_enabled  # Control whether VAD detects silence for auto-stop
        self.vad_processor = VADProcessor(
            config=self.vad_config,
            pubsub=self.pubsub,
            session_id=self.session_id
        )

        # State
        self.running = False
        self.event_processor_task: Optional[asyncio.Task] = None

        # TTS lock to prevent sentence interruption
        self.tts_lock = asyncio.Lock()

        # TTS speaking flag - used to pause VAD during TTS output
        # This prevents the microphone from picking up TTS audio as user speech
        self.tts_speaking = False

        # Audio state
        self.current_audio_buffer: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.buffer_start_time = 0.0

    async def start(self) -> None:
        """
        Start voice session

        This coroutine blocks until the session is stopped.
        It runs the event processor loop indefinitely.
        """
        if self.running:
            return

        self.running = True

        # Set event loop for thread-safe publishing (CRITICAL for cross-thread events)
        import asyncio
        import logging
        loop = asyncio.get_event_loop()
        self.pubsub.set_event_loop(loop)
        logging.getLogger("DictatorService").info(f"🔗 Event loop configured for thread-safe publishing")

        # Emit session started
        self.pubsub.publish_nowait(Event(
            type=EventType.SESSION_STARTED,
            data={"session_id": self.session_id},
            session_id=self.session_id
        ))

        # Start event processor task
        self.event_processor_task = asyncio.create_task(
            self._event_processor(),
            name=f"event_processor_{self.session_id}"
        )

        # IMPORTANT: Await the task to keep the session alive
        # This blocks until stop() cancels the task
        try:
            await self.event_processor_task
        except asyncio.CancelledError:
            # Expected when stop() is called
            pass

    async def stop(self) -> None:
        """Stop voice session"""
        if not self.running:
            return

        self.running = False

        # Cancel event processor
        if self.event_processor_task:
            self.event_processor_task.cancel()
            try:
                await self.event_processor_task
            except asyncio.CancelledError:
                pass

        # Emit session ended
        self.pubsub.publish_nowait(Event(
            type=EventType.SESSION_ENDED,
            data={"session_id": self.session_id},
            session_id=self.session_id
        ))

        # Cleanup
        self.vad_processor.unload()

    async def process_audio_chunk(
        self,
        audio_chunk: NDArray[np.float32],
        timestamp_ms: int
    ) -> None:
        """
        Process incoming audio chunk

        This is called from the audio capture loop.
        VAD processor will emit events when speech starts/stops.

        Args:
            audio_chunk: Audio data (float32, 16kHz)
            timestamp_ms: Timestamp of this chunk
        """
        # DON'T publish AUDIO_CHUNK events - they flood the queue and delay critical events
        # Audio is processed inline here, no need for event loop
        # This prevents SPEECH_STOPPED from being delayed by 320+ queued AUDIO_CHUNK events

        # Process with VAD ONLY if enabled AND TTS is not speaking
        # VAD enabled: Detects silence for auto-stop
        # VAD disabled: User must stop manually via hotkey
        if self.vad_enabled and not self.tts_speaking:
            self.vad_processor.process_audio_chunk(audio_chunk, timestamp_ms)

        # Buffer audio
        self.current_audio_buffer = np.append(self.current_audio_buffer, audio_chunk)
        
        # Debug log for first few chunks when VAD disabled
        import logging
        logger = logging.getLogger("DictatorService")
        if not self.vad_enabled and len(self.current_audio_buffer) < 50000:  # First ~3 seconds
            if len(self.current_audio_buffer) % 16000 < len(audio_chunk):  # Log every ~1 second
                logger.debug(f"📊 Buffer size: {len(self.current_audio_buffer)} samples ({len(self.current_audio_buffer)/16000:.2f}s)")

    async def _event_processor(self) -> None:
        """
        Main event processing loop

        This is the CORE of the event-driven architecture.
        Listens for events and dispatches handlers.

        Runs 100% locally - NO Claude Code involvement here!
        """
        import logging
        logger = logging.getLogger("DictatorService")
        logger.info("🔄 Event processor started")

        # Track event count for debugging
        event_count = 0
        
        async for event in self.pubsub.poll():
            event_count += 1
            logger.info(f"📨 Event received: {event.type} (total events: {event_count})")
            
            try:
                await self._handle_event(event)
            except Exception as e:
                logger.error(f"❌ Error handling event {event.type}: {e}")
                import traceback
                logger.error(f"📋 Traceback: {traceback.format_exc()}")
                self.pubsub.publish_nowait(Event(
                    type=EventType.SESSION_ERROR,
                    data={"error": str(e), "event_type": event.type},
                    session_id=self.session_id
                ))

    async def _handle_event(self, event: Event) -> None:
        """
        Handle individual event

        Routes events to appropriate handlers.
        """
        import logging
        logger = logging.getLogger("DictatorService")

        logger.info(f"🎯 Handling event: {event.type}")

        # Speech started
        if event.type == EventType.SPEECH_STARTED:
            # Log speech detection
            import logging
            logging.getLogger("DictatorService").info("🎤 Speech detected by VAD")

        # Speech stopped → Transcribe
        elif event.type == EventType.SPEECH_STOPPED:
            import logging
            logging.getLogger("DictatorService").info("🛑 Speech ended, transcribing...")
            
            # Notify service to reset recording state (if VAD detected the stop)
            if self.vad_enabled and self.vad_stop_callback:
                self.vad_stop_callback()
            
            await self._handle_speech_stopped(event)

        # Transcription completed → Call LLM
        elif event.type == EventType.TRANSCRIPTION_COMPLETED:
            import logging
            logging.getLogger("DictatorService").info("📝 Transcription complete, calling LLM...")
            await self._handle_transcription_completed(event)

        # TTS sentence ready → Synthesize
        elif event.type == EventType.TTS_SENTENCE_READY:
            import logging
            logging.getLogger("DictatorService").info("🔊 TTS sentence ready, synthesizing...")
            await self._handle_tts_sentence_ready(event)

        # Error events → Reset to idle state
        elif event.type in (EventType.SESSION_ERROR, EventType.LLM_RESPONSE_FAILED, EventType.TRANSCRIPTION_FAILED, EventType.TTS_FAILED):
            import logging
            logging.getLogger("DictatorService").warning(f"⚠️ Error event received: {event.type}")
            # Call error callback to reset UI state
            if self.error_callback:
                self.error_callback("idle")

        # Session ended → Stop processing
        elif event.type == EventType.SESSION_ENDED:
            return

    async def _handle_speech_stopped(self, event: Event) -> None:
        """
        Handle speech stopped event

        Transcribe the buffered audio using Whisper (local).
        """
        # Emit transcription started
        self.pubsub.publish_nowait(Event(
            type=EventType.TRANSCRIPTION_STARTED,
            data={},
            session_id=self.session_id
        ))

        try:
            # Get audio depending on VAD mode
            if self.vad_enabled:
                # VAD enabled: use VAD processor's buffer (contains detected speech)
                audio = self.vad_processor.get_speech_audio()
            else:
                # VAD disabled: use session's buffer (all recorded audio)
                audio = self.current_audio_buffer

            # Debug log
            import logging
            logger = logging.getLogger("DictatorService")
            logger.info(f"🎯 Speech stopped - VAD mode: {'enabled' if self.vad_enabled else 'disabled'}, Buffer size: {len(audio)} samples ({len(audio)/16000:.2f}s)")

            # Check if we have enough audio to transcribe (at least 0.5 seconds)
            min_samples = int(0.5 * 16000)  # 0.5s at 16kHz
            if len(audio) < min_samples:
                logger.warning(
                    f"⚠️ Ignoring speech event with insufficient audio ({len(audio)} samples, need {min_samples})"
                )
                # Clear buffer and skip transcription
                if self.vad_enabled:
                    self.vad_processor.clear_buffer()
                self.current_audio_buffer = np.array([], dtype=np.float32)
                return

            # Transcribe (LOCAL - no tokens!)
            transcription = await asyncio.to_thread(
                self.stt_callback,
                audio
            )

            # Clear buffers
            if self.vad_enabled:
                self.vad_processor.clear_buffer()
            self.current_audio_buffer = np.array([], dtype=np.float32)

            # Only emit if transcription is not empty
            if transcription and transcription.strip():
                # Emit transcription completed
                self.pubsub.publish_nowait(Event(
                    type=EventType.TRANSCRIPTION_COMPLETED,
                    data={"transcription": transcription},
                    session_id=self.session_id
                ))
            else:
                import logging
                logging.getLogger("DictatorService").debug("Empty transcription, skipping LLM call")

        except Exception as e:
            self.pubsub.publish_nowait(Event(
                type=EventType.TRANSCRIPTION_FAILED,
                data={"error": str(e)},
                session_id=self.session_id
            ))

    async def _handle_transcription_completed(self, event: Event) -> None:
        """
        Handle transcription completed event

        Call LLM ONCE with transcription.
        This is the ONLY token-consuming operation!
        """
        transcription = event.data.get("transcription", "")

        if not transcription.strip():
            return

        import logging
        logger = logging.getLogger("DictatorService")
        logger.info(f"📝 Transcribed text: '{transcription}'")

        # Call LLM (SINGLE CALL - efficient!)
        await self.llm_caller.process_transcription(transcription)

    async def _handle_tts_sentence_ready(self, event: Event) -> None:
        """
        Handle TTS sentence ready event

        Synthesize speech using Kokoro (local).
        """
        import logging
        logger = logging.getLogger("DictatorService")

        sentence = event.data.get("text", "")
        logger.info(f"🎵 TTS handler received: '{sentence[:100]}...'")

        if not sentence.strip():
            logger.warning("⚠️ Empty sentence, skipping TTS")
            return

        # Use lock to ensure sentences play sequentially
        async with self.tts_lock:
            try:
                # Set flag to pause VAD during TTS
                self.tts_speaking = True

                logger.info("🔊 Starting TTS synthesis...")
                # Synthesize (LOCAL - no tokens!)
                await asyncio.to_thread(
                    self.tts_callback,
                    sentence
                )
                logger.info("✅ TTS synthesis completed")

                # Emit audio generated
                self.pubsub.publish_nowait(Event(
                    type=EventType.TTS_AUDIO_GENERATED,
                    data={"text": sentence},
                    session_id=self.session_id
                ))

            except Exception as e:
                logger.error(f"❌ TTS synthesis failed: {e}")
                self.pubsub.publish_nowait(Event(
                    type=EventType.TTS_FAILED,
                    data={"error": str(e)},
                    session_id=self.session_id
                ))
            finally:
                # Always re-enable VAD after TTS completes (even on error)
                self.tts_speaking = False
                logger.info("🎤 VAD re-enabled after TTS")

    def get_stats(self) -> dict:
        """Get session statistics"""
        return {
            "session_id": self.session_id,
            "running": self.running,
            "subscribers": self.pubsub.subscriber_count,
            "recent_events": len(self.pubsub.get_recent_events(10)),
            "vad_active": self.vad_processor.speech_active,
            "buffer_size": len(self.current_audio_buffer)
        }
