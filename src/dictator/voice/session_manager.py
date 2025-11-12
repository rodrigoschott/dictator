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
        session_id: Optional[str] = None
    ):
        """
        Initialize voice session

        Args:
            stt_callback: Function to transcribe audio (Whisper)
            tts_callback: Function to synthesize speech (Kokoro)
            llm_caller: LLM caller for conversation
            vad_config: VAD configuration
            session_id: Session identifier
        """
        self.stt_callback = stt_callback
        self.tts_callback = tts_callback
        self.llm_caller = llm_caller
        self.session_id = session_id or f"session_{int(time.time())}"

        # Event system
        self.pubsub = EventPubSub()

        # Connect pubsub and session_id to LLM caller (it was passed with pubsub=None)
        self.llm_caller.pubsub = self.pubsub
        self.llm_caller.session_id = self.session_id

        # VAD processor
        self.vad_config = vad_config or VADConfig()
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
        # Publish audio chunk event
        self.pubsub.publish_nowait(Event(
            type=EventType.AUDIO_CHUNK,
            data={
                "timestamp_ms": timestamp_ms,
                "length_samples": len(audio_chunk)
            },
            session_id=self.session_id
        ))

        # Process with VAD
        self.vad_processor.process_audio_chunk(audio_chunk, timestamp_ms)

        # Buffer audio
        self.current_audio_buffer = np.append(self.current_audio_buffer, audio_chunk)

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

        async for event in self.pubsub.poll():
            # Don't log AUDIO_CHUNK events to avoid spam (80+ per second)
            if event.type != EventType.AUDIO_CHUNK:
                logger.info(f"📨 Event received: {event.type}")
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

        # Skip AUDIO_CHUNK events - they're processed inline in process_audio_chunk()
        # and don't need event loop handling. This prevents queue flooding.
        if event.type == EventType.AUDIO_CHUNK:
            return

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
            # Get audio from VAD processor
            audio = self.vad_processor.get_speech_audio()

            # Check if we have enough audio to transcribe (at least 0.5 seconds)
            min_samples = int(0.5 * 16000)  # 0.5s at 16kHz
            if len(audio) < min_samples:
                import logging
                logging.getLogger("DictatorService").warning(
                    f"⚠️ Ignoring speech event with insufficient audio ({len(audio)} samples, need {min_samples})"
                )
                # Clear buffer and skip transcription
                self.vad_processor.clear_buffer()
                self.current_audio_buffer = np.array([], dtype=np.float32)
                return

            # Transcribe (LOCAL - no tokens!)
            transcription = await asyncio.to_thread(
                self.stt_callback,
                audio
            )

            # Clear VAD buffer
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

        # Use lock to ensure sentences play sequentially without interruption
        async with self.tts_lock:
            try:
                logger.info(f"🔊 Starting TTS synthesis...")
                # Synthesize (LOCAL - no tokens!)
                await asyncio.to_thread(
                    self.tts_callback,
                    sentence
                )
                logger.info(f"✅ TTS synthesis completed")

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
