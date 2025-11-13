"""
Event-Driven PubSub System for Voice Processing

Based on Speaches.ai architecture - uses async queues with blocking wait
instead of polling loops. Zero polling overhead.
"""

import asyncio
from enum import Enum
from typing import Any, AsyncGenerator
from collections.abc import Set
from dataclasses import dataclass, field
from datetime import datetime


class EventType(str, Enum):
    """Voice processing event types"""

    # Audio events
    AUDIO_CHUNK = "audio.chunk"
    AUDIO_BUFFER_CLEAR = "audio.buffer.clear"

    # Speech detection events (VAD)
    SPEECH_STARTED = "speech.started"
    SPEECH_STOPPED = "speech.stopped"
    SPEECH_TIMEOUT = "speech.timeout"

    # Transcription events
    TRANSCRIPTION_STARTED = "transcription.started"
    TRANSCRIPTION_COMPLETED = "transcription.completed"
    TRANSCRIPTION_FAILED = "transcription.failed"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE_STARTED = "llm.response.started"
    LLM_RESPONSE_CHUNK = "llm.response.chunk"
    LLM_RESPONSE_COMPLETED = "llm.response.completed"
    LLM_RESPONSE_FAILED = "llm.response.failed"

    # TTS events
    TTS_SENTENCE_READY = "tts.sentence.ready"
    TTS_AUDIO_GENERATED = "tts.audio.generated"
    TTS_COMPLETED = "tts.completed"
    TTS_FAILED = "tts.failed"

    # Session events
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    SESSION_ERROR = "session.error"


@dataclass
class Event:
    """
    Voice processing event

    Immutable event object passed through the PubSub system.
    """
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str | None = None

    def model_copy(self) -> 'Event':
        """Create a copy of this event (for safe distribution)"""
        return Event(
            type=self.type,
            data=self.data.copy(),
            timestamp=self.timestamp,
            session_id=self.session_id
        )


class EventPubSub:
    """
    Event-driven publish-subscribe system

    Based on Speaches.ai implementation:
    - Uses async Queue for zero-latency event distribution
    - Blocking wait with await queue.get() (NOT polling)
    - Multiple subscribers supported
    - Thread-safe publish_nowait()

    Usage:
        pubsub = EventPubSub()

        # Subscribe to events
        async for event in pubsub.poll():
            if event.type == EventType.SPEECH_STOPPED:
                # Handle event
                pass

        # Publish events
        pubsub.publish_nowait(Event(
            type=EventType.SPEECH_STARTED,
            data={"timestamp": time.time()}
        ))
    """

    def __init__(self):
        self.subscribers: set[asyncio.Queue[Event]] = set()
        self._lock = asyncio.Lock()
        self.event_history: list[Event] = []  # For debugging
        self.max_history = 1000
        self._loop: asyncio.AbstractEventLoop | None = None  # Event loop reference for thread-safe publishing

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Set event loop for thread-safe publishing
        
        Must be called from the event loop's thread.
        """
        self._loop = loop

    def publish_nowait(self, event: Event) -> None:
        """
        Publish event to all subscribers instantly (THREAD-SAFE)

        Can be called from any thread. If event loop is set and we're in a different thread,
        uses call_soon_threadsafe() for thread-safe delivery.

        Args:
            event: Event to publish
        """
        # Store in history for debugging
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]

        # Distribute to all subscribers
        for subscriber in self.subscribers:
            try:
                # Check if we need thread-safe call
                if self._loop is not None:
                    # Use call_soon_threadsafe to safely put item from any thread
                    self._loop.call_soon_threadsafe(subscriber.put_nowait, event.model_copy())
                else:
                    # No loop set, use direct put (legacy behavior)
                    subscriber.put_nowait(event.model_copy())
            except (asyncio.QueueFull, RuntimeError) as e:
                # If queue is full or runtime error, skip this subscriber
                pass

    async def poll(self) -> AsyncGenerator[Event, None]:
        """
        Poll for events (async generator)

        Blocks on await queue.get() - NO polling overhead.
        When event is published, all waiting pollers wake up instantly.

        Yields:
            Event objects as they arrive
        """
        # Create subscriber queue
        subscriber_queue: asyncio.Queue[Event] = asyncio.Queue()

        async with self._lock:
            self.subscribers.add(subscriber_queue)

        try:
            while True:
                # BLOCKS until event arrives - zero polling overhead!
                event = await subscriber_queue.get()
                yield event
        finally:
            # Cleanup on exit
            async with self._lock:
                self.subscribers.discard(subscriber_queue)

    def get_recent_events(self, count: int = 100) -> list[Event]:
        """Get recent events for debugging"""
        return self.event_history[-count:]

    def clear_history(self) -> None:
        """Clear event history"""
        self.event_history.clear()

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers"""
        return len(self.subscribers)

    def get_subscriber_depths(self) -> list[int]:
        """Return queue sizes for each subscriber (for diagnostics)."""

        return [queue.qsize() for queue in list(self.subscribers)]
