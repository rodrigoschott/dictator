"""
Dictator Voice Assistant Module

Event-driven voice interaction system with local processing.
Based on Speaches.ai architecture - zero polling, 100% event-driven.
"""

from .events import Event, EventType, EventPubSub
from .vad_processor import VADProcessor, VADResult, VADConfig
from .session_manager import VoiceSessionManager
from .llm_caller import LLMCaller, DirectLLMCaller, OllamaLLMCaller
from .sentence_chunker import SentenceChunker, MarkdownCleaner

__all__ = [
    'Event',
    'EventType',
    'EventPubSub',
    'VADProcessor',
    'VADResult',
    'VADConfig',
    'VoiceSessionManager',
    'LLMCaller',
    'DirectLLMCaller',
    'OllamaLLMCaller',
    'SentenceChunker',
    'MarkdownCleaner',
]
