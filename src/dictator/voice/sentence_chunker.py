"""
Sentence Chunker for Streaming TTS

Detects sentence boundaries in streaming text to enable incremental TTS.
Start synthesizing audio before LLM finishes generating full response.

Based on Speaches.ai pattern.
"""

import asyncio
import re
from typing import AsyncGenerator


class SentenceChunker:
    """
    Detects sentence boundaries in streaming text

    Allows TTS to start before LLM finishes generating the complete response.
    This dramatically reduces perceived latency.

    Usage:
        chunker = SentenceChunker()

        # Feed tokens from LLM stream
        async for token in llm_stream:
            await chunker.add_token(token)

        # Get complete sentences
        async for sentence in chunker:
            audio = await tts.synthesize(sentence)
            play(audio)

        # Don't forget to flush remaining text
        await chunker.flush()
    """

    # Sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]+(?:\s|$)')
    MIN_SENTENCE_LENGTH = 10  # Minimum chars to consider a sentence

    def __init__(self):
        self.buffer = ""
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._flushed = False

    async def add_token(self, token: str) -> None:
        """
        Add token from LLM stream

        Detects sentence boundaries and queues complete sentences.

        Args:
            token: Text token from LLM
        """
        self.buffer += token

        # Check for sentence boundary
        if self._has_sentence_boundary():
            await self._extract_sentences()

    async def flush(self) -> None:
        """
        Flush remaining buffer as final sentence

        Call this when LLM stream ends to get any remaining text.
        """
        if self.buffer.strip() and not self._flushed:
            await self.queue.put(self.buffer.strip())
            self.buffer = ""
            self._flushed = True

    def _has_sentence_boundary(self) -> bool:
        """Check if buffer contains a sentence boundary"""
        return bool(self.SENTENCE_ENDINGS.search(self.buffer))

    async def _extract_sentences(self) -> None:
        """Extract complete sentences from buffer"""
        # Find all sentence boundaries
        matches = list(self.SENTENCE_ENDINGS.finditer(self.buffer))

        if not matches:
            return

        # Process each sentence
        last_end = 0
        for match in matches:
            sentence_end = match.end()
            sentence = self.buffer[last_end:sentence_end].strip()

            # Only queue if long enough
            if len(sentence) >= self.MIN_SENTENCE_LENGTH:
                await self.queue.put(sentence)

            last_end = sentence_end

        # Keep remaining text in buffer
        self.buffer = self.buffer[last_end:]

    def __aiter__(self) -> AsyncGenerator[str, None]:
        """Async iterator for sentences"""
        return self._sentence_generator()

    async def _sentence_generator(self) -> AsyncGenerator[str, None]:
        """Generate sentences as they become available"""
        while True:
            try:
                # Wait for next sentence (with timeout)
                sentence = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.1
                )
                yield sentence
            except asyncio.TimeoutError:
                # Check if we should continue waiting
                if self._flushed and self.queue.empty():
                    # No more sentences coming
                    break
                # Otherwise keep waiting
                continue

    def reset(self) -> None:
        """Reset chunker state"""
        self.buffer = ""
        self._flushed = False
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class MarkdownCleaner:
    """
    Clean markdown formatting from text before TTS

    Based on Speaches.ai:
    - Removes **bold** and *italic* markers
    - Removes emojis
    - Preserves actual content
    """

    EMPHASIS_PATTERN = re.compile(r'(\*\*|__)(.*?)\1')  # **bold** or __bold__
    ITALIC_PATTERN = re.compile(r'(\*|_)(.*?)\1')  # *italic* or _italic_
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Clean text for TTS

        Args:
            text: Raw text (may contain markdown/emojis)

        Returns:
            Cleaned text suitable for TTS
        """
        # Remove markdown emphasis
        text = cls.EMPHASIS_PATTERN.sub(r'\2', text)
        text = cls.ITALIC_PATTERN.sub(r'\2', text)

        # Remove emojis
        text = cls.EMOJI_PATTERN.sub('', text)

        # Clean up whitespace
        text = ' '.join(text.split())

        return text
