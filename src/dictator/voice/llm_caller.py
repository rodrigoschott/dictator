"""
LLM Caller for Voice Assistant

Single-call pattern to Claude Code via MCP.
NO continuous polling - called once per user utterance.

Based on Speaches.ai pattern: LLM is ONLY for conversation content,
not for voice loop management.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional
from pathlib import Path

from .events import Event, EventType, EventPubSub
from .sentence_chunker import SentenceChunker, MarkdownCleaner


class LLMCaller:
    """
    Single-call LLM integration

    Calls Claude Code ONCE per user utterance (via MCP tool).
    Streams response and emits events for sentence-based TTS.

    Key difference from old approach:
    - OLD: Continuous polling loop in Claude Code (wastes tokens)
    - NEW: Single call per transcription (efficient)
    """

    def __init__(
        self,
        pubsub: EventPubSub,
        mcp_request_file: Path,
        mcp_response_file: Path,
        session_id: str | None = None
    ):
        self.pubsub = pubsub
        self.mcp_request_file = mcp_request_file
        self.mcp_response_file = mcp_response_file
        self.session_id = session_id

        # Conversation history
        self.conversation_history: list[dict] = []
        self.max_history = 10

    async def process_transcription(self, transcription: str) -> None:
        """
        Process user transcription and get LLM response

        This is the ONLY LLM call per utterance.
        Calls Claude Code CLI directly (subprocess).

        Args:
            transcription: User's transcribed speech
        """
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": transcription
        })

        # Emit LLM request event
        self.pubsub.publish_nowait(Event(
            type=EventType.LLM_REQUEST,
            data={"transcription": transcription},
            session_id=self.session_id
        ))

        try:
            # Call Claude Code CLI directly (subprocess)
            import logging
            logger = logging.getLogger("DictatorService")
            logger.info("🔵 About to call _call_claude_cli...")
            response = await self._call_claude_cli(transcription)
            logger.info(f"🟢 _call_claude_cli returned: {len(response)} chars")

            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            logger.info(f"📄 Response text: {response[:200]}...")
            logger.info("🔄 Splitting response into sentences...")

            # Stream response sentences for TTS
            sentence_count = 0
            async for sentence in self._split_into_sentences(response):
                sentence_count += 1
                logger.info(f"📝 Sentence {sentence_count}: {sentence[:100]}...")

                # Clean text for TTS
                clean_sentence = MarkdownCleaner.clean(sentence)
                logger.info(f"🧹 Cleaned sentence: {clean_sentence[:100]}...")

                # Emit TTS event
                logger.info(f"📢 Publishing TTS_SENTENCE_READY event...")
                self.pubsub.publish_nowait(Event(
                    type=EventType.TTS_SENTENCE_READY,
                    data={"text": clean_sentence},
                    session_id=self.session_id
                ))

            logger.info(f"✅ Finished splitting ({sentence_count} sentences)")

            # Emit completion
            logger.info("📢 Publishing LLM_RESPONSE_COMPLETED event")
            self.pubsub.publish_nowait(Event(
                type=EventType.LLM_RESPONSE_COMPLETED,
                data={},
                session_id=self.session_id
            ))

        except Exception as e:
            # Emit failure event
            self.pubsub.publish_nowait(Event(
                type=EventType.LLM_RESPONSE_FAILED,
                data={"error": str(e)},
                session_id=self.session_id
            ))
            raise

    async def _call_claude_cli(self, transcription: str) -> str:
        """
        Call Claude Code CLI directly via subprocess

        Uses local Claude Code installation - zero API calls, zero tokens!

        NOTE: Runs subprocess.run in a dedicated daemon thread to avoid issues
        with Windows asyncio loops. Similar to how Speaches handles blocking I/O.

        Args:
            transcription: User's speech

        Returns:
            Claude's response text
        """
        import subprocess
        import logging
        import threading
        import queue

        logger = logging.getLogger("DictatorService")

        # Build conversation context for Claude
        context = self._build_conversation_context(transcription)

        # Find Claude Code CLI executable
        claude_cmd = self._find_claude_executable()
        if not claude_cmd:
            logger.error("❌ Claude Code CLI not found")
            raise RuntimeError(
                "Claude Code CLI not found. Please install from: "
                "https://docs.anthropic.com/en/docs/claude-code\n"
                "Or ensure 'claude' is in your PATH"
            )

        # Use a queue to get result from thread
        result_queue: queue.Queue = queue.Queue()

        def _run_in_thread():
            """Run Claude CLI in dedicated thread (not asyncio executor)"""
            try:
                logger.info(f"🤖 [Thread {threading.current_thread().name}] Calling Claude CLI")
                logger.info(f"📝 Prompt length: {len(context)} chars")

                result = subprocess.run(
                    [claude_cmd, '--print', '--output-format', 'text'],
                    input=context.encode('utf-8'),
                    capture_output=True,
                    timeout=30.0,
                )

                logger.info(f"✅ Subprocess completed with code: {result.returncode}")

                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='replace')
                    logger.error(f"❌ Claude CLI error: {error_msg}")
                    result_queue.put(('error', f"Claude CLI failed: {error_msg}"))
                    return

                response = result.stdout.decode('utf-8', errors='replace').strip()
                logger.info(f"🤖 Response length: {len(response)} chars")
                result_queue.put(('success', response))

            except subprocess.TimeoutExpired:
                logger.error("❌ Claude CLI timed out")
                result_queue.put(('error', "Claude CLI timed out after 30 seconds"))
            except Exception as e:
                logger.error(f"❌ Exception in thread: {e}")
                result_queue.put(('error', str(e)))

        # Start thread
        logger.info("🚀 Starting dedicated thread for subprocess...")
        thread = threading.Thread(target=_run_in_thread, daemon=True, name="ClaudeCLI")
        thread.start()

        # Wait for result with timeout (non-blocking for asyncio)
        import time
        timeout = 35.0  # Slightly longer than subprocess timeout
        start_time = time.time()

        while thread.is_alive():
            if time.time() - start_time > timeout:
                logger.error("❌ Thread timeout")
                raise RuntimeError("Claude CLI thread timed out")

            # Sleep briefly to avoid busy-waiting (yield to asyncio)
            await asyncio.sleep(0.1)

        # Get result from queue
        try:
            status, data = result_queue.get_nowait()
            if status == 'error':
                raise RuntimeError(data)
            logger.info("✅ Got response from thread")
            return data
        except queue.Empty:
            raise RuntimeError("Thread completed but no result in queue")

    def _find_claude_executable(self) -> Optional[str]:
        """
        Find Claude Code CLI executable

        Searches in PATH and common Windows installation locations.

        Returns:
            Path to claude executable, or None if not found
        """
        import shutil
        import os
        from pathlib import Path

        # 1. Check if 'claude' is in PATH
        claude_path = shutil.which('claude')
        if claude_path:
            return claude_path

        # 2. Check common Windows installation paths
        common_paths = [
            # NPM global install
            Path(os.environ.get('APPDATA', '')) / 'npm' / 'claude.cmd',
            Path(os.environ.get('APPDATA', '')) / 'npm' / 'claude.exe',
            # User local bin
            Path.home() / '.local' / 'bin' / 'claude.exe',
            Path.home() / '.local' / 'bin' / 'claude',
            # Windows Program Files
            Path('C:/Program Files/Claude/claude.exe'),
            Path('C:/Program Files (x86)/Claude/claude.exe'),
        ]

        for path in common_paths:
            if path.exists():
                return str(path)

        return None

    def _build_conversation_context(self, transcription: str) -> str:
        """
        Build conversation context with system prompt and history

        Args:
            transcription: Current user message

        Returns:
            Formatted prompt for Claude
        """
        # System prompt optimized for voice TTS
        system_prompt = """You are Dictator, a concise voice assistant integrated into a Windows dictation system.

IMPORTANT - Your responses will be read aloud via text-to-speech:
- Keep answers under 3 sentences maximum
- Be direct, natural, and conversational
- Avoid markdown formatting, code blocks, or complex punctuation
- Don't use asterisks, backticks, or special symbols
- Speak as if talking to the user face-to-face
- Match the user's language automatically (Portuguese or English)

Provide helpful, accurate information in a friendly, spoken style."""

        # Build prompt with history
        prompt_parts = [system_prompt, "\n\n"]

        # Add recent conversation history (last 5 exchanges)
        recent_history = self.conversation_history[-(self.max_history):]
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}\n")

        # Add current transcription
        prompt_parts.append(f"\nUser: {transcription}\nAssistant:")

        return "".join(prompt_parts)

    async def _split_into_sentences(self, response: str) -> AsyncGenerator[str, None]:
        """
        Split response into sentences for streaming TTS

        Args:
            response: Full response text

        Yields:
            Individual sentences
        """
        chunker = SentenceChunker()

        # Feed response character by character
        for char in response:
            await chunker.add_token(char)

        # Flush remaining content
        await chunker.flush()

        # Collect all sentences from queue
        sentences = []
        while not chunker.queue.empty():
            try:
                sentence = chunker.queue.get_nowait()
                sentences.append(sentence)
            except asyncio.QueueEmpty:
                break

        # Yield sentences
        for sentence in sentences:
            yield sentence

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()

    def get_history(self) -> list[dict]:
        """Get conversation history"""
        return self.conversation_history.copy()


class DirectLLMCaller(LLMCaller):
    """
    Direct LLM caller using Anthropic SDK

    For cases where MCP is not available or for testing.
    Uses streaming API for real-time sentence generation.
    """

    def __init__(
        self,
        pubsub: EventPubSub,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        session_id: str | None = None
    ):
        # Initialize without MCP files
        super().__init__(
            pubsub=pubsub,
            mcp_request_file=Path("/dev/null"),
            mcp_response_file=Path("/dev/null"),
            session_id=session_id
        )

        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy load Anthropic client"""
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def _stream_response(self) -> AsyncGenerator[str, None]:
        """
        Stream LLM response directly from Anthropic API

        Yields:
            Complete sentences ready for TTS
        """
        chunker = SentenceChunker()

        # Prepare messages
        messages = [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in self.conversation_history[-self.max_history:]
        ]

        # System prompt for voice assistant
        system_prompt = (
            "You are a helpful voice assistant. "
            "Provide concise, conversational responses (2-3 sentences max). "
            "Speak naturally as if in a conversation. "
            "Match the user's language automatically."
        )

        # Stream from Claude
        full_response = ""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        ) as stream:
            async for text in stream.text_stream:
                full_response += text
                await chunker.add_token(text)

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

        # Flush remaining
        await chunker.flush()

        # Yield sentences
        async for sentence in chunker:
            yield sentence
