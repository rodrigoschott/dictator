"""
LLM Caller for Voice Assistant

Single-call pattern to Claude Code via MCP.
NO continuous polling - called once per user utterance.

Based on Speaches.ai pattern: LLM is ONLY for conversation content,
not for voice loop management.
"""

import asyncio
import json
import re
from typing import Optional
from pathlib import Path

from .events import Event, EventType, EventPubSub
from .sentence_chunker import MarkdownCleaner


def remove_thinking_tags(text: str) -> str:
    """
    Remove <think>...</think> tags and their content from LLM response.
    
    Many thinking models (like Qwen, DeepSeek-R1) expose their reasoning
    process using <think> tags. We don't want to speak this internal monologue.
    
    Examples:
        Input: "<think>Let me analyze this...</think>The answer is 42."
        Output: "The answer is 42."
        
        Input: "Here's what I found <think>hmm...</think> after searching."
        Output: "Here's what I found  after searching."
    """
    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace/newlines left behind
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)  # Multiple blank lines -> double newline
    cleaned = cleaned.strip()
    
    return cleaned


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

        # ADDED - Phase 3: Call serialization
        self._call_lock = asyncio.Lock()  # Prevent concurrent LLM calls
        self._processing = False  # Flag to track if currently processing

        # ADDED - Phase 4: Interrupt handling
        self._current_task: asyncio.Task | None = None  # Track current LLM task for cancellation
        self._cancel_requested = False  # Flag to signal cancellation

    def cancel_current_call(self) -> None:
        """
        Cancel the current LLM call if one is in progress

        ADDED - Phase 4: Allow interrupting LLM calls when user presses hotkey.
        """
        import logging
        logger = logging.getLogger("DictatorService")

        if self._current_task and not self._current_task.done():
            logger.info("[STOP] Cancelling current LLM call...")
            self._cancel_requested = True
            self._current_task.cancel()
        else:
            logger.debug("No active LLM call to cancel")

    async def process_transcription(self, transcription: str) -> None:
        """
        Process user transcription and get LLM response

        This is the ONLY LLM call per utterance.
        Calls Claude Code CLI directly (subprocess).

        ADDED - Phase 3: Uses lock to serialize concurrent calls.
        If already processing, the new call waits in queue.

        ADDED - Phase 4: Cancels previous call if one is in progress.

        Args:
            transcription: User's transcribed speech
        """
        import logging
        logger = logging.getLogger("DictatorService")

        # ADDED - Phase 4: Cancel previous call if one is in progress
        if self._processing:
            logger.warning(f"LLM call already in progress, cancelling it for new request: '{transcription[:50]}...'")
            self.cancel_current_call()

        # ADDED - Phase 3: Acquire lock (blocks if another call is in progress)
        async with self._call_lock:
            self._processing = True
            self._cancel_requested = False  # Reset cancel flag
            logger.debug(f" LLM lock acquired for: '{transcription[:50]}...'")

            try:
                # Create task for this call (for cancellation tracking)
                self._current_task = asyncio.create_task(
                    self._process_transcription_internal(transcription)
                )
                await self._current_task
            except asyncio.CancelledError:
                logger.info("[CANCELLED] LLM call was cancelled")
                # Emit cancellation event
                self.pubsub.publish_nowait(Event(
                    type=EventType.LLM_RESPONSE_FAILED,
                    data={"error": "Cancelled by user"},
                    session_id=self.session_id
                ))
                raise
            finally:
                self._processing = False
                self._current_task = None
                logger.debug(" LLM lock released")

    async def _process_transcription_internal(self, transcription: str) -> None:
        """Internal implementation of transcription processing (called within lock)"""
        import logging
        logger = logging.getLogger("DictatorService")

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
            logger.info(" About to call _call_claude_cli...")
            response = await self._call_claude_cli(transcription)
            logger.info(f"🟢 _call_claude_cli returned: {len(response)} chars")

            # ADDED - Phase 4: Check if cancelled after LLM call
            if self._cancel_requested:
                logger.info("[CANCELLED] Cancellation detected after LLM call, aborting...")
                raise asyncio.CancelledError()

            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            logger.info(f" Response text: {response[:200]}...")

            # Remove thinking tags first (for thinking models like Qwen, DeepSeek-R1)
            response = remove_thinking_tags(response)

            # Clean entire response for TTS (single shot)
            clean_response = MarkdownCleaner.clean(response)
            logger.info(f"[CLEANUP] Cleaned response: {clean_response[:100]}...")

            # ADDED - Phase 4: Check if cancelled before emitting TTS
            if self._cancel_requested:
                logger.info("[CANCELLED] Cancellation detected before TTS, aborting...")
                raise asyncio.CancelledError()

            # Emit single TTS event
            logger.info(f" Publishing TTS_SENTENCE_READY event...")
            self.pubsub.publish_nowait(Event(
                type=EventType.TTS_SENTENCE_READY,
                data={"text": clean_response},
                session_id=self.session_id
            ))

            # Emit completion
            logger.info(" Publishing LLM_RESPONSE_COMPLETED event")
            self.pubsub.publish_nowait(Event(
                type=EventType.LLM_RESPONSE_COMPLETED,
                data={},
                session_id=self.session_id
            ))

        except asyncio.CancelledError:
            # Re-raise to propagate cancellation
            logger.info("[CANCELLED] LLM processing cancelled")
            raise
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
            logger.error("Claude Code CLI not found in PATH or common locations")
            raise RuntimeError(
                "Claude Code CLI not found. Please install from: "
                "https://docs.anthropic.com/en/docs/claude-code\n"
                "Or ensure 'claude' is in your PATH"
            )
        
        logger.info(f" Using Claude CLI: {claude_cmd}")

        # Use a queue to get result from thread
        result_queue: queue.Queue = queue.Queue()

        def _run_in_thread():
            """Run Claude CLI in dedicated thread (not asyncio executor)"""
            try:
                logger.info(f"🤖 [Thread {threading.current_thread().name}] Calling Claude CLI")
                logger.info(f"[NOTE] Prompt length: {len(context)} chars")
                logger.info(f"[CONFIG] Command: {claude_cmd} --print --output-format text")
                logger.info(f" First 200 chars of prompt: {context[:200]}...")
                
                # Run in project directory (important for Claude CLI context)
                import os
                cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                logger.info(f" Working directory: {cwd}")

                result = subprocess.run(
                    [claude_cmd, '--print', '--output-format', 'text'],
                    input=context.encode('utf-8'),
                    capture_output=True,
                    timeout=30.0,
                    cwd=cwd  # Run in project root
                )

                logger.info(f"Subprocess completed with code: {result.returncode}")

                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='replace').strip()
                    stdout_msg = result.stdout.decode('utf-8', errors='replace').strip()
                    logger.error(f"Claude CLI stderr: {error_msg}")
                    logger.error(f"Claude CLI stdout: {stdout_msg}")
                    
                    # Use stdout if stderr is empty (Claude might output error to stdout)
                    full_error = error_msg or stdout_msg or "Unknown error (no output)"
                    result_queue.put(('error', f"Claude CLI failed (exit {result.returncode}): {full_error}"))
                    return

                response = result.stdout.decode('utf-8', errors='replace').strip()
                logger.info(f"🤖 Response length: {len(response)} chars")
                result_queue.put(('success', response))

            except subprocess.TimeoutExpired:
                logger.error("Claude CLI timed out")
                result_queue.put(('error', "Claude CLI timed out after 30 seconds"))
            except Exception as e:
                logger.error(f"Exception in thread: {e}")
                result_queue.put(('error', str(e)))

        # Start thread
        logger.info("Starting dedicated thread for subprocess...")
        thread = threading.Thread(target=_run_in_thread, daemon=True, name="ClaudeCLI")
        thread.start()

        # Wait for result with timeout (non-blocking for asyncio)
        import time
        timeout = 35.0  # Slightly longer than subprocess timeout
        start_time = time.time()

        while thread.is_alive():
            if time.time() - start_time > timeout:
                logger.error("Thread timeout")
                raise RuntimeError("Claude CLI thread timed out")

            # Sleep briefly to avoid busy-waiting (yield to asyncio)
            await asyncio.sleep(0.1)

        # Get result from queue
        try:
            status, data = result_queue.get_nowait()
            if status == 'error':
                raise RuntimeError(data)
            logger.info("Got response from thread")
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

    async def _get_response(self) -> str:
        """
        Get LLM response directly from Anthropic API

        Returns:
            Complete response text
        """
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

        # Get response from Claude
        full_response = ""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        ) as stream:
            async for text in stream.text_stream:
                full_response += text

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

        return full_response


class OllamaLLMCaller(LLMCaller):
    """
    Ollama LLM caller for local model inference

    Calls Ollama API (running in Docker) for unlimited local testing.
    Supports streaming and conversation history.

    Advantages:
    - Free, unlimited usage (no token limits)
    - Low latency (local inference)
    - Privacy (no cloud API calls)

    Requirements:
    - Ollama container running (docker-compose up ollama)
    - Model pulled (e.g., llama3.2:latest)
    """

    def __init__(
        self,
        pubsub: EventPubSub,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:latest",
        session_id: str | None = None
    ):
        # Initialize without MCP files (not needed for Ollama)
        super().__init__(
            pubsub=pubsub,
            mcp_request_file=Path("/dev/null"),
            mcp_response_file=Path("/dev/null"),
            session_id=session_id
        )

        self.base_url = base_url
        self.model = model

    async def _process_transcription_internal(self, transcription: str) -> None:
        """
        Process user transcription with Ollama (internal implementation)

        Overrides parent method to use Ollama API instead of Claude CLI.

        Args:
            transcription: User's transcribed speech
        """
        import logging
        logger = logging.getLogger("DictatorService")

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
            logger.info(f"🦙 Calling Ollama API ({self.model})...")

            # Call Ollama API
            response = await self._call_ollama_api(transcription)
            logger.info(f"🟢 Ollama returned: {len(response)} chars")

            # ADDED - Phase 4: Check if cancelled after LLM call
            if self._cancel_requested:
                logger.info("[CANCELLED] Cancellation detected after Ollama call, aborting...")
                raise asyncio.CancelledError()

            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            logger.info(f" Response text: {response[:200]}...")

            # Remove thinking tags first (for thinking models like Qwen, DeepSeek-R1)
            response = remove_thinking_tags(response)

            # Clean response for TTS
            clean_response = MarkdownCleaner.clean(response)
            logger.info(f"[CLEANUP] Cleaned response: {clean_response[:100]}...")

            # ADDED - Phase 4: Check if cancelled before emitting TTS
            if self._cancel_requested:
                logger.info("[CANCELLED] Cancellation detected before TTS, aborting...")
                raise asyncio.CancelledError()

            # Emit TTS event
            logger.info(f" Publishing TTS_SENTENCE_READY event...")
            self.pubsub.publish_nowait(Event(
                type=EventType.TTS_SENTENCE_READY,
                data={"text": clean_response},
                session_id=self.session_id
            ))

            # Emit completion
            logger.info(" Publishing LLM_RESPONSE_COMPLETED event")
            self.pubsub.publish_nowait(Event(
                type=EventType.LLM_RESPONSE_COMPLETED,
                data={},
                session_id=self.session_id
            ))

        except asyncio.CancelledError:
            # Re-raise to propagate cancellation
            logger.info("[CANCELLED] Ollama processing cancelled")
            raise
        except Exception as e:
            logger = logging.getLogger("DictatorService")
            logger.error(f"Ollama API error: {e}")
            # Emit failure event
            self.pubsub.publish_nowait(Event(
                type=EventType.LLM_RESPONSE_FAILED,
                data={"error": str(e)},
                session_id=self.session_id
            ))
            raise

    async def _call_ollama_api(self, transcription: str) -> str:
        """
        Call Ollama API with conversation context

        Uses /api/chat endpoint for conversational interaction.

        Args:
            transcription: User's current message

        Returns:
            Complete response from Ollama
        """
        import aiohttp
        import logging

        logger = logging.getLogger("DictatorService")

        # Build system prompt optimized for TTS
        system_prompt = """You are Dictator, a concise voice assistant integrated into a Windows dictation system.

IMPORTANT - Your responses will be read aloud via text-to-speech:
- Keep answers under 3 sentences maximum
- Be direct, natural, and conversational
- Avoid markdown formatting, code blocks, or complex punctuation
- Don't use asterisks, backticks, or special symbols
- Speak as if talking to the user face-to-face
- Match the user's language automatically (Portuguese or English)

Provide helpful, accurate information in a friendly, spoken style."""

        # Build messages with history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history
        recent_history = self.conversation_history[-(self.max_history):]
        messages.extend([
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent_history
        ])

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # Get complete response at once
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }

        logger.info(f" Calling Ollama: {self.base_url}/api/chat")
        logger.info(f"[PACKAGE] Payload: {len(messages)} messages, model={self.model}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60.0)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Ollama API error (HTTP {response.status}): {error_text}"
                        )

                    # Parse JSON response
                    result = await response.json()
                    
                    # Extract message content
                    if "message" not in result or "content" not in result["message"]:
                        raise RuntimeError(f"Unexpected Ollama response format: {result}")

                    response_text = result["message"]["content"].strip()
                    logger.info(f"Ollama response: {len(response_text)} chars")

                    return response_text

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling Ollama: {e}")
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {e}")
        except asyncio.TimeoutError:
            logger.error("Ollama API timeout")
            raise RuntimeError("Ollama API timed out after 60 seconds")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


class N8NToolCallingLLMCaller(LLMCaller):
    """
    N8N Tool-Calling LLM caller

    Calls N8N webhook which orchestrates:
    - Ollama LLM processing
    - Tool/function execution (weather, calendar, APIs, etc.)
    - Multi-step reasoning with tool results
    - Response generation

    Advantages:
    - Centralized tool orchestration via N8N visual workflows
    - Easy tool addition/modification without code changes
    - Debug tool calls visually in N8N UI
    - Reusable workflows across different contexts
    - Separates conversation logic from tool execution

    Requirements:
    - N8N instance running (e.g., http://localhost:15678)
    - Dictator workflow imported in N8N
    - Ollama container available to N8N
    """

    def __init__(
        self,
        pubsub: EventPubSub,
        webhook_url: str,
        timeout: int = 120,
        session_id: str | None = None
    ):
        """
        Initialize N8N caller

        Args:
            pubsub: Event pub/sub system
            webhook_url: N8N webhook URL (e.g., http://localhost:15678/webhook/dictator-llm)
            timeout: Request timeout in seconds (default: 120s for tool execution)
            session_id: Session identifier
        """
        # Initialize without MCP files (not needed for N8N)
        super().__init__(
            pubsub=pubsub,
            mcp_request_file=Path("/dev/null"),
            mcp_response_file=Path("/dev/null"),
            session_id=session_id
        )

        self.webhook_url = webhook_url
        self.timeout = timeout

    async def _process_transcription_internal(self, transcription: str) -> None:
        """
        Process user transcription with N8N orchestration (internal implementation)

        Overrides parent method to use N8N webhook instead of direct LLM calls.
        N8N handles the complete flow: LLM → tool calls → tool execution → final response.

        Args:
            transcription: User's transcribed speech
        """
        import logging
        logger = logging.getLogger("DictatorService")

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
            logger.info(f" Calling N8N webhook: {self.webhook_url}")

            # Call N8N webhook
            response = await self._call_n8n_webhook(transcription)
            logger.info(f"🟢 N8N returned: {len(response)} chars")

            # ADDED - Phase 4: Check if cancelled after N8N call
            if self._cancel_requested:
                logger.info("[CANCELLED] Cancellation detected after N8N call, aborting...")
                raise asyncio.CancelledError()

            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            logger.info(f" Response text: {response[:200]}...")

            # Remove thinking tags first (for thinking models)
            response = remove_thinking_tags(response)

            # Clean response for TTS
            clean_response = MarkdownCleaner.clean(response)
            logger.info(f"[CLEANUP] Cleaned response: {clean_response[:100]}...")

            # ADDED - Phase 4: Check if cancelled before emitting TTS
            if self._cancel_requested:
                logger.info("[CANCELLED] Cancellation detected before TTS, aborting...")
                raise asyncio.CancelledError()

            # Emit TTS event
            logger.info(f" Publishing TTS_SENTENCE_READY event...")
            self.pubsub.publish_nowait(Event(
                type=EventType.TTS_SENTENCE_READY,
                data={"text": clean_response},
                session_id=self.session_id
            ))

            # Emit completion
            logger.info(" Publishing LLM_RESPONSE_COMPLETED event")
            self.pubsub.publish_nowait(Event(
                type=EventType.LLM_RESPONSE_COMPLETED,
                data={},
                session_id=self.session_id
            ))

        except asyncio.CancelledError:
            # Re-raise to propagate cancellation
            logger.info("[CANCELLED] N8N processing cancelled")
            raise
        except Exception as e:
            logger = logging.getLogger("DictatorService")
            logger.error(f"N8N webhook error: {e}")
            # Emit failure event
            self.pubsub.publish_nowait(Event(
                type=EventType.LLM_RESPONSE_FAILED,
                data={"error": str(e)},
                session_id=self.session_id
            ))
            raise

    async def _call_n8n_webhook(self, transcription: str) -> str:
        """
        Call N8N webhook with conversation context

        N8N workflow receives full conversation history and orchestrates:
        1. LLM analyzes request and identifies needed tools
        2. N8N executes tools (API calls, DB queries, scripts, etc.)
        3. LLM generates final response with tool results
        4. Response returned to Dictator

        Args:
            transcription: User's current message

        Returns:
            Complete response from N8N (after tool execution and LLM processing)
        """
        import aiohttp
        import logging
        from datetime import datetime

        logger = logging.getLogger("DictatorService")

        # Build system prompt (same as Ollama for consistency)
        system_prompt = """You are Dictator, a concise voice assistant integrated into a Windows dictation system.

IMPORTANT - Your responses will be read aloud via text-to-speech:
- Keep answers under 3 sentences maximum
- Be direct, natural, and conversational
- Avoid markdown formatting, code blocks, or complex punctuation
- Don't use asterisks, backticks, or special symbols
- Speak as if talking to the user face-to-face
- Match the user's language automatically (Portuguese or English)

You have access to tools for external information. Use them when needed, but keep responses concise.
Provide helpful, accurate information in a friendly, spoken style."""

        # Build conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent history
        recent_history = self.conversation_history[-(self.max_history):]
        messages.extend([
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent_history
        ])

        # Prepare webhook payload
        payload = {
            "messages": messages,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id or "default"
        }

        logger.info(f" Calling N8N webhook: {self.webhook_url}")
        logger.info(f"[PACKAGE] Payload: {len(messages)} messages")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"N8N webhook error (HTTP {response.status}): {error_text}"
                        )

                    # Parse JSON response
                    result = await response.json()

                    # Extract response text
                    # Expected format: {"response": "text here", "tools_used": [...]}
                    if "response" not in result:
                        raise RuntimeError(f"Unexpected N8N response format: {result}")

                    response_text = result["response"].strip()

                    # Log tool usage if available
                    if "tools_used" in result and result["tools_used"]:
                        tools = ", ".join(result["tools_used"])
                        logger.info(f"[CONFIG] Tools used: {tools}")

                    logger.info(f"N8N response: {len(response_text)} chars")
                    return response_text

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling N8N: {e}")
            raise RuntimeError(f"Failed to connect to N8N at {self.webhook_url}: {e}")
        except asyncio.TimeoutError:
            logger.error(f"N8N webhook timeout ({self.timeout}s)")
            raise RuntimeError(f"N8N webhook timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
