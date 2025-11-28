# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dictator** is a Windows voice-to-text service using faster-whisper (optimized Whisper AI) with local GPU acceleration. It runs as a Windows Service with system tray control, providing push-to-talk/toggle recording triggered by mouse buttons or keyboard hotkeys.

Key features:
- 100% local processing (no external APIs required, privacy-focused)
- GPU-accelerated transcription with faster-whisper + CTranslate2
- Local TTS via Kokoro-ONNX (56 voices, 9 languages)
- Event-driven voice assistant mode with LLM integration (Ollama, Claude Direct, N8N Tool-Calling)
- VAD (Voice Activity Detection) for auto-stop on silence
- Visual overlay for recording/processing feedback
- N8N workflow integration for AI agent capabilities with tool-calling

## Development Commands

### Environment Setup
```batch
# First-time setup (installs Poetry and all dependencies)
setup.bat

# Verify dependencies
poetry run python verify_deps.py
```

### Local Testing
```batch
# Run locally (normal mode - doesn't work in elevated apps due to UIPI)
run_local.bat

# Run as admin (works in all apps)
run_local_admin.bat  # Right-click → "Run as Administrator"

# Direct execution (for debugging)
poetry run python src/dictator/tray.py config.yaml
poetry run python src/dictator/service.py config.yaml  # Service only, no tray
```

### Windows Service Management
```batch
# Install as Windows Service (run as Administrator)
install_service.bat

# Restart service
restart_dictator.bat

# Uninstall service (run as Administrator)
uninstall_service.bat

# Manual service commands
nssm start Dictator
nssm stop Dictator
nssm restart Dictator
sc query Dictator
```

### Testing
```batch
# Test Portuguese TTS voices
poetry run python tests/manual/test_portuguese_voices.py

# Run unit tests
poetry run pytest tests/unit/

# Run integration tests (requires Ollama running)
poetry run pytest tests/integration/

# Manual tests
poetry run python tests/manual/test_claude_cli.py
poetry run python tests/manual/test_menu_callbacks.py
```

### Poetry Commands
```batch
# Install dependencies
poetry install

# Update dependencies
poetry update

# Add new dependency
poetry add package-name

# Clear cache
poetry cache clear . --all
```

## Architecture Overview

### Core Components

**Entry Points:**
- `src/dictator/tray.py` - System tray GUI + service orchestration (main entry point for Windows Service)
- `src/dictator/service.py` - Core service: recording, transcription, state management
- `src/dictator/main.py` - Original standalone script (legacy)

**Audio Processing:**
- `src/dictator/service.py` - Audio recording with sounddevice, Whisper transcription with faster-whisper
- `src/dictator/tts_engine.py` - Text-to-Speech using Kokoro-ONNX with interrupt support

**Voice Assistant (Event-Driven Architecture):**
- `src/dictator/voice/events.py` - Zero-polling PubSub system (based on Speaches.ai)
- `src/dictator/voice/session_manager.py` - Orchestrates VAD → STT → LLM → TTS pipeline
- `src/dictator/voice/vad_processor.py` - Silero VAD for speech detection
- `src/dictator/voice/llm_caller.py` - LLM integration (Claude CLI, Direct API, Ollama, N8N)
- `src/dictator/voice/sentence_chunker.py` - Real-time sentence chunking for streaming TTS

**UI & Monitoring:**
- `src/dictator/overlay.py` - Visual feedback overlay (red=recording, orange=processing, green=speaking)
- `src/dictator/logging_setup.py` - Centralized logging with structured logs and run directories
- `src/dictator/monitoring/thread_monitor.py` - Thread health monitoring and deadlock detection

### Execution Flow

**Standard Dictation Mode:**
1. `tray.py` starts `service.py` in background thread
2. `service.py` listens for mouse button or keyboard hotkey
3. On trigger: records audio → transcribes with Whisper → pastes to clipboard
4. `overlay.py` shows visual state (recording/processing)

**Voice Assistant Mode (claude_mode: true):**
1. Event-driven architecture: audio chunks → VAD → STT → LLM → sentence chunking → TTS
2. `VoiceSessionManager` orchestrates all components via `EventPubSub`
3. Zero-polling design: async queues with `await queue.get()` (NOT polling loops)
4. TTS interrupt: pressing hotkey during speech stops TTS instantly (~170ms latency)
5. Thinking models (Qwen3, DeepSeek-R1): `<think>` tags automatically filtered

**N8N Tool-Calling Integration:**
- N8N workflows in `n8n_workflows/` provide AI agent with web search and tool capabilities
- `N8NToolCallingLLMCaller` sends webhook requests to N8N workflow
- Supports streaming responses and thinking tag filtering
- Default webhook: `http://localhost:15678/webhook/dictator-llm`

### Configuration System

All settings in `config.yaml` (YAML format):

**Key Sections:**
- `whisper` - Model selection (tiny/base/small/medium/large/large-v3), language, device (cuda/cpu)
- `hotkey` - Trigger type (mouse/keyboard), button/key, mode (toggle/push_to_talk), VAD settings
- `tts` - Kokoro TTS config: voice selection, speed, volume, interrupt behavior
- `voice` - LLM provider selection (ollama/claude-direct/claude-cli/n8n_toolcalling), VAD config, streaming
- `logging` - Structured logging, run retention, thread tracing
- `overlay` - Visual indicator position and appearance

**LLM Providers:**
- `ollama` - Local models (automatically discovered in tray menu)
- `claude-direct` - Direct Anthropic API calls
- `claude-cli` - MCP-based Claude CLI integration
- `n8n_toolcalling` - N8N workflow with tool-calling support

**Auto-Restart Behavior:**
When changing LLM model/provider in tray menu, service automatically restarts to apply changes.

### Models & Dependencies

**Critical Files (Git LFS):**
- `kokoro-v1.0.onnx` (310 MB) - TTS model
- `voices-v1.0.bin` (43 MB) - TTS voices

**Tech Stack:**
- `faster-whisper` - Optimized Whisper (uses CTranslate2, NOT PyTorch)
- `kokoro-onnx[gpu]` - Local TTS with ONNX Runtime GPU support
- `pynput` - Global hotkey capture
- `pystray` - System tray integration
- `sounddevice/soundfile` - Audio recording
- `aiohttp` - Async HTTP for LLM APIs

## Important Implementation Details

### Event-Driven Voice Session
- Based on Speaches.ai architecture: zero-polling design
- `EventPubSub` uses `asyncio.Queue` with blocking `await queue.get()`
- Thread-safe publishing: `publish_nowait()` uses `call_soon_threadsafe()`
- Multiple subscribers supported, each gets copy of event
- Event types: AUDIO_CHUNK, SPEECH_STARTED/STOPPED, TRANSCRIPTION_*, LLM_*, TTS_*

### Logging & Monitoring
- Centralized logging via `logging_setup.bootstrap_logging()`
- Run directories: `logs/run-YYYYMMDD-HHMMSS/` with retention
- Structured logging: JSON lines (`.jsonl`) when enabled
- Thread monitor: periodic thread dumps to detect deadlocks/freezes
- Health checks: monitors event queue depths and subscriber counts

### VAD (Voice Activity Detection)
- Two modes: legacy VAD (in service.py) and Silero VAD (in voice session)
- Legacy VAD: RMS-based with adaptive threshold
- Silero VAD: ML-based speech detection with configurable thresholds
- Auto-stop on silence configurable (default: 2s silence)
- Event-driven mode: VAD emits SPEECH_STARTED/STOPPED events

### TTS Interrupt System
- User pressing hotkey during TTS → instant stop (~170ms latency)
- Uses `tts_engine.stop()` + threading event
- Voice session aware: TTS interrupt handled by event processor
- Config: `tts.interrupt_on_speech: true`

### Windows Service Integration
- NSSM (Non-Sucking Service Manager) wraps Python process
- Poetry environment activated automatically
- Logs to `logs/dictator.log` (configurable)
- Auto-start on Windows boot
- Admin elevation required for service installation

### Thinking Models Support
- Models like Qwen3 and DeepSeek-R1 expose `<think>` reasoning tags
- `llm_caller.py` filters thinking tags before TTS
- Configurable via provider implementation
- Ensures only final response is spoken, not internal reasoning

## Common Workflows

### Adding a New LLM Provider
1. Create subclass of `LLMCaller` in `src/dictator/voice/llm_caller.py`
2. Implement `call_llm()` method with async streaming
3. Add provider config to `config.yaml` under `voice.llm.provider`
4. Update `service.py` to instantiate new provider in `load_voice_session()`
5. Optionally add to tray menu in `tray.py`

### Testing New Whisper Models
1. Edit `config.yaml`: `whisper.model: "model-name"`
2. Run `run_local.bat` to test
3. Monitor GPU VRAM usage (check logs for CUDA info)
4. Available models: tiny, base, small, medium, large, large-v3

### Debugging Voice Session Issues
1. Enable debug logging: `logging.level: DEBUG` in config
2. Enable thread tracing: `logging.trace_threads: true`
3. Check event queue depths: look for "subscriber_count" in logs
4. Review event history: `pubsub.get_recent_events(100)`
5. Monitor thread health: `logs/run-*/thread_monitor.latest.json`

### Changing TTS Voice
1. Test voices: `poetry run python tests/manual/test_portuguese_voices.py`
2. Edit `config.yaml`: `tts.kokoro.voice: "voice-id"`
3. Restart service: `restart_dictator.bat`
4. Available voices: 56 voices in 9 languages (see README.md)

### Adding N8N Workflows
1. Import workflow JSON from `n8n_workflows/`
2. Configure webhook URL in `config.yaml`: `voice.llm.n8n_toolcalling.webhook_url`
3. Ensure N8N is running and webhook is accessible
4. Set provider: `voice.llm.provider: n8n_toolcalling`
5. Test with voice assistant mode enabled

## Platform & Environment

- **Windows 10/11 only** (uses Windows Service, NSSM, pynput Windows-specific features)
- **Python 3.10-3.13** (< 3.14)
- **NVIDIA GPU recommended** (CUDA) - works with RTX 5080 out-of-the-box
- **Poetry** for dependency management (installed automatically by setup.bat)
- **Git LFS** for large model files

## Configuration Files

- `config.yaml` - Main configuration (user-editable)
- `pyproject.toml` - Poetry dependencies and project metadata
- `.gitattributes` - Git LFS tracking for `*.onnx` and `*.bin` files
- `logs/` - Log files and run directories (auto-cleaned based on retention)

## Notes for Development

- Always test locally before installing as service
- Service requires admin elevation for install/uninstall
- Local mode has UIPI limitations (can't inject into elevated apps)
- Config changes require service restart to take effect
- Logs are the primary debugging tool - check `logs/dictator.log` or run-specific logs
- Thread monitor helps identify deadlocks and blocking operations
- Event-driven mode is the recommended architecture for voice assistant features
