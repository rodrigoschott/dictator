# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dictator** is a Windows Service for local voice-to-text transcription using faster-whisper (GPU-accelerated Whisper) with integrated TTS (Kokoro-ONNX) and LLM capabilities (Ollama/Claude). It runs 100% locally with zero external API dependencies.

**Key Features:**
- Push-to-talk/toggle recording via mouse buttons or keyboard hotkeys
- GPU-accelerated transcription (CUDA required)
- Event-driven architecture with zero polling overhead
- Voice assistant mode with LLM integration
- TTS with instant interrupt capability
- VAD (Voice Activity Detection) for auto-stop

## Development Commands

### Environment Setup
```bash
# Install dependencies (first time only)
setup.bat

# Install Poetry manually if needed
python -m pip install poetry

# Install all dependencies
poetry install
```

### Running Locally
```bash
# Normal mode (doesn't work in elevated apps due to Windows UIPI)
run_local.bat
# Or: poetry run python src/dictator/tray.py config.yaml

# Admin mode (works everywhere)
run_local_admin.bat  # Right-click → Run as Administrator
```

### Testing
```bash
# Run unit tests
poetry run python -m pytest tests/unit/

# Run integration tests
poetry run python -m pytest tests/integration/

# Manual tests
poetry run python tests/manual/test_portuguese_voices.py
poetry run python tests/manual/test_menu_callbacks.py

# Test compilation
poetry run python -m py_compile src/dictator/service.py
```

### Running Dictator (Application Mode)

```bash
# Development mode
run_app.bat
# Or: poetry run python src/dictator/tray.py config.yaml

# Stop application
stop_app.bat
# Or: taskkill /IM Dictator.exe /F

# Add to Windows Startup
add_to_startup.bat

# Remove from Startup
remove_from_startup.bat
```

### Auto-Start Configuration

Dictator runs as a normal Windows application and can start automatically via Windows Startup folder.

**Startup Management:**
- Installer creates shortcut automatically in `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`
- Manual control: Use `add_to_startup.bat` / `remove_from_startup.bat`
- Shortcut launches: `Dictator.exe config.yaml`

**IMPORTANT - UIPI Limitation:**
- Dictator runs with normal user privileges (not SYSTEM)
- **Hotkeys WILL work in:** Browsers, editors, Office, games, most apps (95% of use cases)
- **Hotkeys WON'T work in:** Apps running as Administrator (Admin CMD, Task Manager, etc.)
- **Reason:** Windows security (UIPI) blocks normal processes from sending input to elevated processes
- **Workaround:** Run `Dictator.exe` as Administrator (right-click → Run as Administrator) if you need hotkeys in elevated apps

### Development Tools
```bash
# Verify dependencies
poetry run python verify_deps.py

# View logs (real-time)
Get-Content logs/dictator.log -Wait  # PowerShell

# Clear Poetry cache
poetry cache clear . --all

# Update dependencies
poetry update
```

## Architecture

### Event-Driven Design

The system uses a **zero-polling event-driven architecture** inspired by Speaches.ai:

```
┌─────────────────────────────────────────────────┐
│              EventPubSub (events.py)            │
│  - asyncio.Queue-based pub/sub                  │
│  - Thread-safe publish_nowait()                 │
│  - Blocking await (NO polling)                  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         VoiceSessionManager (session_manager.py)│
│  - Coordinates all voice processing             │
│  - Parallel async tasks (VAD, LLM, TTS)         │
│  - Single LLM call per utterance                │
└─────────────────────────────────────────────────┘
         ↓                  ↓                 ↓
┌─────────────┐   ┌──────────────┐   ┌────────────┐
│VADProcessor │   │  LLMCaller   │   │SentenceChunker│
│(vad_processor)│  │(llm_caller) │   │(sentence_chunker)│
└─────────────┘   └──────────────┘   └────────────┘
```

### State Machine (service.py)

The main service implements a state machine for voice interactions:

```
IDLE → RECORDING (hotkey pressed)
RECORDING → PROCESSING (hotkey released or VAD detects silence)
PROCESSING → SPEAKING (LLM response ready, TTS enabled)
PROCESSING → IDLE (dictation mode, TTS disabled)
SPEAKING → IDLE (TTS finished)
ANY → INTERRUPTED (hotkey pressed during processing/speaking)
INTERRUPTED → RECORDING (new recording starts)
```

States are protected by `_state_lock` (RLock) for thread safety.

### Core Components

**service.py** (DictatorService)
- Main service orchestrator
- Handles hotkey/mouse triggers via pynput
- Manages Whisper model and state machine
- Coordinates AudioProcessor and VoiceSessionManager
- Thread-safe state management with locks

**tray.py** (DictatorTray)
- System tray GUI with pystray
- Dynamic menu generation (Ollama models discovered at runtime)
- Config persistence (saves to config.yaml)
- State callbacks for visual feedback

**overlay.py**
- Visual feedback overlay (tkinter)
- Color-coded states: Red (recording), Orange (processing), Green (TTS)
- Thread-safe state updates

**audio_processor.py** (AudioProcessor)
- Encapsulates sounddevice audio callback
- Thread-safe audio chunk collection
- Forwards chunks to VoiceSessionManager in event-driven mode
- VAD metrics calculation (legacy mode)

**voice/session_manager.py** (VoiceSessionManager)
- Event-driven voice session coordinator
- Runs parallel async tasks for VAD, LLM, TTS
- Manages audio buffering and transcription
- TTS interrupt support (~170ms latency)

**voice/llm_caller.py** (LLMCaller, DirectLLMCaller, OllamaLLMCaller, N8NToolCallingLLMCaller)
- LLM integration with single-call pattern (NOT continuous polling)
- Streaming response with sentence chunking
- Thinking tag filtering for models like Qwen3, DeepSeek-R1
- Call serialization with asyncio.Lock to prevent concurrent calls
- Interrupt handling via task cancellation

**voice/vad_processor.py** (VADProcessor)
- Silero VAD integration for speech detection
- Async audio chunk processing
- Configurable thresholds and timeouts

**voice/events.py** (EventPubSub, Event, EventType)
- Zero-polling pub/sub system using asyncio.Queue
- Thread-safe event publishing with call_soon_threadsafe
- Event history for debugging (auto-trimming deque)

**tts_engine.py** (KokoroTTSEngine)
- Kokoro-ONNX TTS with GPU support
- Thread-safe interrupt capability
- 56 voices across 9 languages
- Session-based playback tracking for interrupt safety

### Thread Safety Principles

1. **State locks**: Use RLock for state that may be accessed recursively
2. **Recording data lock**: Separate lock for audio chunk list operations
3. **Session counters**: Atomic session tracking to discard stale TTS/LLM responses
4. **Event loop thread-safety**: Use `call_soon_threadsafe()` for cross-thread event publishing
5. **AudioProcessor**: All operations protected by RLock
6. **LLM call serialization**: asyncio.Lock prevents concurrent calls

### Configuration (config.yaml)

The config is loaded at startup and saved via tray menu. Key sections:

- `whisper`: Model selection (tiny/base/small/medium/large/large-v3), language, device (cuda/cpu)
- `hotkey`: Trigger type (mouse/keyboard), button/key combo, mode (toggle/push_to_talk), VAD settings
- `tts`: Engine (kokoro-onnx), voice selection, speed, volume, interrupt settings
- `voice`: LLM integration (claude_mode, provider selection, streaming, sentence chunking)
- `overlay`: Visual feedback position and appearance
- `paste`: Auto-paste behavior and delay
- `logging`: Structured logging, trace options, retention

**Dynamic config updates**: Changing LLM provider or model triggers automatic service restart via tray menu.

### LLM Integration Modes

Three provider types in `voice/llm_caller.py`:

1. **DirectLLMCaller**: Claude API direct (requires API key)
2. **OllamaLLMCaller**: Local Ollama models (http://localhost:11434)
3. **N8NToolCallingLLMCaller**: n8n webhook integration

**Thinking tag filtering**: Models like Qwen3 and DeepSeek-R1 expose reasoning in `<think>...</think>` tags. The `remove_thinking_tags()` function strips these before TTS to avoid speaking internal monologue.

### Audio Pipeline

1. **Capture**: sounddevice callback → AudioProcessor.process_chunk()
2. **Event-driven mode**: AudioProcessor → VoiceSessionManager.process_audio_chunk()
3. **VAD**: VADProcessor detects speech/silence via Silero VAD
4. **Transcription**: faster-whisper processes complete audio buffer
5. **LLM**: Single call to LLM provider with conversation history
6. **TTS**: Sentence chunking → Kokoro-ONNX synthesis → sounddevice playback
7. **Interrupt**: Hotkey during TTS → immediate stop via session counter check

### Key Files for Common Tasks

- **Add new LLM provider**: Edit `voice/llm_caller.py`, add new class inheriting from `LLMCaller`
- **Change hotkey handling**: Edit `service.py` hotkey setup functions
- **Modify state machine**: Edit `ServiceState` enum and transition logic in `service.py`
- **Add new event types**: Edit `voice/events.py` EventType enum
- **Change TTS voices**: Edit `tts_engine.py` AVAILABLE_VOICES dict
- **Modify VAD behavior**: Edit `voice/vad_processor.py` VADConfig and processing logic

### Testing Strategy

- **Unit tests** (`tests/unit/`): Component isolation (mocks, stubs)
  - `test_health_check.py`: Dependency validation
  - `test_thinking_tags.py`: LLM tag filtering
  - `test_vad_tts_interrupt.py`: Interrupt behavior
  - `test_auto_restart.py`: Service restart logic

- **Integration tests** (`tests/integration/`): Multi-component workflows
  - `test_ollama_discovery.py`: Ollama API validation
  - `test_ollama_integration.py`: Full LLM flow
  - `test_degradation.py`: Fallback behavior

- **Manual tests** (`tests/manual/`): Interactive verification
  - `test_portuguese_voices.py`: TTS voice samples
  - `test_menu_callbacks.py`: Config auditing
  - `test_claude_cli.py`: Claude CLI verification

### Git LFS Models

Large binary files tracked via Git LFS:
- `kokoro-v1.0.onnx` (310 MB) - TTS model
- `voices-v1.0.bin` (43 MB) - Voice data

These are automatically downloaded on clone if Git LFS is installed.

### Python Version Requirements

- **Required**: Python 3.10 to 3.13 (< 3.14)
- **Reason**: faster-whisper and kokoro-onnx compatibility

### Windows-Specific Considerations

1. **UIPI (User Interface Privilege Isolation)**: Normal process cannot send input to elevated apps
   - Solution: Install as Windows Service (runs with SYSTEM privileges)
   - Alternative: Run as administrator for testing

2. **NSSM (Non-Sucking Service Manager)**: Used for Windows Service installation
   - Auto-downloaded by `install_service.bat`
   - Handles Python script as Windows Service

3. **Path handling**: Always use Windows-style paths (backslashes) in config
   - Use `Path` objects from pathlib for cross-platform compatibility

### Logging

Structured logging via `logging_setup.py`:
- Logs to `logs/dictator.log` (rotated, 5 run retention)
- Levels: DEBUG, INFO, WARNING, ERROR
- Trace options: `trace_main_loop`, `trace_threads`
- Thread monitor for deadlock detection

### Health Check System

`health_check.py` provides dependency validation:
- CUDA availability check
- Model file existence
- Library import verification
- Returns structured ComponentStatus reports

Use for troubleshooting deployment issues.

### Common Pitfalls

1. **Race conditions**: Always acquire locks before state changes
2. **Stale TTS playback**: Check session counter before playing audio
3. **Event loop thread safety**: Use `publish_nowait()` with thread-safe delivery
4. **LLM concurrent calls**: Single call pattern enforced by asyncio.Lock
5. **VAD false positives**: Tune `vad_threshold` in config (0.001-0.01 range)
6. **Memory leaks**: Event history auto-trims at 1000 events
7. **Blocking operations**: Never block event loop with synchronous I/O

### Performance Optimization

- **GPU required**: CPU mode is 5-10x slower for transcription
- **Model selection**: `large-v3` best accuracy (~10GB VRAM), `small` fastest (~2GB VRAM)
- **TTS speed**: Adjust `tts.kokoro.speed` (0.5-2.0, default 1.25)
- **VAD thresholds**: Lower threshold = faster detection but more false positives
- **Async where possible**: All I/O operations use async patterns

### Debugging Tips

- Enable trace logging: Set `logging.trace_main_loop: true` and `logging.trace_threads: true`
- Check event history: `pubsub.get_recent_events(100)` for last 100 events
- Monitor threads: ThreadMonitor logs active threads and detects freezes
- Test components in isolation: Use manual test scripts in `tests/manual/`
- Verify CUDA: Run `verify_deps.py` to check GPU availability
