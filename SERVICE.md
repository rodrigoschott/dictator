# Dictator - Voice to Text Service - Technical Documentation

## Overview

Dictator is a Windows background service that provides global voice-to-text functionality using **faster-whisper** (optimized Whisper AI) running locally with GPU acceleration via CTranslate2. Simply press a trigger (mouse button or hotkey), speak, and your words are automatically transcribed and pasted into any active application.

### Key Features

- **Mouse/Keyboard Triggers**: Mouse side button (default) or Ctrl+Alt+V (configurable)
- **Local Processing**: No internet required - all processing happens on your machine
- **GPU Accelerated**: Uses CTranslate2 with CUDA for ultrafast transcription
- **System Tray Integration**: Full control and configuration via tray menu
- **Visual Overlay**: Colored indicator showing recording/processing status
- **TTS Support**: Local text-to-speech with Kokoro-ONNX
- **Auto-paste**: Transcribed text automatically appears in your focused field
- **Windows Service**: Starts automatically with Windows (optional)
- **Multi-language**: Supports 99+ languages via Whisper
- **VAD Support**: Auto-stop recording after silence detection
- **Push-to-Talk**: Record while holding button/key

## Technical Stack

### Core Technologies

- **faster-whisper**: Optimized Whisper implementation using CTranslate2
  - 4-5x faster than original openai-whisper
  - Uses less VRAM
  - **No PyTorch required!** Uses CTranslate2 directly
- **CTranslate2**: Efficient inference engine for Transformer models
  - Supports CUDA, CPU, and quantization
  - Automatic mixed precision (FP16/INT8)
- **kokoro-onnx**: High-quality local TTS engine
  - ONNX Runtime with GPU acceleration
  - Multiple voices and languages
- **pynput**: Global hotkey/mouse event capture
- **pystray**: System tray integration
- **sounddevice/soundfile**: Audio recording and processing

### Why faster-whisper instead of openai-whisper?

| Feature | openai-whisper | faster-whisper |
|---------|----------------|----------------|
| Speed | Baseline | **4-5x faster** |
| VRAM Usage | High | **50% less** |
| Dependencies | PyTorch (~2GB) | CTranslate2 (~200MB) |
| Quantization | Limited | **INT8, FP16, FP32** |
| Streaming | No | **Yes** |

## Dependency Management

This project uses **Poetry** for Python dependency management. Poetry creates an isolated virtual environment and manages all package versions automatically.

### Benefits of Poetry
- **Isolated environment**: Dependencies don't conflict with other Python projects
- **Reproducible builds**: `poetry.lock` ensures everyone uses the same versions
- **Easy dependency management**: Simple commands to add/remove packages
- **Virtual environment**: Automatic .venv creation and management
- **No manual CUDA/PyTorch installation**: Everything via Poetry!

### Poetry Commands

```batch
# Install all dependencies (includes faster-whisper with CUDA)
poetry install

# Update dependencies
poetry update

# Add a new package
poetry add <package-name>

# Run a command in the Poetry environment
poetry run python script.py

# Activate the virtual environment
poetry shell
```

### Important Note on PyTorch

⚠️ **You do NOT need to install PyTorch manually!**

Unlike the original openai-whisper, faster-whisper uses CTranslate2 which is automatically installed via Poetry. This means:
- ✅ Smaller install size (~200MB vs ~2GB)
- ✅ Faster inference
- ✅ No manual CUDA toolkit installation
- ✅ Works out-of-the-box with NVIDIA GPUs

## System Requirements

### Minimum Requirements
- **OS**: Windows 10 or later
- **GPU**: NVIDIA GPU with CUDA support (RTX series recommended)
- **VRAM**: 2GB minimum (model dependent)
- **RAM**: 8GB minimum
- **Python**: 3.8 or later
- **Disk Space**: 5-10GB (depending on model size)

### Your System
- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
- **CUDA**: Version 13.0 (Driver 581.57)
- **Recommended Model**: large or large-v3 (plenty of VRAM available)

## Installation

### Quick Install (Recommended)

**IMPORTANT: First time only** - Install dependencies:
```batch
setup.bat
```

Then, to install as Windows service:

1. **Right-click** `install_service.bat` and select **"Run as Administrator"**
2. The script will automatically:
   - Check Python installation
   - Download and install NSSM (service manager)
   - Install all required Python packages
   - Configure the Windows service
   - Prompt to start the service

3. Check the system tray for the microphone icon

### Manual Installation

If you prefer manual installation:

```batch
# 1. Install Poetry (if not already installed)
python -m pip install poetry

# 2. Install Python dependencies with Poetry
cd D:\Dev\py\Dictator
poetry install --no-dev

# 3. All dependencies are installed via Poetry, including faster-whisper with CUDA support
# No need to manually install PyTorch - faster-whisper uses CTranslate2 which is included!

# 4. Download NSSM from https://nssm.cc/download
# Extract and copy nssm.exe to C:\Windows\System32\

# 5. Install service (using Poetry's virtual environment)
nssm install Dictator "D:\Dev\py\Dictator\.venv\Scripts\python.exe" "D:\Dev\py\Dictator\src\dictator\tray.py" "D:\Dev\py\Dictator\config.yaml"
nssm set Dictator AppDirectory "D:\Dev\py\Dictator"
nssm set Dictator DisplayName "Dictator - Voice to Text"
nssm set Dictator Start SERVICE_AUTO_START

# 6. Start service
nssm start Dictator
```

### Testing Without Installing

To test the application before installing as a service:

```batch
run_local.bat
```

This runs Dictator in local mode without installing it as a Windows service.

## Configuration

All settings are in `config.yaml`:

### Whisper Settings

```yaml
whisper:
  model: "medium"          # Model size (see Model Selection below)
  language: "pt"           # Language code (pt, en, es, etc.)
  device: "cuda"           # Use "cuda" for GPU, "cpu" for CPU only
```

### Hotkey Configuration

```yaml
hotkey:
  trigger: "ctrl+alt+v"         # Hotkey combination
  auto_stop_silence: 2.0        # Stop after N seconds of silence
  max_duration: 60              # Maximum recording duration (seconds)
```

Supported modifiers: `ctrl`, `alt`, `shift`, `win`

### Audio Settings

```yaml
audio:
  sample_rate: 16000       # Sample rate (Whisper expects 16kHz)
  channels: 1              # Mono audio
```

### Paste Settings

```yaml
paste:
  delay: 0.5               # Delay before auto-paste (seconds)
  auto_paste: true         # Automatically paste transcribed text
```

Set `auto_paste: false` to only copy to clipboard without pasting.

### Service Settings

```yaml
service:
  auto_start: true         # Start with Windows
  notifications: true      # Enable notifications (future)
  log_level: "INFO"        # Logging level (DEBUG, INFO, WARNING, ERROR)
  log_file: ""             # Custom log file path (blank = default)
```

### Tray Settings

```yaml
tray:
  enabled: true
  tooltip: "Dictator - Voice to Text (Ctrl+Alt+V)"
```

## Usage

### Basic Usage

1. **Activate Recording**: Press `Ctrl+Alt+V` (or your configured hotkey)
2. **Speak**: The microphone is now recording
3. **Stop Recording**: Press `Ctrl+Alt+V` again or wait for auto-stop
4. **Result**: Transcribed text is automatically pasted into the active field

### System Tray Menu

Right-click the microphone icon in the system tray:

- **Dictator Service**: Shows service name (info only)
- **Hotkey**: Displays current hotkey
- **Model**: Shows active Whisper model
- **Open Config**: Opens config.yaml in Notepad
- **Open Logs**: Opens the log file
- **Restart Service**: Restarts the service (reloads config)
- **Exit**: Stops the service and closes the tray icon

### Viewing Logs

Logs are stored in `logs/dictator.log`

You can:
- Access from tray menu: Right-click icon → **Open Logs**
- Navigate manually: `D:\Dev\py\Dictator\logs\dictator.log`

When running as Windows service, additional logs are in:
- `logs/service.log` - Service stdout
- `logs/service_error.log` - Service stderr

## Model Selection

Whisper offers several models with different accuracy/speed trade-offs:

| Model | VRAM Usage | Speed | Accuracy | Recommended For |
|-------|------------|-------|----------|----------------|
| tiny | ~1GB | Fastest | Low | Testing only |
| base | ~1GB | Very Fast | Medium | Quick notes |
| small | ~2GB | Fast | Good | General use |
| **medium** | ~5GB | Medium | Very Good | **Default** |
| large | ~10GB | Slow | Excellent | High accuracy |
| large-v3 | ~10GB | Slow | Best | Maximum quality |

### For Your RTX 5080 (16GB VRAM)

You have plenty of VRAM available. Recommendations:

- **General Use**: `medium` (default) - Best balance of speed and accuracy
- **Maximum Quality**: `large-v3` - If accuracy is critical
- **Fastest**: `small` - If speed is more important than perfect accuracy

To change models, edit `config.yaml`:

```yaml
whisper:
  model: "large-v3"  # or "medium", "small", etc.
```

Then restart the service (right-click tray icon → **Restart Service**)

## Service Management

### Start/Stop Service

```batch
# Start
nssm start Dictator

# Stop
nssm stop Dictator

# Restart
nssm restart Dictator

# Check status
sc query Dictator
```

### Disable Auto-Start

1. Edit `config.yaml`:
   ```yaml
   service:
     auto_start: false
   ```

2. Update service:
   ```batch
   nssm set Dictator Start SERVICE_DEMAND_START
   ```

### View Service Configuration

```batch
nssm edit Dictator
```

Opens a GUI editor for all service settings.

## Troubleshooting

### Service Won't Start

1. **Check logs**: `logs/service_error.log`
2. **Test locally first**: Run `run_local.bat` to see error messages
3. **Verify Python**: `python --version` (should be 3.8+)
4. **Check CUDA**: Ensure NVIDIA drivers are installed

### Hotkey Not Working

1. **Check conflicts**: Another app might be using the same hotkey
2. **Change hotkey**: Edit `config.yaml` and restart service
3. **Admin rights**: Service needs proper permissions (should be automatic)

### Audio Not Recording

1. **Check microphone**: Windows Settings → Sound → Input
2. **Default device**: Ensure correct mic is set as default
3. **Permissions**: Check Windows microphone privacy settings

### Transcription Errors

1. **Wrong language**: Verify `language: "pt"` in config.yaml
2. **Model too small**: Try a larger model (medium or large)
3. **Audio quality**: Ensure clear speech, minimal background noise
4. **VRAM issues**: Check if GPU is being used by other apps

### GPU Not Being Used

1. **Check CUDA**: Verify `device: "cuda"` in config.yaml
2. **Driver check**: Ensure NVIDIA drivers are current
3. **PyTorch CUDA**: Reinstall torch with CUDA support:
   ```batch
   cd D:\Dev\py\Dictator
   poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
4. **Verify GPU in Python**:
   ```batch
   poetry run python -c "import torch; print(torch.cuda.is_available())"
   # Should print: True
   ```

### High VRAM Usage

Your RTX 5080 has 16GB, so this shouldn't be an issue. However:

1. **Multiple models**: Whisper only loads one model at a time
2. **Ollama coexistence**: Whisper and Ollama can share GPU without conflicts
3. **Model size**: Use `small` or `medium` if needed (large-v3 uses ~10GB)

### Service Fails After Windows Update

Sometimes Windows updates can affect services:

```batch
# Uninstall and reinstall
uninstall_service.bat
install_service.bat
```

## Uninstallation

### Quick Uninstall

1. **Right-click** `uninstall_service.bat` and select **"Run as Administrator"**
2. The script will:
   - Stop the Dictator service
   - Remove the service
   - Optionally remove logs and config

### Manual Uninstall

```batch
# Stop and remove service
nssm stop Dictator
nssm remove Dictator confirm

# (Optional) Remove files
rmdir /s /q "D:\Dev\py\Dictator"
```

## Advanced Configuration

### Custom Log Location

```yaml
service:
  log_file: "C:\Logs\dictator.log"
```

### Multiple Language Support

The model can auto-detect language, or you can specify:

```yaml
whisper:
  language: ""  # Auto-detect (slower)
  # OR specific language:
  language: "en"  # English
  language: "es"  # Spanish
```

### Disable Auto-Paste (Clipboard Only)

```yaml
paste:
  auto_paste: false
```

Text will be copied to clipboard but not automatically pasted.

### Debug Mode

```yaml
service:
  log_level: "DEBUG"
```

Restart service to see detailed debug information in logs.

## Coexistence with Ollama

Dictator and Ollama can run simultaneously on your RTX 5080:

- **Ollama**: Runs continuously in Docker, uses VRAM as needed
- **Whisper**: Only uses GPU during transcription (~5-30 seconds)
- **No conflicts**: They don't compete since Whisper usage is brief

Your 16GB VRAM is more than sufficient for both services.

## Performance Tips

1. **Keep model loaded**: Service keeps model in VRAM for instant transcription
2. **First use slower**: Initial model load takes 5-10 seconds
3. **Recording quality**: Clear speech = better accuracy
4. **Background noise**: Minimize for best results
5. **Silence detection**: Adjust `auto_stop_silence` in config

## Support and Logs

When reporting issues, include:

1. **Config**: Your `config.yaml` file
2. **Logs**: Contents of `logs/dictator.log`
3. **Service logs**: `logs/service_error.log` if running as service
4. **System info**: GPU model, VRAM, Windows version

## Configuration Examples

### English with Large Model
```yaml
whisper:
  model: "large-v3"
  language: "en"
  device: "cuda"
```

### Fast Mode with Small Model
```yaml
whisper:
  model: "small"
  language: "pt"
  device: "cuda"

hotkey:
  auto_stop_silence: 1.0  # Stop after 1 second silence
```

### Clipboard Only (No Auto-Paste)
```yaml
paste:
  auto_paste: false
  delay: 0
```

### Custom Hotkey
```yaml
hotkey:
  trigger: "ctrl+shift+space"
```

## Technical Details

- **Architecture**: Windows Service → System Tray → Hotkey Listener → Audio Recorder → Whisper → Clipboard → Auto-paste
- **Threading**: Non-blocking audio recording and transcription
- **Service Manager**: NSSM (Non-Sucking Service Manager)
- **Audio Library**: sounddevice (PortAudio backend)
- **Clipboard**: pyperclip (Windows clipboard API)
- **Hotkeys**: pynput (low-level keyboard hooks)
- **Tray Icon**: pystray (Windows system tray)

## License

This project uses OpenAI's Whisper model, which is under MIT License.

## Version

Current Version: 1.0.0
Last Updated: 2025-01-10
