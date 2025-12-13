# Dictator Installer

Self-sufficient installer for Dictator voice-to-text application with complete auto-recovery capabilities and **feature-focused** installation flow.

## Overview

This installer provides a guided, user-friendly experience for installing Dictator with **automatic dependency resolution** based on selected features.

### Key Features

- ✅ **Feature-focused GUI Wizard** - Clear explanations of what each feature does
- ✅ **Conditional dependency installation** - Only installs what you need
- ✅ **Auto-recovery** - Resume interrupted installations
- ✅ **Pre-flight validation** - Checks system requirements before starting
- ✅ **Multi-mirror downloads** - Automatic fallback if one mirror fails
- ✅ **Rollback on failure** - Undoes changes if installation fails
- ✅ **Progress tracking** - Real-time progress bars and logs
- ✅ **Cross-platform** - Windows (GUI) and Linux (shell script)

## Architecture

```
installer/
├── windows/              # Windows installer (GUI)
│   ├── installer_gui.py          # Feature-focused tkinter wizard
│   ├── installer_core.py          # State machine engine (11 steps)
│   ├── dependency_checker.py      # Pre-flight system validation
│   ├── model_downloader.py        # Resume-capable downloads
│   ├── service_installer.py       # NSSM Windows service
│   ├── rollback_manager.py        # LIFO rollback system
│   ├── config_wizard.py           # Config.yaml generation
│   ├── build_installer.py         # PyInstaller build script
│   └── assets/
│       ├── model_manifest.json    # SHA256 checksums & mirrors
│       └── nssm.exe                # Windows service manager
│
├── linux/                # Linux installer (shell script)
│   ├── install.sh                 # Feature-focused bash installer
│   ├── uninstall.sh               # Clean uninstaller
│   └── dictator.service           # systemd template
│
└── shared/               # Cross-platform logic
    ├── state_manager.py           # JSON state persistence
    ├── state_schema.py            # TypedDict schemas
    ├── installer_base.py          # Venv & dependency management
    └── validation.py              # System requirement checks
```

## Feature Selection

The installer presents **5 features** with clear descriptions:

### 1. Core Transcription (Required)
- **What:** Voice-to-text transcription using Whisper AI
- **Benefits:** High-quality local transcription, 99+ languages, 100% offline
- **Dependencies:** faster-whisper, sounddevice, numpy
- **Disk:** 500 MB

### 2. GPU Acceleration (Optional)
- **What:** GPU-accelerated transcription (5-10x faster)
- **Benefits:** 1-3s latency vs 10-30s on CPU
- **Requirements:** NVIDIA GPU, CUDA drivers, 4GB+ VRAM
- **Dependencies:** PyTorch with CUDA
- **Disk:** 2000 MB | **Download:** 800 MB
- **Fallback:** CPU mode (slower but functional)

### 3. Text-to-Speech (Optional)
- **What:** High-quality voice synthesis with 56 voices
- **Benefits:** Natural voice, adjustable speed, instant interrupt
- **Dependencies:** kokoro-onnx, model files (338 MB download)
- **Disk:** 500 MB | **Download:** 338 MB
- **Fallback:** Silent mode (transcription only)

### 4. Voice Activity Detection (Optional)
- **What:** Auto-stop recording after silence
- **Benefits:** Hands-free operation, no click to stop
- **Requirements:** GPU recommended for best performance
- **Dependencies:** silero-vad, torch
- **Disk:** 50 MB
- **Fallback:** Manual mode (click to stop)

### 5. LLM Voice Assistant (Optional)
- **What:** Conversational AI with local LLMs
- **Benefits:** Talk to Ollama models, context preservation, spoken responses
- **Requirements:** Ollama installed or Claude API
- **Dependencies:** aiohttp, requests
- **Disk:** 100 MB
- **Fallback:** Simple dictation mode

## Building the Installer

### Windows Executable

**Prerequisites:**
```bash
# Install PyInstaller
poetry add --group dev pyinstaller

# Download NSSM (required for service)
# 1. Go to https://nssm.cc/download
# 2. Download nssm-2.24.zip
# 3. Extract nssm.exe from win64/ folder
# 4. Place in: installer/windows/assets/
```

**Build:**
```bash
cd installer/windows
poetry run python build_installer.py
```

**Output:** `dist/DictatorInstaller.exe` (~80-120 MB)

**What it includes:**
- Python interpreter (embedded)
- All installer Python code
- Model manifest with checksums
- NSSM service manager
- Requirements.txt for runtime install

**What it downloads during installation:**
- Python dependencies (based on selected features)
- TTS models (338 MB - only if TTS enabled)
- PyTorch with CUDA (800 MB - only if GPU enabled)

### Linux Installation

```bash
cd installer/linux
chmod +x install.sh
sudo ./install.sh
```

**Interactive prompts will ask:**
1. Enable GPU? (y/N)
2. Enable TTS? (y/N)
3. Enable VAD? (y/N)
4. Enable LLM? (y/N)
5. Whisper model? (tiny/small/medium/large-v3)

**What it does:**
- ✓ Checks system dependencies
- ✓ Creates virtual environment
- ✓ Installs only selected features
- ✓ Downloads models (if TTS enabled)
- ✓ Generates config.yaml
- ✓ Installs systemd service
- ✓ Starts service automatically

## Installation Flow

### Windows GUI Wizard (8 Screens)

1. **Welcome** - Introduction
2. **System Check** - Pre-flight validation (OS, Python, RAM, disk, GPU, network)
3. **Feature Selection** ⭐ - Choose features with detailed explanations
4. **Whisper Model** - Select model size (with GPU/CPU recommendations)
5. **Installation Location** - Choose install directory
6. **Confirmation** - Review summary before proceeding
7. **Installation Progress** - Real-time progress bars and logs
8. **Completion** - Success/failure with next steps

### Linux Shell Script

1. Root check
2. Distro detection
3. Dependency check (Python, pip, venv, portaudio, git)
4. Feature selection (interactive prompts)
5. Confirmation
6. Directory creation
7. Venv creation
8. Dependency installation (conditional)
9. Model download (if TTS enabled)
10. Config generation
11. Systemd service installation
12. Service start
13. Completion summary

## State Management & Recovery

### State File Location
- Windows: `C:\Users\{user}\.dictator\installer_state.json`
- Linux: `~/.dictator/installer_state.json`

### Recovery Options

If installation is interrupted, the installer detects partial state and offers:

1. **Resume** - Continue from last checkpoint
2. **Retry** - Retry failed step
3. **Rollback** - Undo to last successful checkpoint
4. **Clean Install** - Remove all and start fresh
5. **Cancel** - Exit installer

### Rollback Actions

The installer can undo:
- ✓ Created directories
- ✓ Virtual environment
- ✓ Downloaded models
- ✓ Generated config files
- ✓ Installed Windows service

## Development

### Install Dev Dependencies

```bash
poetry install --with dev
```

### Run Tests

```bash
# Unit tests
poetry run pytest tests/installer/unit/

# Integration tests
poetry run pytest tests/installer/integration/

# Manual testing
poetry run python installer/windows/installer_gui.py
```

### Test in Clean Environment

**Windows:**
- Use Windows Sandbox or clean VM
- Test all feature combinations
- Verify service installation
- Test rollback scenarios

**Linux:**
- Use Docker or clean VM
- Test on Ubuntu 22.04, Debian 12, Fedora 39
- Verify systemd service
- Test permissions

## Usage

### Windows

1. Download `DictatorInstaller.exe`
2. Right-click → **Run as Administrator**
3. Follow wizard prompts
4. Select desired features
5. Wait for installation
6. Check system tray for microphone icon

### Linux

```bash
# Download installer
curl -LO https://github.com/rodrigoschott/dictator/releases/latest/download/install.sh
chmod +x install.sh

# Run with sudo
sudo ./install.sh

# Follow prompts and select features

# Check service status
systemctl status dictator

# View logs
journalctl -u dictator -f
```

## Uninstallation

### Windows

Run the uninstaller from Windows Settings or:
```bash
# Via service
cd "C:\Program Files\Dictator"
uninstall_service.bat
```

### Linux

```bash
cd installer/linux
sudo ./uninstall.sh
```

## Troubleshooting

### Windows

**Installer won't start:**
- Ensure running as Administrator
- Check Windows Defender (might block unsigned exe)
- Verify .NET Framework 4.7.2+ installed

**Installation fails at GPU step:**
- Check NVIDIA drivers installed
- Verify CUDA toolkit version
- Feature will gracefully fallback to CPU mode

**Service won't start:**
- Check logs: `C:\Program Files\Dictator\logs\service_error.log`
- Verify Python path in service config
- Try manual start: `nssm start Dictator`

### Linux

**Permission errors:**
- Ensure running with `sudo`
- Check `/opt/dictator` ownership

**Service fails to start:**
- Check logs: `journalctl -u dictator -f`
- Verify Python venv created: `/opt/dictator/venv`
- Check audio permissions

**GPU not detected:**
- Install NVIDIA drivers: `ubuntu-drivers autoinstall`
- Verify CUDA: `nvidia-smi`
- Installer will fallback to CPU mode

## File Locations

### Windows
- **Installation:** `C:\Program Files\Dictator\`
- **Config:** `C:\Program Files\Dictator\config\config.yaml`
- **Logs:** `C:\Program Files\Dictator\logs\`
- **Models:** `C:\Program Files\Dictator\` (if TTS enabled)
- **State:** `C:\Users\{user}\.dictator\installer_state.json`

### Linux
- **Installation:** `/opt/dictator/`
- **Config:** `~/.config/dictator/config.yaml`
- **Logs:** `/opt/dictator/logs/`
- **Models:** `/opt/dictator/` (if TTS enabled)
- **Service:** `/etc/systemd/system/dictator.service`

## Technical Details

### Installation Steps (11 total)

1. `pre_flight_check` - System validation
2. `select_features` - Feature selection
3. `create_directories` - Directory structure ⭐
4. `create_venv` - Virtual environment ⭐
5. `install_dependencies` - Python packages ⭐
6. `download_models` - TTS models (if enabled) ⭐
7. `verify_models` - Checksum validation
8. `generate_config` - Config.yaml creation ⭐
9. `install_service` - NSSM/systemd setup ⭐
10. `final_validation` - Health check
11. `start_service` - Service start

⭐ = Creates rollback point

### Dependencies Matrix

| Feature | Python Packages | System Deps | Download Size |
|---------|----------------|-------------|---------------|
| Core | faster-whisper, sounddevice, soundfile, numpy, pynput, pystray, pillow, pyyaml | Microphone | 0 MB |
| GPU | torch (with CUDA index) | NVIDIA drivers, CUDA | 800 MB |
| TTS | kokoro-onnx[gpu] | Model files | 338 MB |
| VAD | silero-vad (via torch) | - | 0 MB |
| LLM | aiohttp, requests, psutil | Ollama (optional) | 0 MB |

### Total Install Sizes

| Configuration | Disk Space | Download |
|--------------|------------|----------|
| Minimal (CPU only) | ~1 GB | ~0 MB |
| Typical (GPU + TTS) | ~4 GB | ~1.1 GB |
| Full (all features) | ~4.3 GB | ~1.1 GB |

## Contributing

See main [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](../../LICENSE)
