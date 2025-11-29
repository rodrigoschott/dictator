#!/usr/bin/env python3
"""
Verify all dependencies for Dictator installation

IMPORTANT: Run this script using Poetry's environment:
    poetry run python verify_deps.py

Performs comprehensive validation of:
- Python packages (critical and optional)
- Git LFS installation and model files
- External services (Ollama, N8N) availability
- System capabilities (GPU, audio devices)
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Verify Python version is 3.10+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        return False, f"Python {version.major}.{version.minor}.{version.micro}"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"


def check_python_packages():
    """Check critical and optional Python packages"""
    critical_deps = [
        'faster_whisper',    # Whisper AI (optimized)
        'sounddevice',       # Audio recording
        'soundfile',         # Audio file handling
        'pyperclip',         # Clipboard operations
        'pyautogui',         # Auto-paste functionality
        'pynput',            # Global hotkey/mouse capture
        'pystray',           # System tray icon
        'PIL',               # Image processing (for tray icon)
        'yaml',              # Config file parsing
        'numpy',             # Array operations
        'aiohttp',           # Async HTTP client
        'requests'           # HTTP client
    ]

    optional_deps = [
        'kokoro_onnx',       # TTS engine (optional)
        'torch'              # PyTorch (for VAD, optional)
    ]

    missing_critical = []
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_critical.append(dep)

    missing_optional = []
    for dep in optional_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(dep)

    return critical_deps, missing_critical, optional_deps, missing_optional


def check_git_lfs():
    """Check if Git LFS is installed"""
    try:
        result = subprocess.run(
            ['git', 'lfs', 'version'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0] if result.stdout else "unknown"
            return True, version
        return False, "Not installed"
    except FileNotFoundError:
        return False, "Git not found"
    except Exception as e:
        return False, str(e)


def check_model_files():
    """Check if TTS model files exist and are not LFS pointers"""
    model_path = Path("kokoro-v1.0.onnx")
    voices_path = Path("voices-v1.0.bin")

    if not model_path.exists():
        return False, f"Model file not found: {model_path}"

    if not voices_path.exists():
        return False, f"Voices file not found: {voices_path}"

    # Check if files are LFS pointers (tiny files starting with "version https://")
    model_size = model_path.stat().st_size
    voices_size = voices_path.stat().st_size

    if model_size < 1000:  # LFS pointers are very small
        with open(model_path, 'rb') as f:
            header = f.read(20)
            if header.startswith(b'version https://'):
                return False, "Model files are LFS pointers (not downloaded)"

    total_size_mb = (model_size + voices_size) / 1e6
    return True, f"{total_size_mb:.0f}MB total"


def check_gpu_cuda():
    """Check GPU/CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "unknown"
            return True, f"{gpu_name} (CUDA {cuda_version})"
        else:
            return False, "CUDA not available"
    except ImportError:
        return None, "Cannot verify (torch not installed)"


def check_audio_device():
    """Check if audio input device is accessible"""
    try:
        import sounddevice as sd
        device_info = sd.query_devices(kind='input')
        return True, device_info['name']
    except Exception as e:
        return False, str(e)


def check_ollama():
    """Check if Ollama service is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, f"{len(models)} model(s) available"
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


def check_poetry():
    """Check Poetry installation"""
    try:
        result = subprocess.run(
            ['poetry', '--version'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, "Not installed"
    except FileNotFoundError:
        return False, "Not found in PATH"
    except Exception as e:
        return False, str(e)


def check_poetry_env():
    """Check if running inside Poetry's virtual environment"""
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix

    if not in_venv:
        print("=" * 70)
        print("  WARNING: Not running in Poetry virtual environment")
        print("=" * 70)
        print("\nYou are running this script with system Python.")
        print("Dependencies are installed in Poetry's virtual environment.\n")
        print("Please run:")
        print("    poetry run python verify_deps.py\n")
        print("Or activate the Poetry shell first:")
        print("    poetry shell")
        print("    python verify_deps.py\n")
        return False
    return True


def main():
    """Run all dependency checks"""
    print("=" * 70)
    print("  DICTATOR DEPENDENCY VERIFICATION")
    print("=" * 70)

    # Check if running in Poetry environment
    if not check_poetry_env():
        print("\n" + "=" * 70)
        print("[FAIL] Please run with: poetry run python verify_deps.py")
        print("=" * 70)
        sys.exit(1)

    all_passed = True
    warnings = []

    # Python version
    print("\n[System Requirements]\n")
    ok, info = check_python_version()
    if ok:
        print(f"[OK] {info}")
    else:
        print(f"[FAIL] {info} (requires 3.10+)")
        all_passed = False

    # Poetry
    ok, info = check_poetry()
    if ok:
        print(f"[OK] {info}")
    else:
        print(f"[WARN] Poetry: {info}")
        warnings.append("Poetry not found (required for development)")

    # Critical dependencies
    print("\n[Critical Dependencies]\n")
    critical_deps, missing_critical, optional_deps, missing_optional = check_python_packages()

    if not missing_critical:
        print(f"[OK] Python packages ({len(critical_deps)}/{len(critical_deps)})")
    else:
        print(f"[FAIL] Python packages ({len(critical_deps) - len(missing_critical)}/{len(critical_deps)})")
        print(f"       Missing: {', '.join(missing_critical)}")
        print(f"       Fix: Run 'poetry install'")
        all_passed = False

    # Git LFS
    ok, info = check_git_lfs()
    if ok:
        print(f"[OK] Git LFS ({info})")
    else:
        print(f"[FAIL] Git LFS: {info}")
        print(f"       Fix: Install from https://git-lfs.github.com/")
        all_passed = False

    # Model files
    ok, info = check_model_files()
    if ok:
        print(f"[OK] Model files ({info})")
    else:
        print(f"[WARN] Model files: {info}")
        print(f"       Fix: Run 'git lfs pull'")
        warnings.append("TTS model files missing")

    # Optional dependencies
    print("\n[Optional Dependencies]\n")

    # Kokoro TTS
    if 'kokoro_onnx' not in missing_optional:
        print(f"[OK] Kokoro TTS engine")
    else:
        print(f"[WARN] Kokoro TTS: Not installed (TTS will be disabled)")
        warnings.append("TTS disabled")

    # Torch (for VAD)
    if 'torch' not in missing_optional:
        print(f"[OK] PyTorch (for VAD)")
    else:
        print(f"[WARN] PyTorch: Not installed (VAD will be disabled)")

    # External services
    print("\n[External Services - Optional]\n")

    # Ollama
    ok, info = check_ollama()
    if ok:
        print(f"[OK] Ollama ({info})")
    else:
        print(f"[WARN] Ollama: {info}")
        print(f"       Voice assistant features will be disabled")
        warnings.append("Ollama not running")

    # System info
    print("\n[System Info]\n")

    # GPU/CUDA
    ok, info = check_gpu_cuda()
    if ok:
        print(f"[OK] GPU: {info}")
    elif ok is False:
        print(f"[WARN] GPU: {info} (will use CPU mode)")
        warnings.append("GPU not available")
    else:
        print(f"[INFO] GPU: {info}")

    # Audio device
    ok, info = check_audio_device()
    if ok:
        print(f"[OK] Audio device: {info}")
    else:
        print(f"[FAIL] Audio device: {info}")
        print(f"       Fix: Check Windows microphone permissions")
        all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed and not warnings:
        print("\n[PASS] ALL CHECKS PASSED")
        print("\nDictator is ready to install and use!")
        print("\nNext steps:")
        print("  1. Run: install_service.bat (as administrator)")
        print("  2. Start the Dictator service")
        sys.exit(0)

    elif all_passed and warnings:
        print(f"\n[PASS] PASSED WITH {len(warnings)} WARNING(S)")
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nDictator is ready to install with limited features")
        print("\nNext steps:")
        print("  1. Run: install_service.bat (as administrator)")
        print("  2. Start the Dictator service")
        print("\nNote: Some features will be disabled due to warnings above")
        sys.exit(0)

    else:
        print("\n[FAIL] VERIFICATION FAILED")
        print("\nPlease fix the errors above before installing.")
        print("\nCommon fixes:")
        print("  - Install dependencies: poetry install")
        print("  - Install Git LFS: https://git-lfs.github.com/")
        print("  - Download model files: git lfs pull")
        sys.exit(1)


if __name__ == "__main__":
    main()
