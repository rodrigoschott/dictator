#!/usr/bin/env python3
"""Verify all critical dependencies are installed"""
import sys

# Core dependencies required for Dictator to run
deps = [
    'faster_whisper',    # Whisper AI (optimized)
    'sounddevice',       # Audio recording
    'soundfile',         # Audio file handling
    'pyperclip',         # Clipboard operations
    'pyautogui',         # Auto-paste functionality
    'pynput',            # Global hotkey/mouse capture
    'pystray',           # System tray icon
    'PIL',               # Image processing (for tray icon)
    'yaml',              # Config file parsing
    'numpy'              # Array operations
]

# Optional but recommended
optional_deps = [
    'kokoro_onnx'        # TTS engine (optional)
]

print("🔍 Verifying critical dependencies...\n")

missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f"✅ {dep}")
    except ImportError:
        print(f"❌ {dep} - MISSING!")
        missing.append(dep)

print("\n🔍 Checking optional dependencies...\n")

missing_optional = []
for dep in optional_deps:
    try:
        __import__(dep)
        print(f"✅ {dep}")
    except ImportError:
        print(f"⚠️  {dep} - Not installed (TTS will be disabled)")
        missing_optional.append(dep)

print("\n" + "="*50)

if missing:
    print(f"\n❌ CRITICAL: Missing {len(missing)} required dependencies!")
    print(f"   {', '.join(missing)}")
    print("\n💡 Fix: Run 'poetry install' again")
    sys.exit(1)
else:
    print("\n✅ All critical dependencies installed!")
    if missing_optional:
        print(f"⚠️  Optional: {', '.join(missing_optional)} not available")
        print("   TTS features will be disabled")
    print("\n🚀 Dictator is ready to use!")
    sys.exit(0)
