#!/usr/bin/env python3
"""Verify all critical dependencies are installed"""
import sys

deps = [
    'faster_whisper',
    'sounddevice',
    'soundfile',
    'pyperclip',
    'pyautogui',
    'pynput',
    'pystray',
    'PIL',
    'yaml'
]

missing = []
for dep in deps:
    try:
        __import__(dep)
    except ImportError:
        missing.append(dep)

if missing:
    print(f"❌ Missing: {', '.join(missing)}")
    sys.exit(1)
else:
    print("✅ All critical dependencies installed!")
    sys.exit(0)
