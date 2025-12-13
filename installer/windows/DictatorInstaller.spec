# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Dictator Installer

import sys
from pathlib import Path

block_cipher = None

# Base directory
base_dir = Path(SPECPATH)

# Data files to include
datas = [
    # Model manifest
    (str(base_dir / 'assets' / 'model_manifest.json'), 'assets'),

    # Requirements file (from project root)
    (str(base_dir.parent.parent / 'requirements.txt'), '.'),

    # pyproject.toml (needed for pip install -e)
    (str(base_dir.parent.parent / 'pyproject.toml'), '.'),

    # README.md (referenced by pyproject.toml)
    (str(base_dir.parent.parent / 'README.md'), '.'),

    # NSSM executable
    (str(base_dir / 'assets' / 'nssm.exe'), 'assets'),

    # Dictator launcher executable (built separately)
    (str(base_dir / 'assets' / 'Dictator.exe'), 'assets'),

    # Uninstaller executable (built separately)
    (str(base_dir / 'assets' / 'uninstall.exe'), 'assets'),

    # Dictator source code (to be installed on target system)
    (str(base_dir.parent.parent / 'src' / 'dictator'), 'src/dictator'),
]

# Hidden imports (ONLY what installer needs)
hiddenimports = [
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'winshell',
    'win32com.client',
    'pywintypes',
    'pythoncom',
]

# Binary files
binaries = []

a = Analysis(
    ['installer_gui.py'],
    pathex=[],  # Don't add extra paths - avoid picking up src/
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude ALL heavy packages - these will be installed by the installer
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'jupyter',
        'IPython',
        'faster_whisper',
        'onnxruntime',
        'sounddevice',
        'pynput',
        'pystray',
        'pyaudio',
        'silero',
        'anthropic',
        'openai',
        # Dictator source code - NOT needed in installer
        'dictator',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DictatorInstaller',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # TODO: Add installer icon
    version_file=None,  # TODO: Add version info
    uac_admin=True,  # Require administrator privileges
    uac_uiaccess=False,
)
