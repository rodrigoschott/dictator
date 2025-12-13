# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Dictator Uninstaller

block_cipher = None

a = Analysis(
    ['uninstaller.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'tkinter',
        'tkinter.messagebox',
        'winshell',
        'win32com.client',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude everything heavy
        'torch',
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
        'anthropic',
        'openai',
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
    name='uninstall',  # Output: uninstall.exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # TODO: Add icon when available
    uac_admin=True,  # Request admin rights for service removal
)
