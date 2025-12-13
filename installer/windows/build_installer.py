"""
Build Installer Script

Builds standalone Windows installer executable using PyInstaller.
Bundles all dependencies, assets, and creates single-file .exe.
"""

import sys
import shutil
import subprocess
from pathlib import Path

# Ensure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("✗ PyInstaller not found")
        print("\nInstall with: poetry add --group dev pyinstaller")
        return False


def clean_build_dirs():
    """Clean previous build artifacts"""
    dirs_to_clean = ["build", "dist", "__pycache__"]

    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"Cleaning {dir_name}/...")
            try:
                shutil.rmtree(dir_path)
            except PermissionError as e:
                print(f"  Warning: Could not remove {dir_name} (in use): {e}")
                print(f"  Continuing anyway...")

    # Clean .spec files
    for spec_file in Path(".").glob("*.spec"):
        print(f"Removing {spec_file}...")
        try:
            spec_file.unlink()
        except PermissionError as e:
            print(f"  Warning: Could not remove {spec_file} (in use): {e}")


def create_pyinstaller_spec():
    """Create PyInstaller .spec file"""

    spec_content = """# -*- mode: python ; coding: utf-8 -*-
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
"""

    spec_path = Path("DictatorInstaller.spec")
    spec_path.write_text(spec_content, encoding='utf-8')
    print(f"✓ Created {spec_path}")
    return spec_path


def run_pyinstaller(spec_file: Path):
    """Run PyInstaller with spec file"""
    print(f"\nBuilding installer with PyInstaller...")
    print(f"Spec file: {spec_file}")
    print("-" * 70)

    try:
        # Run PyInstaller
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", str(spec_file), "--clean"],
            check=True,
            capture_output=False
        )

        print("-" * 70)
        print("✓ Build completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print("-" * 70)
        print(f"✗ Build failed with exit code {e.returncode}")
        return False


def check_output():
    """Check if output executable exists"""
    exe_path = Path("dist") / "DictatorInstaller.exe"

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Installer created: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"\n✗ Installer not found at: {exe_path}")
        return False


def create_version_file():
    """Create Windows version info file for executable"""

    version_info = """# UTF-8
#
# For more details about fixed file info:
# See https://docs.microsoft.com/en-us/windows/win32/menurc/versioninfo-resource

VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'Dictator'),
        StringStruct(u'FileDescription', u'Dictator Voice to Text Installer'),
        StringStruct(u'FileVersion', u'1.0.0.0'),
        StringStruct(u'InternalName', u'DictatorInstaller'),
        StringStruct(u'LegalCopyright', u'© 2025. Licensed under MIT.'),
        StringStruct(u'OriginalFilename', u'DictatorInstaller.exe'),
        StringStruct(u'ProductName', u'Dictator Installer'),
        StringStruct(u'ProductVersion', u'1.0.0.0')])
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""

    version_path = Path("version_info.txt")
    version_path.write_text(version_info, encoding='utf-8')
    print(f"✓ Created {version_path}")
    return version_path


def download_nssm():
    """Download NSSM if not present"""
    assets_dir = Path("assets")
    nssm_path = assets_dir / "nssm.exe"

    if nssm_path.exists():
        print(f"✓ NSSM already present: {nssm_path}")
        return True

    print("\n⚠ NSSM not found in assets/")
    print("\nNSSM is required for Windows service installation.")
    print("Please download NSSM manually:")
    print("  1. Go to: https://nssm.cc/download")
    print("  2. Download nssm-2.24.zip")
    print("  3. Extract nssm.exe from win64/ folder")
    print(f"  4. Place it in: {assets_dir.absolute()}")
    print("\nContinue build without NSSM? (Service installation will fail)")

    response = input("Continue? [y/N]: ").strip().lower()
    return response == 'y'


def check_assets():
    """Check if all required assets are present"""
    assets_dir = Path("assets")
    required_files = [
        "model_manifest.json",
        "Dictator.exe",  # Launcher executable (must be built first)
        "uninstall.exe"  # Uninstaller executable (must be built first)
    ]

    optional_files = [
        "nssm.exe",
        "icon.ico",
        "logo.png"
    ]

    all_present = True

    print("\nChecking required assets:")
    for filename in required_files:
        file_path = assets_dir / filename
        if file_path.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")
            if filename == "Dictator.exe":
                print("\n    To build Dictator.exe launcher:")
                print("      cd installer/launcher")
                print("      poetry run python build_launcher.py")
                print("      (will be copied to assets/ automatically)\n")
            elif filename == "uninstall.exe":
                print("\n    To build uninstall.exe:")
                print("      cd installer/uninstaller")
                print("      poetry run python build_uninstaller.py")
                print("      (will be copied to assets/ automatically)\n")
            all_present = False

    print("\nChecking optional assets:")
    for filename in optional_files:
        file_path = assets_dir / filename
        if file_path.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  - {filename} - Not present (optional)")

    return all_present


def main():
    """Main build script"""
    print("=" * 70)
    print("  DICTATOR INSTALLER BUILD SCRIPT")
    print("=" * 70)

    # Change to installer directory
    installer_dir = Path(__file__).parent
    print(f"\nWorking directory: {installer_dir.absolute()}\n")

    # Check PyInstaller
    if not check_pyinstaller():
        return 1

    # Check assets
    if not check_assets():
        print("\n✗ Required assets missing!")
        return 1

    # Check NSSM
    if not download_nssm():
        print("\n⚠ Building without NSSM (service installation will not work)")

    # Clean previous builds
    print("\nCleaning previous builds...")
    clean_build_dirs()

    # Create version info
    print("\nCreating version info...")
    create_version_file()

    # Create spec file
    print("\nCreating PyInstaller spec...")
    spec_file = create_pyinstaller_spec()

    # Run PyInstaller
    if not run_pyinstaller(spec_file):
        return 1

    # Check output
    if not check_output():
        return 1

    print("\n" + "=" * 70)
    print("  BUILD SUCCESSFUL!")
    print("=" * 70)
    print("\nInstaller location: dist/DictatorInstaller.exe")
    print("\nNext steps:")
    print("  1. Test the installer in a clean VM")
    print("  2. Test all feature combinations")
    print("  3. Verify service installation works")
    print("  4. Check rollback functionality")
    print("\nOptional:")
    print("  - Code sign the executable (for production)")
    print("  - Create MSI wrapper with WiX")
    print("  - Upload to GitHub Releases")

    return 0


if __name__ == "__main__":
    sys.exit(main())
