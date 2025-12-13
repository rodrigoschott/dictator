"""
Build Uninstaller Script

Compiles uninstaller.py em uninstall.exe usando PyInstaller.
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
            shutil.rmtree(dir_path)

    # Clean .spec files
    for spec_file in Path(".").glob("*.spec"):
        print(f"Removing {spec_file}...")
        spec_file.unlink()


def create_pyinstaller_spec():
    """Create PyInstaller .spec file for uninstaller"""

    spec_content = """# -*- mode: python ; coding: utf-8 -*-
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
"""

    spec_path = Path("uninstall.spec")
    spec_path.write_text(spec_content, encoding='utf-8')
    print(f"✓ Created {spec_path}")
    return spec_path


def run_pyinstaller(spec_file: Path):
    """Run PyInstaller with spec file"""
    print(f"\nBuilding uninstaller with PyInstaller...")
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
    exe_path = Path("dist") / "uninstall.exe"

    if exe_path.exists():
        size_kb = exe_path.stat().st_size / 1024
        print(f"\n✓ Uninstaller created: {exe_path}")
        print(f"  Size: {size_kb:.1f} KB")
        return True
    else:
        print(f"\n✗ Uninstaller not found at: {exe_path}")
        return False


def copy_to_assets():
    """Copy uninstaller to installer assets"""
    exe_path = Path("dist") / "uninstall.exe"
    assets_dir = Path("..") / "windows" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    target_path = assets_dir / "uninstall.exe"

    try:
        shutil.copy2(exe_path, target_path)
        print(f"\n✓ Copied uninstaller to: {target_path}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to copy uninstaller: {e}")
        return False


def main():
    """Main build script"""
    print("=" * 70)
    print("  DICTATOR UNINSTALLER BUILD SCRIPT")
    print("=" * 70)

    # Change to uninstaller directory
    uninstaller_dir = Path(__file__).parent
    print(f"\nWorking directory: {uninstaller_dir.absolute()}\n")

    # Check PyInstaller
    if not check_pyinstaller():
        return 1

    # Clean previous builds
    print("\nCleaning previous builds...")
    clean_build_dirs()

    # Create spec file
    print("\nCreating PyInstaller spec...")
    spec_file = create_pyinstaller_spec()

    # Run PyInstaller
    if not run_pyinstaller(spec_file):
        return 1

    # Check output
    if not check_output():
        return 1

    # Copy to assets
    if not copy_to_assets():
        print("\n⚠ Warning: Could not copy to assets directory")
        print("  You may need to copy manually:")
        print(f"  Copy: dist/uninstall.exe")
        print(f"  To:   installer/windows/assets/uninstall.exe")

    print("\n" + "=" * 70)
    print("  BUILD SUCCESSFUL!")
    print("=" * 70)
    print("\nUninstaller location: dist/uninstall.exe")
    print("\nNext steps:")
    print("  1. Test uninstaller manually")
    print("  2. Ensure uninstaller is included in installer assets")
    print("  3. Modify installer to copy uninstaller to installation directory")
    print("  4. Test full installation and uninstallation flow")

    return 0


if __name__ == "__main__":
    sys.exit(main())
