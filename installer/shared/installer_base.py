"""
Installer Base

Shared base logic for cross-platform installers.
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger("InstallerBase")


class VirtualEnvManager:
    """
    Virtual environment management

    Handles creation and activation of Python virtual environments.
    """

    @staticmethod
    def create_venv(venv_dir: Path, python_exe: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Create virtual environment

        Args:
            venv_dir: Directory for virtual environment
            python_exe: Python executable to use (default: system Python)

        Returns:
            (success, message)
        """
        if python_exe is None:
            # When running from PyInstaller, sys.executable points to the .exe
            # We need to find the actual Python interpreter
            if getattr(sys, 'frozen', False):
                # Running from PyInstaller - find system Python
                import shutil
                python_exe = shutil.which("python")
                if python_exe is None:
                    python_exe = shutil.which("python3")
                if python_exe is None:
                    # Try common locations
                    for path in [r"C:\Python313\python.exe", r"C:\Python312\python.exe", r"C:\Python311\python.exe"]:
                        if Path(path).exists():
                            python_exe = path
                            break
                if python_exe is None:
                    logger.error("Could not find Python interpreter!")
                    return False, "Python interpreter not found. Please install Python 3.10-3.13"
                python_exe = Path(python_exe)
                logger.info(f"Found system Python: {python_exe}")
            else:
                python_exe = Path(sys.executable)

        logger.info(f"Creating virtual environment at: {venv_dir}")

        try:
            # Ensure parent directory exists
            venv_dir.parent.mkdir(parents=True, exist_ok=True)

            # Create venv
            result = subprocess.run(
                [str(python_exe), "-m", "venv", str(venv_dir)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                logger.info(f"✓ Virtual environment created: {venv_dir}")
                return True, "Virtual environment created"
            else:
                error = result.stderr.strip() or result.stdout.strip()
                logger.error(f"✗ Failed to create venv: {error}")
                return False, f"Failed to create venv: {error}"

        except subprocess.TimeoutExpired:
            logger.error("Virtual environment creation timed out")
            return False, "Timeout creating virtual environment"
        except Exception as e:
            logger.error(f"Error creating virtual environment: {e}")
            return False, str(e)

    @staticmethod
    def get_venv_python(venv_dir: Path) -> Path:
        """
        Get path to Python executable in virtual environment

        Args:
            venv_dir: Virtual environment directory

        Returns:
            Path to python.exe (Windows) or python (Unix)
        """
        if sys.platform == "win32":
            return venv_dir / "Scripts" / "python.exe"
        else:
            return venv_dir / "bin" / "python"

    @staticmethod
    def get_venv_pip(venv_dir: Path) -> Path:
        """
        Get path to pip executable in virtual environment

        Args:
            venv_dir: Virtual environment directory

        Returns:
            Path to pip.exe (Windows) or pip (Unix)
        """
        if sys.platform == "win32":
            return venv_dir / "Scripts" / "pip.exe"
        else:
            return venv_dir / "bin" / "pip"

    @staticmethod
    def verify_venv(venv_dir: Path) -> bool:
        """
        Verify virtual environment is valid

        Args:
            venv_dir: Virtual environment directory

        Returns:
            True if valid
        """
        python_exe = VirtualEnvManager.get_venv_python(venv_dir)
        pip_exe = VirtualEnvManager.get_venv_pip(venv_dir)

        return python_exe.exists() and pip_exe.exists()


class DependencyInstaller:
    """
    Python dependency installer

    Handles pip package installation in virtual environments.
    """

    def __init__(self, venv_dir: Path):
        """
        Initialize dependency installer

        Args:
            venv_dir: Virtual environment directory
        """
        self.venv_dir = venv_dir
        self.pip_exe = VirtualEnvManager.get_venv_pip(venv_dir)

    def upgrade_pip(self) -> Tuple[bool, str]:
        """
        Upgrade pip to latest version

        Returns:
            (success, message)
        """
        logger.info("Upgrading pip...")

        try:
            result = subprocess.run(
                [str(self.pip_exe), "install", "--upgrade", "pip"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                logger.info("✓ Pip upgraded")
                return True, "Pip upgraded"
            else:
                error = result.stderr.strip() or result.stdout.strip()
                logger.warning(f"Pip upgrade failed: {error}")
                return False, f"Pip upgrade failed: {error}"

        except Exception as e:
            logger.warning(f"Error upgrading pip: {e}")
            return False, str(e)

    def install_requirements(
        self,
        requirements_file: Path,
        timeout: int = 600
    ) -> Tuple[bool, str]:
        """
        Install packages from requirements.txt

        Args:
            requirements_file: Path to requirements.txt
            timeout: Timeout in seconds

        Returns:
            (success, message)
        """
        logger.info(f"Installing requirements from: {requirements_file}")

        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False, "Requirements file not found"

        try:
            result = subprocess.run(
                [
                    str(self.pip_exe),
                    "install",
                    "-r",
                    str(requirements_file)
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                logger.info("✓ Requirements installed")
                return True, "Requirements installed"
            else:
                error = result.stderr.strip() or result.stdout.strip()
                logger.error(f"✗ Failed to install requirements: {error}")
                return False, f"Failed to install: {error}"

        except subprocess.TimeoutExpired:
            logger.error(f"Installation timed out after {timeout}s")
            return False, "Installation timed out"
        except Exception as e:
            logger.error(f"Error installing requirements: {e}")
            return False, str(e)

    def install_package(
        self,
        package: str,
        version: Optional[str] = None,
        extra_args: Optional[list] = None,
        timeout: int = 300
    ) -> Tuple[bool, str]:
        """
        Install a single package

        Args:
            package: Package name
            version: Optional version specifier
            extra_args: Extra pip arguments
            timeout: Timeout in seconds

        Returns:
            (success, message)
        """
        package_spec = f"{package}=={version}" if version else package
        logger.info(f"Installing package: {package_spec}")

        try:
            cmd = [str(self.pip_exe), "install", package_spec]

            if extra_args:
                cmd.extend(extra_args)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                logger.info(f"✓ Installed: {package_spec}")
                return True, f"Installed {package_spec}"
            else:
                error = result.stderr.strip() or result.stdout.strip()
                logger.error(f"✗ Failed to install {package_spec}: {error}")
                return False, f"Failed to install: {error}"

        except subprocess.TimeoutExpired:
            logger.error(f"Installation timed out for {package_spec}")
            return False, "Installation timed out"
        except Exception as e:
            logger.error(f"Error installing {package_spec}: {e}")
            return False, str(e)

    def install_package_editable(
        self,
        package_dir: Path,
        timeout: int = 300
    ) -> Tuple[bool, str]:
        """
        Install a package in editable mode (pip install -e)

        Args:
            package_dir: Directory containing setup.py or pyproject.toml
            timeout: Timeout in seconds

        Returns:
            (success, message)
        """
        logger.info(f"Installing package in editable mode: {package_dir}")

        if not package_dir.exists():
            logger.error(f"Package directory not found: {package_dir}")
            return False, "Package directory not found"

        try:
            result = subprocess.run(
                [str(self.pip_exe), "install", "-e", str(package_dir)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                logger.info(f"✓ Package installed in editable mode: {package_dir}")
                return True, f"Package installed: {package_dir}"
            else:
                error = result.stderr.strip() or result.stdout.strip()
                logger.error(f"✗ Failed to install package: {error}")
                return False, f"Failed to install: {error}"

        except subprocess.TimeoutExpired:
            logger.error(f"Installation timed out for {package_dir}")
            return False, "Installation timed out"
        except Exception as e:
            logger.error(f"Error installing package: {e}")
            return False, str(e)

    def install_torch_cuda(self, cuda_version: str = "cu121") -> Tuple[bool, str]:
        """
        Install PyTorch and torchvision with CUDA support

        Args:
            cuda_version: CUDA version (cu118, cu121, cu124, etc.)

        Returns:
            (success, message)
        """
        logger.info(f"Installing PyTorch with CUDA {cuda_version}")

        index_url = f"https://download.pytorch.org/whl/{cuda_version}"

        # Install both torch and torchvision
        result = subprocess.run(
            [
                str(self.pip_exe),
                "install",
                "torch",
                "torchvision",
                "--index-url",
                index_url
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            logger.info(f"PyTorch with CUDA {cuda_version} installed successfully")
            return True, "PyTorch installed successfully"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            logger.error(f"Failed to install PyTorch: {error_msg}")
            return False, f"PyTorch installation failed: {error_msg}"

    def verify_package(self, package: str) -> bool:
        """
        Verify a package is installed

        Args:
            package: Package name to check

        Returns:
            True if installed
        """
        try:
            result = subprocess.run(
                [str(self.pip_exe), "show", package],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_time_remaining(seconds: float) -> str:
    """
    Format time remaining in human-readable format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def ensure_directory(directory: Path) -> bool:
    """
    Ensure directory exists, create if needed

    Args:
        directory: Directory path

    Returns:
        True if successful
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False
