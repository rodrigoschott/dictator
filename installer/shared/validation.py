"""
Shared Validation Utilities

Cross-platform validation functions for installer pre-flight checks.
"""

import platform
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger("Validation")


class SystemRequirements:
    """System requirements validation"""

    # Minimum requirements
    MIN_PYTHON_VERSION = (3, 10)
    MAX_PYTHON_VERSION = (3, 14)  # Exclusive - matches pyproject.toml python = ">=3.10,<3.14"
    MIN_RAM_GB = 8
    MIN_DISK_GB = 5
    RECOMMENDED_RAM_GB = 16
    RECOMMENDED_DISK_GB = 15

    @staticmethod
    def check_os() -> Tuple[bool, str, dict]:
        """
        Check operating system compatibility

        Returns:
            (passed, message, details)
        """
        os_name = platform.system()
        os_version = platform.release()

        details = {
            "os": os_name,
            "version": os_version,
            "platform": platform.platform()
        }

        if os_name == "Windows":
            # Windows 10+ required
            try:
                version_num = int(os_version.split('.')[0])
                if version_num >= 10:
                    return True, f"[OK] {os_name} {os_version}", details
                else:
                    return False, f"[ERRO] Windows 10 or later required (found {os_version})", details
            except:
                return False, f"[ERRO] Could not determine Windows version", details

        elif os_name == "Linux":
            return True, f"[OK] {os_name} {os_version}", details

        else:
            return False, f"[ERRO] Unsupported OS: {os_name}", details

    @staticmethod
    def check_python_version() -> Tuple[bool, str, dict]:
        """
        Check Python version compatibility

        Returns:
            (passed, message, details)
        """
        current = sys.version_info[:2]
        min_ver = SystemRequirements.MIN_PYTHON_VERSION
        max_ver = SystemRequirements.MAX_PYTHON_VERSION

        details = {
            "version": f"{current[0]}.{current[1]}",
            "full_version": sys.version,
            "executable": sys.executable
        }

        if min_ver <= current < max_ver:
            return True, f"[OK] Python {current[0]}.{current[1]}", details
        elif current < min_ver:
            return False, f"[ERRO] Python {min_ver[0]}.{min_ver[1]}+ required (found {current[0]}.{current[1]})", details
        else:
            return False, f"[ERRO] Python < {max_ver[0]}.{max_ver[1]} required (found {current[0]}.{current[1]})", details

    @staticmethod
    def check_ram() -> Tuple[bool, str, dict]:
        """
        Check available RAM

        Returns:
            (passed, message, details)
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)

            details = {
                "total_gb": round(total_gb, 2),
                "available_gb": round(mem.available / (1024 ** 3), 2),
                "percent_used": mem.percent
            }

            if total_gb >= SystemRequirements.RECOMMENDED_RAM_GB:
                return True, f"[OK] {total_gb:.1f} GB RAM (recommended)", details
            elif total_gb >= SystemRequirements.MIN_RAM_GB:
                return True, f"[WARN] {total_gb:.1f} GB RAM (minimum met, {SystemRequirements.RECOMMENDED_RAM_GB}GB recommended)", details
            else:
                return False, f"[ERRO] {total_gb:.1f} GB RAM (minimum {SystemRequirements.MIN_RAM_GB}GB required)", details

        except ImportError:
            return True, "[WARN] Could not check RAM (psutil not available)", {}
        except Exception as e:
            return True, f"[WARN] Could not check RAM: {e}", {}

    @staticmethod
    def check_disk_space(path: Path) -> Tuple[bool, str, dict]:
        """
        Check available disk space

        Args:
            path: Installation directory path

        Returns:
            (passed, message, details)
        """
        try:
            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            usage = shutil.disk_usage(path.parent)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)

            details = {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round((usage.used / usage.total) * 100, 1)
            }

            if free_gb >= SystemRequirements.RECOMMENDED_DISK_GB:
                return True, f"[OK] {free_gb:.1f} GB free (recommended)", details
            elif free_gb >= SystemRequirements.MIN_DISK_GB:
                return True, f"[WARN] {free_gb:.1f} GB free (minimum met, {SystemRequirements.RECOMMENDED_DISK_GB}GB recommended)", details
            else:
                return False, f"[ERRO] {free_gb:.1f} GB free (minimum {SystemRequirements.MIN_DISK_GB}GB required)", details

        except Exception as e:
            return True, f"[WARN] Could not check disk space: {e}", {}

    @staticmethod
    def check_network_connectivity(timeout: int = 5) -> Tuple[bool, str, dict]:
        """
        Check internet connectivity

        Args:
            timeout: Timeout in seconds

        Returns:
            (passed, message, details)
        """
        try:
            import requests
            response = requests.get("https://www.google.com", timeout=timeout)

            details = {
                "status_code": response.status_code,
                "response_time_ms": int(response.elapsed.total_seconds() * 1000)
            }

            if response.status_code == 200:
                return True, f"[OK] Network connected ({details['response_time_ms']}ms)", details
            else:
                return False, f"[WARN] Network issue (status {response.status_code})", details

        except ImportError:
            return True, "[WARN] Could not check network (requests not available)", {}
        except Exception as e:
            return False, f"[ERRO] No network connection: {e}", {}

    @staticmethod
    def check_gpu_availability() -> Tuple[bool, str, dict]:
        """
        Check NVIDIA GPU and CUDA availability

        Returns:
            (passed, message, details)
        """
        details = {"gpu_available": False, "cuda_available": False}

        try:
            # Try to detect CUDA via nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_info = lines[0].split(',')
                    details["gpu_available"] = True
                    details["gpu_name"] = gpu_info[0].strip() if len(gpu_info) > 0 else "Unknown"
                    details["driver_version"] = gpu_info[1].strip() if len(gpu_info) > 1 else "Unknown"
                    details["vram_total"] = gpu_info[2].strip() if len(gpu_info) > 2 else "Unknown"

                    # Check if torch.cuda is available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            details["cuda_available"] = True
                            details["cuda_version"] = torch.version.cuda
                            return True, f"[OK] {details['gpu_name']} (CUDA {details['cuda_version']})", details
                        else:
                            return True, f"[WARN] {details['gpu_name']} detected but CUDA not available (will use CPU)", details
                    except ImportError:
                        return True, f"[WARN] {details['gpu_name']} detected (CUDA check requires torch)", details

        except FileNotFoundError:
            pass  # nvidia-smi not found
        except Exception as e:
            logger.debug(f"GPU check failed: {e}")

        # No GPU found
        return True, "[WARN] No NVIDIA GPU detected (CPU mode available)", details

    @staticmethod
    def check_admin_rights() -> Tuple[bool, str, dict]:
        """
        Check if running with administrator/root privileges

        Returns:
            (passed, message, details)
        """
        import os

        details = {}

        if platform.system() == "Windows":
            try:
                import ctypes
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
                details["admin"] = is_admin

                if is_admin:
                    return True, "[OK] Running as Administrator", details
                else:
                    return False, "[ERRO] Administrator rights required", details

            except Exception as e:
                return False, f"[ERRO] Could not check admin rights: {e}", details

        else:  # Linux/Unix
            is_root = os.geteuid() == 0
            details["root"] = is_root

            if is_root:
                return True, "[OK] Running as root", details
            else:
                return False, "[ERRO] Root privileges required", details


def validate_path_writable(path: Path) -> Tuple[bool, str]:
    """
    Check if a path is writable

    Args:
        path: Path to check

    Returns:
        (writable, error_message)
    """
    try:
        # Try to create parent directory
        path.parent.mkdir(parents=True, exist_ok=True)

        # Try to create a test file
        test_file = path.parent / ".dictator_write_test"
        test_file.write_text("test")
        test_file.unlink()

        return True, ""

    except PermissionError:
        return False, f"Permission denied: {path.parent}"
    except Exception as e:
        return False, f"Cannot write to {path.parent}: {e}"


def find_python_executable() -> Optional[Path]:
    """
    Find Python executable path

    Returns:
        Path to Python executable or None if not found
    """
    return Path(sys.executable)
