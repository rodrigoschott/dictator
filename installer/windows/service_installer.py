"""
Service Installer

Windows service installation and management using NSSM.
"""

import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger("ServiceInstaller")


class ServiceInstaller:
    """
    Windows service installer using NSSM

    Handles:
    - NSSM installation/embedding
    - Service creation and configuration
    - Service start/stop/restart
    - Service removal
    """

    def __init__(
        self,
        service_name: str = "Dictator",
        nssm_path: Optional[Path] = None
    ):
        """
        Initialize service installer

        Args:
            service_name: Name of the Windows service
            nssm_path: Path to nssm.exe (if None, will look in assets/)
        """
        self.service_name = service_name

        # Find NSSM executable
        if nssm_path is None:
            # Look in assets directory
            assets_dir = Path(__file__).parent / "assets"
            nssm_path = assets_dir / "nssm.exe"

        self.nssm_path = nssm_path

        if not self.nssm_path.exists():
            logger.warning(f"NSSM not found at: {self.nssm_path}")

    def _run_nssm(self, args: list, timeout: int = 30) -> Tuple[bool, str]:
        """
        Run NSSM command

        Args:
            args: Command arguments
            timeout: Timeout in seconds

        Returns:
            (success, output)
        """
        try:
            command = [str(self.nssm_path)] + args
            logger.debug(f"Running: {' '.join(command)}")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout.strip() or result.stderr.strip()

            if result.returncode == 0:
                logger.debug(f"NSSM command succeeded: {output}")
                return True, output
            else:
                logger.error(f"NSSM command failed (exit {result.returncode}): {output}")
                return False, output

        except subprocess.TimeoutExpired:
            logger.error(f"NSSM command timed out after {timeout}s")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"Failed to run NSSM: {e}")
            return False, str(e)

    def _run_sc(self, args: list, timeout: int = 10) -> Tuple[bool, str]:
        """
        Run Windows sc.exe command

        Args:
            args: Command arguments
            timeout: Timeout in seconds

        Returns:
            (success, output)
        """
        try:
            command = ["sc"] + args
            logger.debug(f"Running: {' '.join(command)}")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout.strip() or result.stderr.strip()
            return result.returncode == 0, output

        except Exception as e:
            logger.error(f"Failed to run sc: {e}")
            return False, str(e)

    def check_service_exists(self) -> bool:
        """
        Check if service already exists

        Returns:
            True if service exists
        """
        success, output = self._run_sc(["query", self.service_name])
        return success

    def install_service(
        self,
        python_exe: Path,
        script_path: Path,
        config_path: Path,
        log_dir: Path,
        description: str = "Dictator Voice to Text Service"
    ) -> Tuple[bool, str]:
        """
        Install Windows service

        Args:
            python_exe: Path to Python executable (in venv)
            script_path: Path to tray.py or service.py
            config_path: Path to config.yaml
            log_dir: Directory for service logs
            description: Service description

        Returns:
            (success, message)
        """
        logger.info(f"Installing service: {self.service_name}")

        # Check if service already exists
        if self.check_service_exists():
            logger.warning(f"Service {self.service_name} already exists, removing first")
            self.remove_service()

        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)

        # Install service
        success, output = self._run_nssm([
            "install",
            self.service_name,
            str(python_exe),
            str(script_path),
            str(config_path)
        ])

        if not success:
            return False, f"Failed to install service: {output}"

        logger.info(f"Service installed, configuring parameters...")

        # Set service description
        self._run_nssm(["set", self.service_name, "Description", description])

        # Set display name
        self._run_nssm(["set", self.service_name, "DisplayName", self.service_name])

        # Set startup type to automatic
        self._run_nssm(["set", self.service_name, "Start", "SERVICE_AUTO_START"])

        # Set stdout log
        stdout_log = log_dir / "service.log"
        self._run_nssm(["set", self.service_name, "AppStdout", str(stdout_log)])

        # Set stderr log
        stderr_log = log_dir / "service_error.log"
        self._run_nssm(["set", self.service_name, "AppStderr", str(stderr_log)])

        # Set environment variable
        self._run_nssm(["set", self.service_name, "AppEnvironmentExtra", "PYTHONUNBUFFERED=1"])

        # Set restart behavior
        self._run_nssm(["set", self.service_name, "AppExit", "Default", "Restart"])

        # Set restart delay (5 seconds)
        self._run_nssm(["set", self.service_name, "AppRestartDelay", "5000"])

        logger.info(f"✓ Service {self.service_name} installed successfully")
        return True, "Service installed"

    def start_service(self) -> Tuple[bool, str]:
        """
        Start the service

        Returns:
            (success, message)
        """
        logger.info(f"Starting service: {self.service_name}")

        success, output = self._run_nssm(["start", self.service_name])

        if success:
            # Wait a bit for service to start
            time.sleep(2)

            # Verify service is running
            if self.is_service_running():
                logger.info(f"✓ Service {self.service_name} started")
                return True, "Service started"
            else:
                logger.error(f"Service started but not running")
                return False, "Service failed to start properly"
        else:
            return False, f"Failed to start service: {output}"

    def stop_service(self) -> Tuple[bool, str]:
        """
        Stop the service

        Returns:
            (success, message)
        """
        logger.info(f"Stopping service: {self.service_name}")

        success, output = self._run_nssm(["stop", self.service_name])

        if success:
            logger.info(f"✓ Service {self.service_name} stopped")
            return True, "Service stopped"
        else:
            return False, f"Failed to stop service: {output}"

    def restart_service(self) -> Tuple[bool, str]:
        """
        Restart the service

        Returns:
            (success, message)
        """
        logger.info(f"Restarting service: {self.service_name}")

        success, output = self._run_nssm(["restart", self.service_name])

        if success:
            logger.info(f"✓ Service {self.service_name} restarted")
            return True, "Service restarted"
        else:
            return False, f"Failed to restart service: {output}"

    def remove_service(self) -> Tuple[bool, str]:
        """
        Remove the service

        Returns:
            (success, message)
        """
        logger.info(f"Removing service: {self.service_name}")

        # Stop service first
        if self.is_service_running():
            self.stop_service()
            time.sleep(2)

        # Remove service
        success, output = self._run_nssm(["remove", self.service_name, "confirm"])

        if success:
            logger.info(f"✓ Service {self.service_name} removed")
            return True, "Service removed"
        else:
            # Check if service is actually gone
            if not self.check_service_exists():
                logger.info(f"Service {self.service_name} not found (already removed)")
                return True, "Service removed"

            return False, f"Failed to remove service: {output}"

    def is_service_running(self) -> bool:
        """
        Check if service is running

        Returns:
            True if service is running
        """
        success, output = self._run_sc(["query", self.service_name])

        if not success:
            return False

        # Check for "RUNNING" state in output
        return "RUNNING" in output.upper()

    def get_service_status(self) -> str:
        """
        Get service status

        Returns:
            Status string ("running", "stopped", "not_installed", "unknown")
        """
        if not self.check_service_exists():
            return "not_installed"

        success, output = self._run_sc(["query", self.service_name])

        if not success:
            return "unknown"

        output_upper = output.upper()

        if "RUNNING" in output_upper:
            return "running"
        elif "STOPPED" in output_upper:
            return "stopped"
        elif "PAUSED" in output_upper:
            return "paused"
        else:
            return "unknown"

    def set_service_user(self, username: str, password: str = "") -> Tuple[bool, str]:
        """
        Set service to run as specific user

        Args:
            username: Windows username (e.g., ".\\username")
            password: User password (empty for current user)

        Returns:
            (success, message)
        """
        logger.info(f"Setting service user to: {username}")

        success, output = self._run_nssm([
            "set",
            self.service_name,
            "ObjectName",
            username,
            password
        ])

        if success:
            logger.info(f"✓ Service user set to {username}")
            return True, f"User set to {username}"
        else:
            return False, f"Failed to set user: {output}"

    def enable_desktop_interaction(self) -> Tuple[bool, str]:
        """
        Enable desktop interaction for service (required for system tray)

        Returns:
            (success, message)
        """
        logger.info("Enabling desktop interaction for service")

        # Run as LocalSystem with desktop interaction enabled via registry
        # This avoids password requirement while still allowing system tray
        success, message = self.set_service_user("LocalSystem", "")

        if success:
            logger.info("✓ Service set to run as LocalSystem")

            # Enable desktop interaction via registry
            try:
                import winreg
                key_path = r"SYSTEM\CurrentControlSet\Services\Dictator"
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE)
                # Set SERVICE_INTERACTIVE_PROCESS flag (0x100)
                winreg.SetValueEx(key, "Type", 0, winreg.REG_DWORD, 0x110)
                winreg.CloseKey(key)
                logger.info("✓ Desktop interaction enabled via registry")
                return True, "Desktop interaction enabled"
            except Exception as e:
                logger.warning(f"Failed to enable desktop interaction: {e}")
                # Not critical - service will still work
                return True, "Service configured (desktop interaction unavailable)"
        else:
            return False, f"Failed to set service user: {message}"

    def get_service_info(self) -> dict:
        """
        Get comprehensive service information

        Returns:
            Dictionary with service details
        """
        info = {
            "name": self.service_name,
            "exists": self.check_service_exists(),
            "status": self.get_service_status(),
            "running": self.is_service_running()
        }

        return info
