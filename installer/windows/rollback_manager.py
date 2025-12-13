"""
Rollback Manager

Handles rollback operations for failed installation steps.
Each installation step has a corresponding rollback handler.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Callable, Optional
import logging

logger = logging.getLogger("RollbackManager")


class RollbackAction:
    """Base class for rollback actions"""

    def __init__(self, description: str):
        self.description = description

    def execute(self) -> bool:
        """
        Execute the rollback action

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serialize to dictionary for state storage"""
        return {
            "type": self.__class__.__name__,
            "description": self.description
        }


class DeleteDirectoryAction(RollbackAction):
    """Delete a directory"""

    def __init__(self, directory: Path, description: str = ""):
        super().__init__(description or f"Delete directory: {directory}")
        self.directory = directory

    def execute(self) -> bool:
        try:
            if self.directory.exists():
                shutil.rmtree(self.directory)
                logger.info(f"✓ Deleted directory: {self.directory}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to delete directory {self.directory}: {e}")
            return False

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["directory"] = str(self.directory)
        return data


class DeleteFileAction(RollbackAction):
    """Delete a file"""

    def __init__(self, file_path: Path, description: str = ""):
        super().__init__(description or f"Delete file: {file_path}")
        self.file_path = file_path

    def execute(self) -> bool:
        try:
            if self.file_path.exists():
                self.file_path.unlink()
                logger.info(f"✓ Deleted file: {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to delete file {self.file_path}: {e}")
            return False

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["file_path"] = str(self.file_path)
        return data


class UninstallServiceAction(RollbackAction):
    """Uninstall Windows service via NSSM"""

    def __init__(self, service_name: str, nssm_path: Path, description: str = ""):
        super().__init__(description or f"Uninstall service: {service_name}")
        self.service_name = service_name
        self.nssm_path = nssm_path

    def execute(self) -> bool:
        try:
            # Stop service first
            subprocess.run(
                [str(self.nssm_path), "stop", self.service_name],
                capture_output=True,
                timeout=30
            )

            # Remove service
            result = subprocess.run(
                [str(self.nssm_path), "remove", self.service_name, "confirm"],
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info(f"✓ Uninstalled service: {self.service_name}")
                return True
            else:
                logger.warning(f"Service {self.service_name} may not exist: {result.stderr.decode()}")
                return True  # Not a critical failure

        except Exception as e:
            logger.error(f"✗ Failed to uninstall service {self.service_name}: {e}")
            return False

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["service_name"] = self.service_name
        data["nssm_path"] = str(self.nssm_path)
        return data


class RunCommandAction(RollbackAction):
    """Run an arbitrary command"""

    def __init__(self, command: List[str], description: str = ""):
        super().__init__(description or f"Run command: {' '.join(command)}")
        self.command = command

    def execute(self) -> bool:
        try:
            result = subprocess.run(
                self.command,
                capture_output=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info(f"✓ Command executed: {' '.join(self.command)}")
                return True
            else:
                logger.error(f"✗ Command failed: {result.stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"✗ Failed to run command: {e}")
            return False

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["command"] = self.command
        return data


class RollbackManager:
    """
    Manages rollback operations for installation

    Tracks actions performed during installation and can undo them
    in reverse order (LIFO stack).
    """

    def __init__(self):
        """Initialize rollback manager"""
        self.actions: List[RollbackAction] = []

    def add_action(self, action: RollbackAction):
        """
        Add a rollback action

        Args:
            action: RollbackAction to add to the stack
        """
        self.actions.append(action)
        logger.debug(f"Added rollback action: {action.description}")

    def execute_rollback(self) -> bool:
        """
        Execute all rollback actions in reverse order

        Returns:
            True if all actions successful, False if any failed
        """
        if not self.actions:
            logger.info("No rollback actions to execute")
            return True

        logger.info(f"Executing rollback: {len(self.actions)} action(s)")

        all_success = True
        # Execute in reverse order (LIFO)
        for action in reversed(self.actions):
            logger.info(f"Rollback: {action.description}")
            success = action.execute()
            if not success:
                all_success = False

        if all_success:
            logger.info("✓ Rollback completed successfully")
        else:
            logger.error("✗ Rollback completed with errors")

        # Clear actions after rollback
        self.actions.clear()

        return all_success

    def clear(self):
        """Clear all rollback actions"""
        self.actions.clear()
        logger.debug("Rollback actions cleared")

    def get_actions(self) -> List[Dict]:
        """
        Get list of current rollback actions

        Returns:
            List of action dictionaries
        """
        return [action.to_dict() for action in self.actions]

    def count(self) -> int:
        """Get number of rollback actions"""
        return len(self.actions)


# Convenience functions for common rollback scenarios

def create_directory_rollback(directory: Path) -> RollbackAction:
    """
    Create rollback action for directory creation

    Args:
        directory: Directory that was created

    Returns:
        DeleteDirectoryAction
    """
    return DeleteDirectoryAction(directory, f"Remove created directory: {directory.name}")


def create_venv_rollback(venv_dir: Path) -> RollbackAction:
    """
    Create rollback action for virtual environment creation

    Args:
        venv_dir: Virtual environment directory

    Returns:
        DeleteDirectoryAction
    """
    return DeleteDirectoryAction(venv_dir, f"Remove virtual environment: {venv_dir.name}")


def create_file_rollback(file_path: Path) -> RollbackAction:
    """
    Create rollback action for file creation

    Args:
        file_path: File that was created

    Returns:
        DeleteFileAction
    """
    return DeleteFileAction(file_path, f"Remove created file: {file_path.name}")


def create_service_rollback(service_name: str, nssm_path: Path) -> RollbackAction:
    """
    Create rollback action for service installation

    Args:
        service_name: Name of the Windows service
        nssm_path: Path to nssm.exe

    Returns:
        UninstallServiceAction
    """
    return UninstallServiceAction(service_name, nssm_path, f"Uninstall service: {service_name}")


def create_model_download_rollback(model_files: List[Path]) -> List[RollbackAction]:
    """
    Create rollback actions for downloaded model files

    Args:
        model_files: List of downloaded model file paths

    Returns:
        List of DeleteFileAction for each model
    """
    actions = []
    for model_file in model_files:
        actions.append(DeleteFileAction(model_file, f"Remove downloaded model: {model_file.name}"))
    return actions
