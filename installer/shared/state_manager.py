"""
Installation State Manager

Manages persistent state for installer with auto-recovery capabilities.
Handles state save/load, checkpoint management, and recovery detection.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from .state_schema import (
    InstallationState,
    InstallationStep,
    FeaturesConfig,
    InstallConfig,
    RollbackPoint,
    DEFAULT_STATE,
    INSTALLATION_STEPS,
    ROLLBACK_STEPS
)


logger = logging.getLogger("StateManager")


class StateManager:
    """
    Manages installation state persistence and recovery

    Features:
    - JSON-based state persistence
    - Checkpoint/rollback point tracking
    - Recovery detection on startup
    - Thread-safe state updates
    """

    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize state manager

        Args:
            state_file: Path to state file (default: ~/.dictator/installer_state.json)
        """
        if state_file is None:
            # Default location: ~/.dictator/installer_state.json
            state_dir = Path.home() / ".dictator"
            state_dir.mkdir(parents=True, exist_ok=True)
            state_file = state_dir / "installer_state.json"

        self.state_file = state_file
        self._state: InstallationState = DEFAULT_STATE.copy()

        # Try to load existing state
        if self.state_file.exists():
            self.load()
        else:
            # Initialize new state
            self._initialize_new_state()

    def _initialize_new_state(self):
        """Initialize a new installation state"""
        self._state = DEFAULT_STATE.copy()
        self._state["install_id"] = str(uuid.uuid4())
        self._state["timestamp"] = datetime.now().isoformat()
        self._state["completed_steps"] = []
        self._state["failed_steps"] = []
        self._state["rollback_points"] = []
        logger.info(f"Initialized new installation state: {self._state['install_id']}")

    def save(self) -> bool:
        """
        Save current state to disk

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update timestamp
            self._state["timestamp"] = datetime.now().isoformat()

            # Write to file with pretty formatting
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)

            logger.debug(f"State saved to {self.state_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def cleanup(self) -> bool:
        """
        Remove state file after successful installation

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                logger.info(f"State file removed: {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove state file: {e}")
            return False

    def reset_state(self) -> None:
        """
        Reset state for clean installation

        Removes existing state file and initializes fresh state.
        Use this to ensure a clean installation without old state interference.
        """
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                logger.info(f"Removed old state file: {self.state_file}")
        except Exception as e:
            logger.warning(f"Failed to remove old state: {e}")

        # Initialize fresh state
        self._initialize_new_state()
        logger.info("Initialized fresh installation state")

    def load(self) -> bool:
        """
        Load state from disk

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)

            # Validate loaded state has required fields
            if "version" in loaded_state and "install_id" in loaded_state:
                self._state = loaded_state
                logger.info(f"Loaded existing state: {self._state['install_id']}")
                return True
            else:
                logger.warning("Invalid state file, initializing new state")
                self._initialize_new_state()
                return False

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in state file: {e}")
            self._initialize_new_state()
            return False

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self._initialize_new_state()
            return False

    def clear(self):
        """Clear current state and delete state file"""
        self._initialize_new_state()
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("State file deleted")

    # State accessors

    @property
    def state(self) -> InstallationState:
        """Get current state (read-only copy)"""
        return self._state.copy()

    @property
    def install_id(self) -> str:
        """Get installation ID"""
        return self._state["install_id"]

    @property
    def current_step(self) -> Optional[InstallationStep]:
        """Get current installation step"""
        return self._state.get("current_step")

    @property
    def completed_steps(self) -> list[InstallationStep]:
        """Get list of completed steps"""
        return self._state["completed_steps"].copy()

    @property
    def failed_steps(self) -> list[InstallationStep]:
        """Get list of failed steps"""
        return self._state["failed_steps"].copy()

    @property
    def config(self) -> InstallConfig:
        """Get installation configuration"""
        return self._state["config"].copy()

    @property
    def features(self) -> FeaturesConfig:
        """Get features configuration"""
        return self._state["config"]["features"].copy()

    # State modifiers

    def set_current_step(self, step: InstallationStep):
        """
        Set current installation step

        Args:
            step: Installation step to set as current
        """
        if step not in INSTALLATION_STEPS:
            raise ValueError(f"Invalid installation step: {step}")

        self._state["current_step"] = step
        logger.info(f"Current step set to: {step}")
        self.save()

    def mark_step_completed(self, step: InstallationStep):
        """
        Mark a step as completed

        Args:
            step: Installation step to mark as completed
        """
        if step not in INSTALLATION_STEPS:
            raise ValueError(f"Invalid installation step: {step}")

        if step not in self._state["completed_steps"]:
            self._state["completed_steps"].append(step)
            logger.info(f"Step completed: {step}")

        # Clear current step if it matches
        if self._state["current_step"] == step:
            self._state["current_step"] = None

        self.save()

    def mark_step_failed(self, step: InstallationStep):
        """
        Mark a step as failed

        Args:
            step: Installation step to mark as failed
        """
        if step not in INSTALLATION_STEPS:
            raise ValueError(f"Invalid installation step: {step}")

        if step not in self._state["failed_steps"]:
            self._state["failed_steps"].append(step)
            logger.error(f"Step failed: {step}")

        self.save()

    def update_config(self, config_updates: dict):
        """
        Update installation configuration

        Args:
            config_updates: Dictionary of config updates to merge
        """
        # Deep merge config updates
        for key, value in config_updates.items():
            if key in self._state["config"]:
                if isinstance(value, dict) and isinstance(self._state["config"][key], dict):
                    self._state["config"][key].update(value)
                else:
                    self._state["config"][key] = value

        logger.debug(f"Config updated: {config_updates}")
        self.save()

    def update_features(self, features: dict):
        """
        Update feature flags

        Args:
            features: Dictionary of feature flag updates
        """
        self._state["config"]["features"].update(features)
        logger.debug(f"Features updated: {features}")
        self.save()

    # Rollback point management

    def create_rollback_point(self, step: InstallationStep, actions: list[dict]):
        """
        Create a rollback checkpoint

        Args:
            step: Current installation step
            actions: List of actions to undo (for rollback)
        """
        if step not in ROLLBACK_STEPS:
            logger.debug(f"Step {step} does not create rollback point")
            return

        rollback_point: RollbackPoint = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "actions": actions
        }

        self._state["rollback_points"].append(rollback_point)
        logger.info(f"Rollback point created for step: {step}")
        self.save()

    def get_latest_rollback_point(self) -> Optional[RollbackPoint]:
        """
        Get the most recent rollback point

        Returns:
            Latest rollback point or None if no rollback points exist
        """
        if self._state["rollback_points"]:
            return self._state["rollback_points"][-1]
        return None

    def pop_rollback_point(self) -> Optional[RollbackPoint]:
        """
        Remove and return the latest rollback point

        Returns:
            Latest rollback point or None if no rollback points exist
        """
        if self._state["rollback_points"]:
            point = self._state["rollback_points"].pop()
            self.save()
            logger.info(f"Rollback point popped: {point['step']}")
            return point
        return None

    def clear_rollback_points(self):
        """Clear all rollback points"""
        self._state["rollback_points"] = []
        logger.info("All rollback points cleared")
        self.save()

    # Recovery detection

    def is_partial_installation(self) -> bool:
        """
        Check if this is a partial/incomplete installation

        Returns:
            True if installation was started but not completed
        """
        # Installation is partial if:
        # 1. Some steps are completed
        # 2. Not all steps are completed
        # 3. Or there are failed steps
        has_completed = len(self._state["completed_steps"]) > 0
        all_completed = len(self._state["completed_steps"]) == len(INSTALLATION_STEPS)
        has_failures = len(self._state["failed_steps"]) > 0

        return (has_completed and not all_completed) or has_failures

    def is_fresh_installation(self) -> bool:
        """
        Check if this is a fresh installation (no prior state)

        Returns:
            True if no steps have been completed
        """
        return len(self._state["completed_steps"]) == 0

    def is_installation_complete(self) -> bool:
        """
        Check if installation is complete

        Returns:
            True if all steps are completed
        """
        return len(self._state["completed_steps"]) == len(INSTALLATION_STEPS)

    def get_next_step(self) -> Optional[InstallationStep]:
        """
        Get the next installation step to execute

        Returns:
            Next step or None if installation is complete
        """
        for step in INSTALLATION_STEPS:
            if step not in self._state["completed_steps"]:
                return step
        return None

    def get_recovery_options(self) -> dict:
        """
        Get available recovery options based on current state

        Returns:
            Dictionary of recovery options with descriptions
        """
        options = {}

        if self.is_partial_installation():
            options["resume"] = {
                "label": "Resume Installation",
                "description": f"Continue from step: {self.get_next_step()}",
                "available": True
            }

            if self._state["failed_steps"]:
                options["retry"] = {
                    "label": "Retry Failed Step",
                    "description": f"Retry: {self._state['failed_steps'][-1]}",
                    "available": True
                }

            if self._state["rollback_points"]:
                latest = self.get_latest_rollback_point()
                options["rollback"] = {
                    "label": "Rollback",
                    "description": f"Undo to: {latest['step']}",
                    "available": True
                }

        options["clean_install"] = {
            "label": "Clean Install",
            "description": "Remove all traces and start fresh",
            "available": True
        }

        options["cancel"] = {
            "label": "Cancel",
            "description": "Exit installer",
            "available": True
        }

        return options
