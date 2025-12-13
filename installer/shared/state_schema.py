"""
Installation State Schema

Defines the JSON schema for installer state tracking and recovery.
"""

from typing import TypedDict, List, Literal
from datetime import datetime


# Type definitions for state schema
InstallationStep = Literal[
    "pre_flight_check",
    "select_features",
    "create_directories",
    "copy_source_code",
    "create_venv",
    "install_dependencies",
    "download_models",
    "verify_models",
    "generate_config",
    "install_service",
    "copy_launcher",
    "create_shortcuts",
    "final_validation",
    "start_service"
]


class FeaturesConfig(TypedDict):
    """Feature flags for installation"""
    gpu_enabled: bool
    tts_enabled: bool
    llm_enabled: bool
    whisper_model: str  # tiny, small, medium, large, large-v3
    install_as_service: bool
    auto_start_service: bool


class InstallConfig(TypedDict):
    """Installation configuration"""
    install_dir: str
    venv_dir: str
    config_dir: str
    log_dir: str
    features: FeaturesConfig


class RollbackPoint(TypedDict):
    """Rollback checkpoint data"""
    step: InstallationStep
    timestamp: str
    actions: List[dict]  # List of actions to undo


class InstallationState(TypedDict):
    """Complete installation state"""
    version: str
    install_id: str
    timestamp: str
    current_step: InstallationStep | None
    completed_steps: List[InstallationStep]
    failed_steps: List[InstallationStep]
    config: InstallConfig
    rollback_points: List[RollbackPoint]


# Default state
DEFAULT_STATE: InstallationState = {
    "version": "1.0.0",
    "install_id": "",
    "timestamp": "",
    "current_step": None,
    "completed_steps": [],
    "failed_steps": [],
    "config": {
        "install_dir": "",
        "venv_dir": "",
        "config_dir": "",
        "log_dir": "",
        "features": {
            "gpu_enabled": True,
            "tts_enabled": True,
            "llm_enabled": False,
            "whisper_model": "large-v3",
            "install_as_service": True,
            "auto_start_service": True
        }
    },
    "rollback_points": []
}


# Installation step order (for validation)
INSTALLATION_STEPS: List[InstallationStep] = [
    "pre_flight_check",
    "select_features",
    "create_directories",
    "copy_source_code",  # Copy Dictator source files
    "create_venv",
    "install_dependencies",
    "download_models",
    "verify_models",
    "generate_config",
    "install_service",
    "copy_launcher",  # Copy Dictator.exe launcher
    "create_shortcuts",  # Create Desktop and Start Menu shortcuts
    "final_validation",
    "start_service"
]


# Steps that create rollback points
ROLLBACK_STEPS: List[InstallationStep] = [
    "create_directories",
    "create_venv",
    "install_dependencies",
    "download_models",
    "generate_config",
    "install_service"
]
