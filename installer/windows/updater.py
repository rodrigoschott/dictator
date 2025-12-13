"""
Installation Updater

Handles update and repair operations for existing Dictator installations.
"""

import shutil
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger("Updater")


class ConfigMerger:
    """Merge existing config with new defaults while preserving user customizations"""

    @staticmethod
    def backup_config(config_path: Path) -> Optional[Path]:
        """
        Create backup of existing config file

        Args:
            config_path: Path to config.yaml

        Returns:
            Path to backup file, or None if backup failed
        """
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return None

        try:
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_path.parent / f"config.yaml.backup.{timestamp}"

            shutil.copy2(config_path, backup_path)
            logger.info(f"Config backed up to: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to backup config: {e}")
            return None

    @staticmethod
    def load_yaml(path: Path) -> Optional[Dict[str, Any]]:
        """Load YAML file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML from {path}: {e}")
            return None

    @staticmethod
    def save_yaml(path: Path, data: Dict[str, Any]) -> bool:
        """Save YAML file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Config saved to: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save YAML to {path}: {e}")
            return False

    @staticmethod
    def merge_configs(existing: Dict[str, Any], new_defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge existing config with new defaults

        Strategy:
        - Preserve all user-customized values
        - Add new keys from defaults if missing
        - Remove obsolete keys (not in new defaults)

        Args:
            existing: User's existing config
            new_defaults: New default config

        Returns:
            Merged config
        """
        merged = {}

        # Iterate through new defaults (this is the authoritative structure)
        for key, default_value in new_defaults.items():
            if key not in existing:
                # New key - use default
                merged[key] = default_value
                logger.info(f"Added new config key: {key}")
            elif isinstance(default_value, dict) and isinstance(existing[key], dict):
                # Both are dicts - recurse
                merged[key] = ConfigMerger.merge_configs(existing[key], default_value)
            else:
                # Use existing value (preserve user customization)
                merged[key] = existing[key]

        # Warn about obsolete keys (exist in old but not in new)
        obsolete_keys = set(existing.keys()) - set(new_defaults.keys())
        if obsolete_keys:
            logger.warning(f"Obsolete config keys removed: {obsolete_keys}")

        return merged


class InstallationUpdater:
    """Handles update and repair operations"""

    def __init__(self, install_dir: Path):
        """
        Initialize updater

        Args:
            install_dir: Installation directory
        """
        self.install_dir = install_dir
        self.config_path = install_dir / "config" / "config.yaml"

    def update_installation(self, source_dir: Path, new_config_defaults: Dict[str, Any]) -> bool:
        """
        Update existing installation

        Steps:
        1. Backup config.yaml
        2. Update src/dictator/ (replace files)
        3. Merge config (preserve customizations)
        4. Verify integrity

        Args:
            source_dir: Directory containing new source code (src/dictator/)
            new_config_defaults: New default config to merge with

        Returns:
            True if successful
        """
        logger.info(f"Updating installation at: {self.install_dir}")

        try:
            # Step 1: Backup config
            backup_path = ConfigMerger.backup_config(self.config_path)
            if backup_path is None and self.config_path.exists():
                logger.error("Failed to backup config - aborting update")
                return False

            # Step 2: Update source code
            target_src = self.install_dir / "src" / "dictator"
            if target_src.exists():
                logger.info(f"Removing old source: {target_src}")
                shutil.rmtree(target_src)

            logger.info(f"Copying new source from: {source_dir}")
            shutil.copytree(source_dir, target_src)

            # Step 3: Merge config
            if self.config_path.exists():
                existing_config = ConfigMerger.load_yaml(self.config_path)
                if existing_config is None:
                    logger.warning("Failed to load existing config - using defaults")
                    merged_config = new_config_defaults
                else:
                    merged_config = ConfigMerger.merge_configs(existing_config, new_config_defaults)

                ConfigMerger.save_yaml(self.config_path, merged_config)
            else:
                # No existing config - use defaults
                logger.info("No existing config - creating from defaults")
                ConfigMerger.save_yaml(self.config_path, new_config_defaults)

            # Step 4: Verify
            if not self._verify_installation():
                logger.error("Installation verification failed after update")
                return False

            logger.info("Update completed successfully")
            return True

        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    def repair_installation(self, source_dir: Path, config_defaults: Dict[str, Any]) -> bool:
        """
        Repair existing installation

        Steps:
        1. Check what's missing
        2. Recreate venv if corrupted
        3. Restore source if missing
        4. Restore config if missing
        5. Verify integrity

        Args:
            source_dir: Directory containing source code
            config_defaults: Default config

        Returns:
            True if successful
        """
        logger.info(f"Repairing installation at: {self.install_dir}")

        try:
            # Check venv
            venv_python = self.install_dir / "venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                logger.warning("Virtual environment missing or corrupted")
                # TODO: Recreate venv (needs to be done by installer_core)

            # Check source
            target_src = self.install_dir / "src" / "dictator"
            service_py = target_src / "service.py"
            if not service_py.exists():
                logger.warning("Source code missing - restoring")
                target_src.parent.mkdir(parents=True, exist_ok=True)
                if target_src.exists():
                    shutil.rmtree(target_src)
                shutil.copytree(source_dir, target_src)

            # Check config
            if not self.config_path.exists():
                logger.warning("Config missing - restoring defaults")
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                ConfigMerger.save_yaml(self.config_path, config_defaults)
            else:
                # Validate config can be loaded
                config = ConfigMerger.load_yaml(self.config_path)
                if config is None:
                    logger.warning("Config corrupted - restoring defaults")
                    ConfigMerger.backup_config(self.config_path)
                    ConfigMerger.save_yaml(self.config_path, config_defaults)

            # Verify
            if not self._verify_installation():
                logger.error("Installation verification failed after repair")
                return False

            logger.info("Repair completed successfully")
            return True

        except Exception as e:
            logger.error(f"Repair failed: {e}")
            return False

    def _verify_installation(self) -> bool:
        """
        Verify installation integrity

        Returns:
            True if installation appears valid
        """
        checks = {
            "Source code": (self.install_dir / "src" / "dictator" / "service.py").exists(),
            "Config": self.config_path.exists(),
            "Virtual environment": (self.install_dir / "venv" / "Scripts" / "python.exe").exists(),
            "Launcher": (self.install_dir / "Dictator.exe").exists(),
        }

        all_ok = True
        for name, exists in checks.items():
            if exists:
                logger.info(f"  ✓ {name}")
            else:
                logger.warning(f"  ✗ {name} - MISSING")
                all_ok = False

        return all_ok

    def get_existing_config(self) -> Optional[Dict[str, Any]]:
        """
        Get existing config

        Returns:
            Existing config dict or None
        """
        return ConfigMerger.load_yaml(self.config_path)
