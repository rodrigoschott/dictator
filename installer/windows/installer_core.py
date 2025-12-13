"""
Installer Core

Core installation engine with state machine orchestrating all installation steps.
"""

import sys
import shutil
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.state_manager import StateManager
from shared.state_schema import InstallationStep, INSTALLATION_STEPS
from shared.installer_base import (
    VirtualEnvManager,
    DependencyInstaller,
    ensure_directory
)
from dependency_checker import DependencyChecker
from model_downloader import ModelDownloader
from config_wizard import ConfigGenerator
from service_installer import ServiceInstaller
from rollback_manager import (
    RollbackManager,
    create_directory_rollback,
    create_venv_rollback,
    create_file_rollback,
    create_service_rollback,
    create_model_download_rollback
)
from updater import InstallationUpdater, ConfigMerger
from shortcuts import ShortcutCreator, UninstallRegistryManager

logger = logging.getLogger("InstallerCore")


class InstallationEngine:
    """
    Core installation engine

    Orchestrates all installation steps with state machine pattern.
    """

    def __init__(
        self,
        state_manager: StateManager,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        installation_mode: str = "install"
    ):
        """
        Initialize installation engine

        Args:
            state_manager: State manager instance
            progress_callback: Optional callback(step_name, progress_percent, message)
            installation_mode: Installation mode ("install", "update", "repair")
        """
        self.state = state_manager
        self.progress_callback = progress_callback
        self.rollback = RollbackManager()
        self.installation_mode = installation_mode

        # Paths from state
        self.install_dir = Path(self.state.config["install_dir"])
        self.venv_dir = Path(self.state.config["venv_dir"])
        self.config_dir = Path(self.state.config["config_dir"])
        self.log_dir = Path(self.state.config["log_dir"])

        # Components (initialized lazily)
        self._dependency_checker: Optional[DependencyChecker] = None
        self._updater: Optional[InstallationUpdater] = None
        self._model_downloader: Optional[ModelDownloader] = None
        self._config_generator: Optional[ConfigGenerator] = None
        self._service_installer: Optional[ServiceInstaller] = None

    def _should_skip_step(self, step: InstallationStep) -> bool:
        """
        Determine if step should be skipped based on installation mode

        Args:
            step: Installation step

        Returns:
            True if step should be skipped
        """
        # In UPDATE mode, skip these steps if they already exist and are valid
        if self.installation_mode == "update":
            if step == "create_directories":
                # Directories already exist
                return self.install_dir.exists()
            elif step == "create_venv":
                # Only skip if venv exists and is valid
                venv_python = self.venv_dir / "Scripts" / "python.exe"
                return venv_python.exists()
            elif step == "download_models":
                # Only skip if models already exist
                kokoro_model = self.install_dir / "kokoro-v1.0.onnx"
                voices_model = self.install_dir / "voices-v1.0.bin"
                return kokoro_model.exists() and voices_model.exists()

        # In REPAIR mode, don't skip anything (recreate what's missing)
        # In INSTALL mode, don't skip anything (clean install)
        return False

    def _report_progress(self, step: str, percent: float, message: str):
        """Report progress via callback"""
        if self.progress_callback:
            try:
                self.progress_callback(step, percent, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _mark_step_start(self, step: InstallationStep):
        """Mark step as started"""
        self.state.set_current_step(step)
        logger.info(f"=== Step started: {step} ===")
        self._report_progress(step, 0, f"Starting {step}")

    def _mark_step_complete(self, step: InstallationStep):
        """Mark step as completed"""
        self.state.mark_step_completed(step)
        logger.info(f"=== Step completed: {step} ===")
        self._report_progress(step, 100, f"Completed {step}")

    def _mark_step_failed(self, step: InstallationStep, error: str):
        """Mark step as failed"""
        self.state.mark_step_failed(step)
        logger.error(f"=== Step failed: {step} - {error} ===")
        self._report_progress(step, 0, f"Failed: {error}")

    # Installation step implementations

    def step_pre_flight_check(self) -> bool:
        """Pre-flight system validation"""
        step = "pre_flight_check"
        self._mark_step_start(step)

        try:
            # Initialize dependency checker
            self._dependency_checker = DependencyChecker(self.install_dir)

            # Run all checks
            self._report_progress(step, 20, "Checking system requirements...")
            all_passed, results = self._dependency_checker.check_all()

            if not all_passed:
                # Get failed checks
                failed = self._dependency_checker.get_failed_checks()
                error_msg = f"{len(failed)} critical check(s) failed"
                self._mark_step_failed(step, error_msg)
                return False

            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_select_features(self) -> bool:
        """Feature selection (handled by GUI, just validate here)"""
        step = "select_features"
        self._mark_step_start(step)

        try:
            # Validate feature selection is present
            features = self.state.features

            if not features:
                self._mark_step_failed(step, "No features selected")
                return False

            logger.info(f"Features selected: {features}")
            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_create_directories(self) -> bool:
        """Create installation directory structure"""
        step = "create_directories"
        self._mark_step_start(step)

        try:
            directories = [
                self.install_dir,
                self.venv_dir.parent,  # venv will be created later
                self.config_dir,
                self.log_dir
            ]

            for i, directory in enumerate(directories):
                self._report_progress(step, (i / len(directories)) * 100, f"Creating {directory.name}...")

                if not ensure_directory(directory):
                    self._mark_step_failed(step, f"Failed to create {directory}")
                    return False

                # Add rollback action
                self.rollback.add_action(create_directory_rollback(directory))

            # Create rollback point
            self.state.create_rollback_point(step, self.rollback.get_actions())

            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_copy_source_code(self) -> bool:
        """Copy Dictator source code to installation directory"""
        step = "copy_source_code"
        self._mark_step_start(step)

        try:
            import shutil

            self._report_progress(step, 10, "Locating source code...")

            # When running from PyInstaller, source is in _MEIPASS
            if getattr(sys, 'frozen', False):
                # Running from PyInstaller bundle
                bundle_dir = Path(sys._MEIPASS)
                source_dir = bundle_dir / "src" / "dictator"
            else:
                # Running from development environment
                source_dir = Path(__file__).parent.parent.parent / "src" / "dictator"

            if not source_dir.exists():
                logger.error(f"Source code not found at: {source_dir}")
                self._mark_step_failed(step, "Source code not found in installer")
                return False

            if not (source_dir / "service.py").exists():
                logger.error(f"service.py not found in: {source_dir}")
                self._mark_step_failed(step, "Incomplete source code package")
                return False

            target_dir = self.install_dir / "src" / "dictator"
            target_dir.parent.mkdir(parents=True, exist_ok=True)

            self._report_progress(step, 30, f"Copying source from {source_dir}...")

            # Copy entire dictator directory
            if target_dir.exists():
                shutil.rmtree(target_dir)

            shutil.copytree(source_dir, target_dir)

            # Copy pyproject.toml and README.md to installation directory (needed for pip install -e)
            self._report_progress(step, 70, "Copying project files...")

            if getattr(sys, 'frozen', False):
                bundle_dir = Path(sys._MEIPASS)
                pyproject_source = bundle_dir / "pyproject.toml"
                readme_source = bundle_dir / "README.md"
            else:
                pyproject_source = Path(__file__).parent.parent.parent / "pyproject.toml"
                readme_source = Path(__file__).parent.parent.parent / "README.md"

            if pyproject_source.exists():
                shutil.copy2(pyproject_source, self.install_dir / "pyproject.toml")
                logger.info(f" Copied pyproject.toml to {self.install_dir}")
            else:
                logger.warning(f"pyproject.toml not found at: {pyproject_source}")

            if readme_source.exists():
                shutil.copy2(readme_source, self.install_dir / "README.md")
                logger.info(f" Copied README.md to {self.install_dir}")
            else:
                logger.warning(f"README.md not found at: {readme_source}")

            self._report_progress(step, 90, "Verifying copied files...")

            # Verify critical files
            critical_files = ["service.py", "tray.py", "main.py"]
            for file in critical_files:
                if not (target_dir / file).exists():
                    self._mark_step_failed(step, f"Critical file missing: {file}")
                    return False

            # Add rollback action
            self.rollback.add_action(create_directory_rollback(target_dir.parent))

            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_create_venv(self) -> bool:
        """Create Python virtual environment"""
        step = "create_venv"
        self._mark_step_start(step)

        try:
            self._report_progress(step, 20, "Creating virtual environment...")

            success, message = VirtualEnvManager.create_venv(self.venv_dir)

            if not success:
                self._mark_step_failed(step, message)
                return False

            # Verify venv
            self._report_progress(step, 80, "Verifying virtual environment...")

            if not VirtualEnvManager.verify_venv(self.venv_dir):
                self._mark_step_failed(step, "Virtual environment verification failed")
                return False

            # Add rollback action
            self.rollback.add_action(create_venv_rollback(self.venv_dir))

            # Create rollback point
            self.state.create_rollback_point(step, self.rollback.get_actions())

            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_install_dependencies(self) -> bool:
        """Install Python dependencies"""
        step = "install_dependencies"
        self._mark_step_start(step)

        try:
            installer = DependencyInstaller(self.venv_dir)

            # Upgrade pip
            self._report_progress(step, 10, "Upgrading pip...")
            installer.upgrade_pip()

            # Install requirements
            requirements_file = Path(__file__).parent / "requirements.txt"

            if not requirements_file.exists():
                # Fallback to project root
                requirements_file = Path(__file__).parent.parent.parent / "pyproject.toml"
                self._mark_step_failed(step, "Requirements file not found")
                return False

            self._report_progress(step, 30, "Installing dependencies...")

            success, message = installer.install_requirements(requirements_file, timeout=900)

            if not success:
                self._mark_step_failed(step, message)
                return False

            # Install PyTorch with CUDA support (REQUIRED for whisper CUDA mode)
            # This is mandatory when whisper.device=cuda (default config)
            self._report_progress(step, 70, "Installing PyTorch with CUDA support...")

            success, message = installer.install_torch_cuda(cuda_version="cu124")

            if not success:
                logger.error(f"PyTorch installation failed: {message}")
                self._mark_step_failed(step, f"PyTorch installation failed: {message}")
                return False

            # Install dictator package from source (editable mode)
            self._report_progress(step, 90, "Installing Dictator package...")

            # Install from root directory (where pyproject.toml is)
            pyproject_path = self.install_dir / "pyproject.toml"
            if pyproject_path.exists():
                success, message = installer.install_package_editable(self.install_dir)

                if not success:
                    logger.error(f"Failed to install Dictator package: {message}")
                    self._mark_step_failed(step, f"Dictator package installation failed: {message}")
                    return False

                logger.info(" Dictator package installed in editable mode")
            else:
                logger.error(f"pyproject.toml not found at: {pyproject_path}")
                self._mark_step_failed(step, "pyproject.toml not found")
                return False

            # Create rollback point
            self.state.create_rollback_point(step, self.rollback.get_actions())

            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_download_models(self) -> bool:
        """Download TTS model files"""
        step = "download_models"
        self._mark_step_start(step)

        try:
            # Check if TTS is enabled
            if not self.state.features.get("tts_enabled", False):
                logger.info("TTS disabled, skipping model download")
                self._mark_step_complete(step)
                return True

            # Initialize model downloader
            manifest_path = Path(__file__).parent / "assets" / "model_manifest.json"
            download_dir = self.install_dir

            def progress_callback(progress):
                percent = progress.percent
                message = f"Downloading {progress.filename}: {percent:.1f}% ({progress.speed_mbps:.2f} MB/s)"
                self._report_progress(step, percent, message)

            self._model_downloader = ModelDownloader(
                manifest_path,
                download_dir,
                progress_callback=progress_callback
            )

            # Download all models
            self._report_progress(step, 0, "Starting model downloads...")

            success, successful, failed = self._model_downloader.download_all_models()

            if not success:
                error_msg = f"Failed to download: {', '.join(failed)}"
                self._mark_step_failed(step, error_msg)
                return False

            # Add rollback actions for downloaded files
            model_files = [download_dir / model for model in successful]
            for action in create_model_download_rollback(model_files):
                self.rollback.add_action(action)

            # Create rollback point
            self.state.create_rollback_point(step, self.rollback.get_actions())

            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_verify_models(self) -> bool:
        """Verify downloaded model checksums"""
        step = "verify_models"
        self._mark_step_start(step)

        try:
            # Skip if TTS disabled
            if not self.state.features.get("tts_enabled", False):
                self._mark_step_complete(step)
                return True

            # Models already verified during download
            # This step is for additional validation if needed

            self._report_progress(step, 100, "Models verified")
            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_generate_config(self) -> bool:
        """Generate config.yaml"""
        step = "generate_config"
        self._mark_step_start(step)

        try:
            self._config_generator = ConfigGenerator()

            # Apply feature selection
            self._report_progress(step, 30, "Applying feature selection...")
            self._config_generator.apply_feature_selection(self.state.features)

            # Set absolute paths for model files
            self._config_generator.config["tts"]["kokoro"]["model_path"] = str(self.install_dir / "kokoro-v1.0.onnx")
            self._config_generator.config["tts"]["kokoro"]["voices_path"] = str(self.install_dir / "voices-v1.0.bin")

            # Generate config file
            config_path = self.config_dir / "config.yaml"

            # UPDATE/REPAIR mode: Backup and merge existing config
            if self.installation_mode in ("update", "repair"):
                if config_path.exists():
                    self._report_progress(step, 40, "Backing up existing config...")

                    # Backup existing config
                    backup_path = ConfigMerger.backup_config(config_path)
                    if backup_path:
                        logger.info(f"Config backed up to: {backup_path}")

                    # Load existing config
                    existing_config = ConfigMerger.load_yaml(config_path)

                    # Generate new defaults
                    new_defaults = self._config_generator.get_config()

                    if existing_config:
                        self._report_progress(step, 60, "Merging with existing config...")
                        # Merge existing with new defaults
                        merged_config = ConfigMerger.merge_configs(existing_config, new_defaults)

                        # Save merged config
                        ConfigMerger.save_yaml(config_path, merged_config)
                        logger.info("Config merged successfully")
                    else:
                        # Existing config corrupted - use new defaults
                        self._report_progress(step, 70, "Writing new configuration file...")
                        success = self._config_generator.generate_config_file(config_path)
                        if not success:
                            self._mark_step_failed(step, "Failed to generate config file")
                            return False
                else:
                    # No existing config - create new
                    self._report_progress(step, 70, "Writing configuration file...")
                    success = self._config_generator.generate_config_file(config_path)
                    if not success:
                        self._mark_step_failed(step, "Failed to generate config file")
                        return False
            else:
                # INSTALL mode: Fresh config
                self._report_progress(step, 70, "Writing configuration file...")

                success = self._config_generator.generate_config_file(config_path)

                if not success:
                    self._mark_step_failed(step, "Failed to generate config file")
                    return False

            # Add rollback action
            self.rollback.add_action(create_file_rollback(config_path))

            # Create rollback point
            self.state.create_rollback_point(step, self.rollback.get_actions())

            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_install_service(self) -> bool:
        """(DEPRECATED) Install Windows service - now runs as normal app"""
        step = "install_service"
        self._mark_step_start(step)

        logger.info("Skipping service installation - Dictator now runs as normal application")
        logger.info("Auto-start configured via Windows Startup folder")

        self._mark_step_complete(step)
        return True

    def step_copy_launcher(self) -> bool:
        """Copy Dictator.exe launcher and uninstall.exe to installation directory"""
        step = "copy_launcher"
        self._mark_step_start(step)

        try:
            # Get source directory
            if getattr(sys, 'frozen', False):
                # Running from PyInstaller - assets in _MEIPASS/assets/
                bundle_dir = Path(sys._MEIPASS)
                assets_dir = bundle_dir / "assets"
            else:
                # Running as script - assets in installer/windows/assets/
                assets_dir = Path(__file__).parent / "assets"

            # Copy launcher
            self._report_progress(step, 20, "Copying launcher executable...")

            launcher_source = assets_dir / "Dictator.exe"
            launcher_target = self.install_dir / "Dictator.exe"

            if not launcher_source.exists():
                error_msg = (
                    f"Launcher executable not found: {launcher_source}\n"
                    f"Please build launcher first:\n"
                    f"  cd installer/launcher\n"
                    f"  poetry run python build_launcher.py"
                )
                self._mark_step_failed(step, error_msg)
                return False

            shutil.copy2(launcher_source, launcher_target)

            if not launcher_target.exists():
                self._mark_step_failed(step, "Failed to copy launcher")
                return False

            launcher_size_kb = launcher_target.stat().st_size / 1024
            logger.info(f"Launcher copied: {launcher_target} ({launcher_size_kb:.1f} KB)")

            # Copy uninstaller
            self._report_progress(step, 60, "Copying uninstaller executable...")

            uninstaller_source = assets_dir / "uninstall.exe"
            uninstaller_target = self.install_dir / "uninstall.exe"

            if not uninstaller_source.exists():
                logger.warning(f"Uninstaller not found: {uninstaller_source}")
                # Not critical - continue anyway
            else:
                shutil.copy2(uninstaller_source, uninstaller_target)

                if uninstaller_target.exists():
                    uninstaller_size_kb = uninstaller_target.stat().st_size / 1024
                    logger.info(f"Uninstaller copied: {uninstaller_target} ({uninstaller_size_kb:.1f} KB)")
                else:
                    logger.warning("Failed to copy uninstaller")

            # Copy NSSM for service management
            self._report_progress(step, 80, "Copying service manager...")

            nssm_source = assets_dir / "nssm.exe"
            nssm_target = self.install_dir / "nssm.exe"

            if not nssm_source.exists():
                logger.warning(f"NSSM not found: {nssm_source}")
                # Not critical - service already installed, but won't be manageable
            else:
                shutil.copy2(nssm_source, nssm_target)

                if nssm_target.exists():
                    nssm_size_kb = nssm_target.stat().st_size / 1024
                    logger.info(f"NSSM copied: {nssm_target} ({nssm_size_kb:.1f} KB)")
                else:
                    logger.warning("Failed to copy NSSM")

            self._report_progress(step, 100, "Launcher and uninstaller installed")
            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_create_shortcuts(self) -> bool:
        """Create Desktop and Start Menu shortcuts"""
        step = "create_shortcuts"
        self._mark_step_start(step)

        try:
            launcher_exe = self.install_dir / "Dictator.exe"

            # Verify launcher exists
            if not launcher_exe.exists():
                logger.warning(f"Launcher not found: {launcher_exe}")
                self._mark_step_failed(step, "Launcher executable not found")
                return False

            self._report_progress(step, 20, "Creating shortcuts...")

            # Create shortcut creator
            shortcut_creator = ShortcutCreator(self.install_dir, launcher_exe)

            # Create desktop shortcut
            self._report_progress(step, 40, "Creating desktop shortcut...")
            desktop_ok = shortcut_creator.create_desktop_shortcut()

            # Create Start Menu shortcut
            self._report_progress(step, 60, "Creating Start Menu shortcut...")
            start_menu_ok = shortcut_creator.create_start_menu_shortcut()

            # Create Startup shortcut for auto-start
            self._report_progress(step, 70, "Creating startup shortcut...")
            startup_ok = shortcut_creator.create_startup_shortcut()

            if startup_ok:
                logger.info("[OK] Startup shortcut created")
                # Note: Startup shortcut will be removed by uninstaller if needed
            else:
                logger.warning("[WARN] Failed to create startup shortcut - auto-start disabled")

            # Register in Add/Remove Programs
            self._report_progress(step, 80, "Registering uninstaller...")

            # TODO: Create uninstaller.exe first
            # For now, use placeholder
            uninstaller_exe = self.install_dir / "uninstall.exe"
            UninstallRegistryManager.register_uninstaller(
                self.install_dir,
                uninstaller_exe,
                version="1.0.0"
            )

            if not (desktop_ok and start_menu_ok):
                logger.warning("Some shortcuts failed to create")
                # Not critical - continue anyway

            self._report_progress(step, 100, "Shortcuts created")
            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_final_validation(self) -> bool:
        """Final validation before completion"""
        step = "final_validation"
        self._mark_step_start(step)

        try:
            # Import and run health check
            self._report_progress(step, 50, "Running health check...")

            # TODO: Import verify_deps.run_health_check() and validate

            self._mark_step_complete(step)
            return True

        except Exception as e:
            self._mark_step_failed(step, str(e))
            return False

    def step_start_service(self) -> bool:
        """Launch Dictator application"""
        step = "start_service"
        self._mark_step_start(step)

        try:
            # Skip if auto-start disabled
            if not self.state.features.get("auto_start_service", True):
                logger.info("Auto-start disabled")
                self._mark_step_complete(step)
                return True

            dictator_exe = self.install_dir / "Dictator.exe"
            config_path = self.config_dir / "config.yaml"

            if not dictator_exe.exists():
                logger.error(f"Dictator.exe not found: {dictator_exe}")
                self._mark_step_failed(step, "Dictator.exe not found")
                return False

            # Launch as normal process (GUI app without console window)
            self._report_progress(step, 50, "Starting Dictator...")

            import subprocess
            # Use CREATE_NO_WINDOW for GUI apps - keeps process visible in Task Manager
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                [str(dictator_exe), str(config_path)],
                cwd=str(self.install_dir),
                creationflags=CREATE_NO_WINDOW,
                close_fds=True
            )

            logger.info("[OK] Dictator launched successfully")
            logger.info("Check system tray for Dictator icon")

            self._mark_step_complete(step)
            return True

        except Exception as e:
            logger.error(f"Failed to launch Dictator: {e}")
            self._mark_step_failed(step, str(e))
            return False

    # Main installation flow

    def run_installation(self) -> bool:
        """
        Run complete installation

        Returns:
            True if successful
        """
        logger.info(f"Starting installation (mode: {self.installation_mode})...")

        # Map steps to methods
        step_methods = {
            "pre_flight_check": self.step_pre_flight_check,
            "select_features": self.step_select_features,
            "create_directories": self.step_create_directories,
            "copy_source_code": self.step_copy_source_code,
            "create_venv": self.step_create_venv,
            "install_dependencies": self.step_install_dependencies,
            "download_models": self.step_download_models,
            "verify_models": self.step_verify_models,
            "generate_config": self.step_generate_config,
            "install_service": self.step_install_service,
            "copy_launcher": self.step_copy_launcher,
            "create_shortcuts": self.step_create_shortcuts,
            "final_validation": self.step_final_validation,
            "start_service": self.step_start_service
        }

        # Get next step
        next_step = self.state.get_next_step()

        if next_step is None:
            # Verify all steps are actually complete
            all_complete = all(step in self.state.completed_steps for step in INSTALLATION_STEPS)

            if all_complete:
                logger.info("Installation already complete")
                return True
            else:
                # Edge case: inconsistent state
                logger.error("No next step but not all steps are complete!")
                logger.error(f"Completed steps: {self.state.completed_steps}")
                logger.error(f"Expected steps: {INSTALLATION_STEPS}")
                return False

        # Execute steps
        for step in INSTALLATION_STEPS:
            # Skip already completed steps
            if step in self.state.completed_steps:
                continue

            # Skip steps based on installation mode
            if self._should_skip_step(step):
                logger.info(f"Skipping step '{step}' (mode: {self.installation_mode})")
                self.state.mark_step_completed(step)
                continue

            # Execute step
            method = step_methods.get(step)
            if method is None:
                logger.error(f"No method for step: {step}")
                return False

            success = method()

            if not success:
                logger.error(f"Installation failed at step: {step}")
                return False

        logger.info(" Installation completed successfully!")
        return True

    def execute_rollback(self) -> bool:
        """
        Execute rollback of installation

        Returns:
            True if successful
        """
        logger.info("Executing rollback...")
        return self.rollback.execute_rollback()
