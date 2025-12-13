"""
Windows Shortcuts Creator

Creates desktop and Start Menu shortcuts for Dictator.
"""

import winshell
from pathlib import Path
from win32com.client import Dispatch
import logging

logger = logging.getLogger("Shortcuts")


class ShortcutCreator:
    """Creates Windows shortcuts for Dictator"""

    def __init__(self, install_dir: Path, launcher_exe: Path):
        """
        Initialize shortcut creator

        Args:
            install_dir: Installation directory
            launcher_exe: Path to Dictator.exe launcher
        """
        self.install_dir = install_dir
        self.launcher_exe = launcher_exe

    def create_desktop_shortcut(self, name: str = "Dictator") -> bool:
        """
        Create desktop shortcut

        Args:
            name: Shortcut name

        Returns:
            True if successful
        """
        try:
            desktop = winshell.desktop()
            shortcut_path = Path(desktop) / f"{name}.lnk"

            logger.info(f"Creating desktop shortcut: {shortcut_path}")

            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = str(self.launcher_exe)
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.Description = "Dictator - Voice to Text"
            shortcut.IconLocation = str(self.launcher_exe) + ",0"
            shortcut.save()

            logger.info(f"✓ Desktop shortcut created: {shortcut_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create desktop shortcut: {e}")
            return False

    def create_start_menu_shortcut(self, name: str = "Dictator") -> bool:
        """
        Create Start Menu shortcut

        Args:
            name: Shortcut name

        Returns:
            True if successful
        """
        try:
            # Get Start Menu Programs folder
            start_menu = winshell.start_menu()
            programs_folder = Path(start_menu) / "Programs" / "Dictator"

            # Create Dictator folder in Start Menu
            programs_folder.mkdir(parents=True, exist_ok=True)

            shortcut_path = programs_folder / f"{name}.lnk"

            logger.info(f"Creating Start Menu shortcut: {shortcut_path}")

            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = str(self.launcher_exe)
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.Description = "Dictator - Voice to Text"
            shortcut.IconLocation = str(self.launcher_exe) + ",0"
            shortcut.save()

            logger.info(f"✓ Start Menu shortcut created: {shortcut_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Start Menu shortcut: {e}")
            return False

    def create_startup_shortcut(self, name: str = "Dictator") -> bool:
        """
        Create Windows Startup shortcut for auto-start

        Args:
            name: Shortcut name

        Returns:
            True if successful
        """
        try:
            # Get Startup folder path
            startup_folder = Path(winshell.startup())
            shortcut_path = startup_folder / f"{name}.lnk"

            logger.info(f"Creating startup shortcut: {shortcut_path}")

            # Get config path for arguments
            config_path = self.install_dir / "config.yaml"

            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = str(self.launcher_exe)
            shortcut.Arguments = f'"{config_path}"'
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.Description = "Dictator - Voice to Text"
            shortcut.IconLocation = str(self.launcher_exe) + ",0"
            shortcut.save()

            logger.info(f"[OK] Startup shortcut created: {shortcut_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create startup shortcut: {e}")
            return False

    def remove_startup_shortcut(self, name: str = "Dictator") -> bool:
        """
        Remove Windows Startup shortcut

        Args:
            name: Shortcut name

        Returns:
            True if successful
        """
        try:
            startup_folder = Path(winshell.startup())
            shortcut_path = startup_folder / f"{name}.lnk"

            if shortcut_path.exists():
                shortcut_path.unlink()
                logger.info(f"[OK] Startup shortcut removed: {shortcut_path}")
            else:
                logger.info("Startup shortcut not found (already removed)")

            return True

        except Exception as e:
            logger.error(f"Failed to remove startup shortcut: {e}")
            return False

    def create_all_shortcuts(self) -> bool:
        """
        Create all shortcuts (Desktop + Start Menu + Startup)

        Returns:
            True if all successful
        """
        desktop_ok = self.create_desktop_shortcut()
        start_menu_ok = self.create_start_menu_shortcut()
        startup_ok = self.create_startup_shortcut()

        return desktop_ok and start_menu_ok and startup_ok

    def remove_desktop_shortcut(self, name: str = "Dictator") -> bool:
        """
        Remove desktop shortcut

        Args:
            name: Shortcut name

        Returns:
            True if successful
        """
        try:
            desktop = winshell.desktop()
            shortcut_path = Path(desktop) / f"{name}.lnk"

            if shortcut_path.exists():
                shortcut_path.unlink()
                logger.info(f"✓ Desktop shortcut removed: {shortcut_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to remove desktop shortcut: {e}")
            return False

    def remove_start_menu_shortcut(self, name: str = "Dictator") -> bool:
        """
        Remove Start Menu shortcuts

        Args:
            name: Shortcut name

        Returns:
            True if successful
        """
        try:
            # Get Start Menu Programs folder
            start_menu = winshell.start_menu()
            programs_folder = Path(start_menu) / "Programs" / "Dictator"

            if programs_folder.exists():
                # Remove shortcut
                shortcut_path = programs_folder / f"{name}.lnk"
                if shortcut_path.exists():
                    shortcut_path.unlink()

                # Remove folder if empty
                try:
                    programs_folder.rmdir()
                    logger.info(f"✓ Start Menu folder removed: {programs_folder}")
                except OSError:
                    # Folder not empty - that's ok
                    pass

            return True

        except Exception as e:
            logger.error(f"Failed to remove Start Menu shortcut: {e}")
            return False

    def remove_all_shortcuts(self) -> bool:
        """
        Remove all shortcuts (Desktop + Start Menu + Startup)

        Returns:
            True if all successful
        """
        desktop_ok = self.remove_desktop_shortcut()
        start_menu_ok = self.remove_start_menu_shortcut()
        startup_ok = self.remove_startup_shortcut()

        return desktop_ok and start_menu_ok and startup_ok


class UninstallRegistryManager:
    """Manages Windows Add/Remove Programs registry entry"""

    REGISTRY_PATH = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Dictator"

    @staticmethod
    def register_uninstaller(
        install_dir: Path,
        uninstaller_exe: Path,
        version: str = "1.0.0"
    ) -> bool:
        """
        Register in Add/Remove Programs

        Args:
            install_dir: Installation directory
            uninstaller_exe: Path to uninstaller executable
            version: Version string

        Returns:
            True if successful
        """
        try:
            import winreg

            # Open/create registry key
            key = winreg.CreateKey(
                winreg.HKEY_LOCAL_MACHINE,
                UninstallRegistryManager.REGISTRY_PATH
            )

            # Set values
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Dictator")
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, version)
            winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Dictator")
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(install_dir))
            winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, str(uninstaller_exe))
            winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)

            # Calculate install size (in KB)
            total_size = 0
            for item in install_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size

            size_kb = int(total_size / 1024)
            winreg.SetValueEx(key, "EstimatedSize", 0, winreg.REG_DWORD, size_kb)

            winreg.CloseKey(key)

            logger.info("✓ Registered in Add/Remove Programs")
            return True

        except Exception as e:
            logger.error(f"Failed to register uninstaller: {e}")
            return False

    @staticmethod
    def unregister_uninstaller() -> bool:
        """
        Remove from Add/Remove Programs

        Returns:
            True if successful
        """
        try:
            import winreg

            # Delete registry key
            winreg.DeleteKey(
                winreg.HKEY_LOCAL_MACHINE,
                UninstallRegistryManager.REGISTRY_PATH
            )

            logger.info("✓ Unregistered from Add/Remove Programs")
            return True

        except FileNotFoundError:
            # Key doesn't exist - that's ok
            return True
        except Exception as e:
            logger.error(f"Failed to unregister: {e}")
            return False
