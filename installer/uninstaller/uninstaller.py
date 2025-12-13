"""
Dictator Uninstaller

Removes Dictator installation completely from the system.
"""

import sys
import subprocess
import shutil
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Uninstaller")


class UninstallerGUI:
    """GUI for Dictator uninstaller"""

    def __init__(self, install_dir: Path):
        """
        Initialize uninstaller GUI

        Args:
            install_dir: Installation directory to uninstall
        """
        self.install_dir = install_dir
        self.root = tk.Tk()
        self.root.title("Dictator Uninstaller")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # State
        self.uninstall_steps = [
            ("Stop Dictator Application", self.stop_application),
            ("Remove Desktop Shortcut", self.remove_desktop_shortcut),
            ("Remove Start Menu Shortcuts", self.remove_start_menu_shortcuts),
            ("Remove Startup Shortcut", self.remove_startup_shortcut),
            ("Remove Registry Entry", self.remove_registry),
            ("Clean Installer State", self.clean_installer_state),
            ("Remove Installation Directory", self.remove_directory),
        ]
        self.current_step = 0

        self.create_ui()

    def create_ui(self):
        """Create UI elements"""
        # Title
        title_frame = tk.Frame(self.root, bg="#f0f0f0", height=80)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        title = tk.Label(
            title_frame,
            text="Dictator Uninstaller",
            font=("Arial", 18, "bold"),
            bg="#f0f0f0"
        )
        title.pack(pady=20)

        # Message
        self.message_label = tk.Label(
            self.root,
            text=f"This will completely remove Dictator from your system.\n\n"
                 f"Installation directory:\n{self.install_dir}\n\n"
                 f"Are you sure you want to continue?",
            font=("Arial", 10),
            justify="center"
        )
        self.message_label.pack(pady=30)

        # Progress frame (hidden initially)
        self.progress_frame = tk.Frame(self.root)

        self.progress_label = tk.Label(
            self.progress_frame,
            text="",
            font=("Arial", 10)
        )
        self.progress_label.pack(pady=10)

        self.progress_text = tk.Text(
            self.progress_frame,
            height=10,
            width=70,
            font=("Courier", 9),
            state="disabled"
        )
        self.progress_text.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side="bottom", pady=20)

        self.btn_cancel = tk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel,
            width=15,
            height=2
        )
        self.btn_cancel.pack(side="left", padx=10)

        self.btn_uninstall = tk.Button(
            button_frame,
            text="Uninstall",
            command=self.start_uninstall,
            width=15,
            height=2,
            bg="#d9534f",
            fg="white"
        )
        self.btn_uninstall.pack(side="right", padx=10)

    def log_progress(self, message: str, success: bool = True):
        """
        Log progress message

        Args:
            message: Message to log
            success: Whether operation was successful
        """
        symbol = "✓" if success else "✗"
        color = "green" if success else "red"

        self.progress_text.config(state="normal")
        self.progress_text.insert("end", f"{symbol} {message}\n")
        self.progress_text.see("end")
        self.progress_text.config(state="disabled")

        logger.info(f"{symbol} {message}")
        self.root.update()

    def start_uninstall(self):
        """Start uninstall process"""
        # Hide initial UI, show progress
        self.message_label.pack_forget()
        self.progress_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.btn_uninstall.config(state="disabled")
        self.btn_cancel.config(state="disabled")

        # Run uninstall steps
        success = self.run_uninstall_steps()

        # Show completion
        if success:
            self.log_progress("Uninstall completed successfully!", True)
            messagebox.showinfo(
                "Uninstall Complete",
                "Dictator has been successfully removed from your system."
            )
        else:
            self.log_progress("Uninstall completed with errors", False)
            messagebox.showwarning(
                "Uninstall Completed",
                "Uninstall completed but some items could not be removed.\n"
                "Please check the log for details."
            )

        self.root.quit()

    def run_uninstall_steps(self) -> bool:
        """
        Run all uninstall steps

        Returns:
            True if all successful
        """
        all_success = True

        for step_name, step_func in self.uninstall_steps:
            self.current_step += 1
            self.progress_label.config(
                text=f"Step {self.current_step}/{len(self.uninstall_steps)}: {step_name}"
            )

            try:
                success = step_func()
                self.log_progress(step_name, success)

                if not success:
                    all_success = False

            except Exception as e:
                self.log_progress(f"{step_name}: {e}", False)
                all_success = False

        return all_success

    def stop_application(self) -> bool:
        """Stop Dictator application if running"""
        try:
            # Try using psutil if available
            try:
                import psutil

                stopped = 0
                for proc in psutil.process_iter(['name', 'exe']):
                    try:
                        if proc.info['name'] == 'Dictator.exe':
                            logger.info(f"Found Dictator process (PID {proc.pid})")
                            proc.terminate()

                            # Wait up to 5 seconds for graceful shutdown
                            try:
                                proc.wait(timeout=5)
                                stopped += 1
                            except psutil.TimeoutExpired:
                                logger.warning(f"Force killing process {proc.pid}")
                                proc.kill()
                                stopped += 1

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if stopped > 0:
                    logger.info(f"[OK] Stopped {stopped} Dictator process(es)")
                else:
                    logger.info("Dictator not running")

                return True

            except ImportError:
                # psutil not available - fallback to taskkill
                result = subprocess.run(
                    ["taskkill", "/IM", "Dictator.exe", "/F"],
                    capture_output=True,
                    text=True,
                    check=False
                )

                if result.returncode == 0:
                    logger.info("Dictator process stopped")
                else:
                    logger.info("Dictator not running")

                return True

        except Exception as e:
            logger.error(f"Failed to stop application: {e}")
            return False

    def remove_desktop_shortcut(self) -> bool:
        """Remove desktop shortcut"""
        try:
            import winshell
            desktop = winshell.desktop()
            shortcut_path = Path(desktop) / "Dictator.lnk"

            if shortcut_path.exists():
                shortcut_path.unlink()
                logger.info(f"Removed desktop shortcut: {shortcut_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to remove desktop shortcut: {e}")
            return False

    def remove_start_menu_shortcuts(self) -> bool:
        """Remove Start Menu shortcuts"""
        try:
            import winshell
            start_menu = winshell.start_menu()
            programs_folder = Path(start_menu) / "Programs" / "Dictator"

            if programs_folder.exists():
                shutil.rmtree(programs_folder)
                logger.info(f"Removed Start Menu folder: {programs_folder}")

            return True

        except Exception as e:
            logger.error(f"Failed to remove Start Menu shortcuts: {e}")
            return False

    def remove_startup_shortcut(self) -> bool:
        """Remove Windows Startup shortcut"""
        try:
            import winshell
            startup_folder = Path(winshell.startup())
            shortcut_path = startup_folder / "Dictator.lnk"

            if shortcut_path.exists():
                shortcut_path.unlink()
                logger.info(f"[OK] Removed startup shortcut: {shortcut_path}")
            else:
                logger.info("Startup shortcut not found (already removed)")

            return True

        except Exception as e:
            logger.error(f"Failed to remove startup shortcut: {e}")
            return False

    def remove_registry(self) -> bool:
        """Remove registry entry from Add/Remove Programs"""
        try:
            import winreg

            try:
                winreg.DeleteKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Dictator"
                )
                logger.info("Removed registry entry")
            except FileNotFoundError:
                logger.info("Registry entry not found")

            return True

        except Exception as e:
            logger.error(f"Failed to remove registry: {e}")
            return False

    def clean_installer_state(self) -> bool:
        """Clean installer state file"""
        try:
            # Remove state file from user directory
            state_file = Path.home() / ".dictator" / "installer_state.json"

            if state_file.exists():
                state_file.unlink()
                logger.info(f"Removed installer state: {state_file}")

                # Try to remove .dictator directory if empty
                try:
                    state_file.parent.rmdir()
                    logger.info(f"Removed .dictator directory: {state_file.parent}")
                except OSError:
                    # Directory not empty - that's ok
                    pass

            return True

        except Exception as e:
            logger.error(f"Failed to clean installer state: {e}")
            return False

    def remove_directory(self) -> bool:
        """Remove installation directory"""
        try:
            if not self.install_dir.exists():
                logger.info("Installation directory not found")
                return True

            # Strategy: Use rmdir /s /q command which can remove files in use
            import subprocess
            import time

            logger.info(f"Removing: {self.install_dir}")

            # Try using Windows rmdir command (more aggressive)
            try:
                # Schedule deletion after uninstaller exits
                bat_content = f"""@echo off
timeout /t 2 /nobreak >nul
rmdir /s /q "{self.install_dir}"
del "%~f0"
"""
                import tempfile
                bat_file = Path(tempfile.gettempdir()) / "dictator_cleanup.bat"
                bat_file.write_text(bat_content, encoding='utf-8')

                # Run batch file in background
                subprocess.Popen(
                    ['cmd', '/c', str(bat_file)],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    close_fds=True
                )

                logger.info("Scheduled directory removal after uninstaller exits")
                return True

            except Exception as e:
                logger.warning(f"Failed to schedule removal: {e}")

                # Fallback: Try direct removal
                try:
                    shutil.rmtree(self.install_dir, ignore_errors=True)
                    logger.info("Installation directory removed (some files may remain)")
                    return True
                except Exception as e2:
                    logger.error(f"Failed to remove directory: {e2}")
                    return False

        except Exception as e:
            logger.error(f"Failed to remove directory: {e}")
            return False

    def cancel(self):
        """Cancel uninstall"""
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel?"):
            self.root.quit()

    def run(self):
        """Run uninstaller GUI"""
        self.root.mainloop()


def main():
    """Main uninstaller entry point"""
    # Determine installation directory
    if getattr(sys, 'frozen', False):
        # Running from PyInstaller - uninstall.exe is in install dir
        install_dir = Path(sys.executable).parent
    else:
        # Running as script (testing) - use argument or default
        if len(sys.argv) > 1:
            install_dir = Path(sys.argv[1])
        else:
            install_dir = Path(r"D:\Programas\Dictator")

    # Verify installation exists
    if not install_dir.exists():
        messagebox.showerror(
            "Error",
            f"Installation directory not found:\n{install_dir}"
        )
        return 1

    # Run uninstaller GUI
    uninstaller = UninstallerGUI(install_dir)
    uninstaller.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
