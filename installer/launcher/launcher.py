"""
Dictator Launcher

Inicia o Dictator Tray application.
Este executável é criado pelo instalador e colocado no diretório de instalação.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Lança a aplicação Dictator Tray"""
    # Get installation directory (where launcher.exe is)
    if getattr(sys, 'frozen', False):
        # Running from PyInstaller bundle
        install_dir = Path(sys.executable).parent
    else:
        # Running as script (for testing)
        # Assume launcher is in installer/launcher/, go up 2 levels
        install_dir = Path(__file__).parent.parent.parent

    # Paths
    python_exe = install_dir / "venv" / "Scripts" / "pythonw.exe"
    tray_script = install_dir / "src" / "dictator" / "tray.py"
    config_file = install_dir / "config" / "config.yaml"

    # Validate installation
    if not python_exe.exists():
        show_error(
            "Dictator não está instalado corretamente.\n\n"
            f"Python não encontrado em:\n{python_exe}\n\n"
            "Por favor, reinstale o Dictator."
        )
        return 1

    if not tray_script.exists():
        show_error(
            "Dictator não está instalado corretamente.\n\n"
            f"Arquivos faltando:\n{tray_script}\n\n"
            "Por favor, reinstale o Dictator."
        )
        return 1

    if not config_file.exists():
        show_error(
            "Dictator não está instalado corretamente.\n\n"
            f"Configuração não encontrada:\n{config_file}\n\n"
            "Por favor, reinstale o Dictator."
        )
        return 1

    # Launch tray application (no console window)
    try:
        # Use CREATE_NO_WINDOW to prevent console flash
        # pythonw.exe already doesn't show console, but this ensures it
        CREATE_NO_WINDOW = 0x08000000

        subprocess.Popen(
            [str(python_exe), str(tray_script), str(config_file)],
            creationflags=CREATE_NO_WINDOW,
            cwd=str(install_dir),
            close_fds=True
        )
        return 0

    except Exception as e:
        show_error(
            f"Erro ao iniciar Dictator:\n\n{e}\n\n"
            f"Instalação: {install_dir}\n"
            f"Python: {python_exe}\n"
            f"Script: {tray_script}"
        )
        return 1


def show_error(message: str):
    """Mostra erro usando MessageBox do Windows"""
    try:
        import ctypes
        # MB_ICONERROR = 0x10
        ctypes.windll.user32.MessageBoxW(0, message, "Dictator - Erro", 0x10)
    except Exception:
        # Fallback: print to console (won't show if frozen without console)
        print(f"ERRO: {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
