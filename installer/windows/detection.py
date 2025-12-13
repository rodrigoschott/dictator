"""
Installation Detection

Detects existing Dictator installations on the system.
"""

import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ExistingInstallation:
    """Information about an existing installation"""
    path: Path
    has_service: bool
    has_venv: bool
    has_source: bool
    has_models: bool
    has_config: bool
    service_status: Optional[str]  # running, stopped, paused, None

    @property
    def is_valid(self) -> bool:
        """Check if installation appears valid"""
        return self.has_venv and self.has_source

    @property
    def completeness_score(self) -> float:
        """Calculate how complete the installation is (0.0 to 1.0)"""
        checks = [
            self.has_service,
            self.has_venv,
            self.has_source,
            self.has_models,
            self.has_config
        ]
        return sum(checks) / len(checks)


class InstallationDetector:
    """Detects and validates existing Dictator installations"""

    # Common installation directories to check
    COMMON_PATHS = [
        Path(r"D:\Programas\Dictator"),
        Path(r"C:\Program Files\Dictator"),
        Path(r"C:\Dictator"),
        Path.home() / "Dictator",
    ]

    def detect_existing_installation(self) -> Optional[ExistingInstallation]:
        """
        Detect existing Dictator installation

        Returns:
            ExistingInstallation if found, None otherwise
        """
        # Strategy 1: Check if service exists and get its path
        service_path = self._get_service_path()
        if service_path:
            print(f"Found Dictator service at: {service_path}")
            return self._analyze_installation(service_path)

        # Strategy 2: Check common installation directories
        for path in self.COMMON_PATHS:
            if path.exists() and self._looks_like_dictator(path):
                print(f"Found Dictator installation at: {path}")
                return self._analyze_installation(path)

        # No installation found
        return None

    def _get_service_path(self) -> Optional[Path]:
        """
        Get installation path from Windows service configuration

        Returns:
            Path to installation directory if service exists, None otherwise
        """
        try:
            # Query service using sc
            result = subprocess.run(
                ["sc", "qc", "Dictator"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                # Service doesn't exist
                return None

            # Parse output for BINARY_PATH_NAME
            # Example: BINARY_PATH_NAME   : D:\Programas\Dictator\nssm.exe
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("BINARY_PATH_NAME"):
                    # Extract path
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        binary_path = parts[1].strip()
                        # Remove quotes if present
                        binary_path = binary_path.strip('"')

                        # Binary is usually nssm.exe in installation directory
                        binary_path = Path(binary_path)
                        if binary_path.name.lower() == "nssm.exe":
                            return binary_path.parent
                        else:
                            # Unexpected binary
                            return binary_path.parent

            return None

        except Exception as e:
            print(f"Error querying service: {e}")
            return None

    def _get_service_status(self) -> Optional[str]:
        """
        Get Windows service status

        Returns:
            "running", "stopped", "paused", or None
        """
        try:
            result = subprocess.run(
                ["sc", "query", "Dictator"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                return None

            # Parse STATE line
            # Example: STATE              : 4  RUNNING
            for line in result.stdout.splitlines():
                line = line.strip()
                if "STATE" in line:
                    if "RUNNING" in line:
                        return "running"
                    elif "STOPPED" in line:
                        return "stopped"
                    elif "PAUSED" in line:
                        return "paused"

            return None

        except Exception:
            return None

    def _looks_like_dictator(self, path: Path) -> bool:
        """
        Quick check if directory looks like Dictator installation

        Args:
            path: Directory to check

        Returns:
            True if looks like Dictator installation
        """
        # Must have src/dictator/ or venv/
        has_source = (path / "src" / "dictator").exists()
        has_venv = (path / "venv").exists()

        return has_source or has_venv

    def _analyze_installation(self, path: Path) -> ExistingInstallation:
        """
        Analyze installation at given path

        Args:
            path: Installation directory

        Returns:
            ExistingInstallation with detailed information
        """
        # Check components
        has_service = self._get_service_status() is not None
        has_venv = (path / "venv").exists() and (path / "venv" / "Scripts" / "python.exe").exists()
        has_source = (path / "src" / "dictator").exists() and (path / "src" / "dictator" / "service.py").exists()
        has_config = (path / "config" / "config.yaml").exists()

        # Check models (either TTS model is enough)
        has_models = (
            (path / "kokoro-v1.0.onnx").exists() or
            (path / "voices-v1.0.bin").exists()
        )

        service_status = self._get_service_status()

        return ExistingInstallation(
            path=path,
            has_service=has_service,
            has_venv=has_venv,
            has_source=has_source,
            has_models=has_models,
            has_config=has_config,
            service_status=service_status
        )

    def get_installation_summary(self, installation: ExistingInstallation) -> str:
        """
        Get human-readable summary of installation

        Args:
            installation: Existing installation info

        Returns:
            Multi-line summary string
        """
        lines = [
            f"Instalação detectada em: {installation.path}",
            "",
            "Componentes encontrados:",
        ]

        # Add component status
        lines.append(f"  {'✓' if installation.has_service else '✗'} Serviço Windows: {installation.service_status or 'não instalado'}")
        lines.append(f"  {'✓' if installation.has_venv else '✗'} Ambiente Python (venv)")
        lines.append(f"  {'✓' if installation.has_source else '✗'} Código fonte")
        lines.append(f"  {'✓' if installation.has_models else '✗'} Modelos TTS")
        lines.append(f"  {'✓' if installation.has_config else '✗'} Configuração")

        lines.append("")
        lines.append(f"Completude: {installation.completeness_score * 100:.0f}%")

        if installation.is_valid:
            lines.append("Status: Instalação válida")
        else:
            lines.append("Status: Instalação incompleta ou corrompida")

        return "\n".join(lines)


# Convenience function
def detect_installation() -> Optional[ExistingInstallation]:
    """
    Convenience function to detect existing installation

    Returns:
        ExistingInstallation if found, None otherwise
    """
    detector = InstallationDetector()
    return detector.detect_existing_installation()
