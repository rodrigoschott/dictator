#!/usr/bin/env python3
"""
Health Check and Dependency Validation Module

Provides comprehensive validation of system dependencies at installation
and startup, with graceful degradation for optional features.
"""

import logging
import subprocess
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

# Export public API for installer integration
__all__ = ['DependencyValidator', 'HealthReport', 'ComponentStatus']


@dataclass
class ComponentStatus:
    """Status of an individual component/dependency"""
    name: str
    status: str  # "healthy", "degraded", "unavailable", "critical"
    required: bool = False
    enabled: bool = True
    message: str = ""
    fix_hint: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v}  # Skip empty fields


@dataclass
class HealthReport:
    """Consolidated health report for the entire system"""
    timestamp: str
    overall_status: str  # "healthy", "degraded", "critical"
    components: List[ComponentStatus]
    degraded_features: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "degraded_features": self.degraded_features,
            "components": [c.to_dict() for c in self.components]
        }

    def get_tray_tooltip(self, base_text: str = "Dictator") -> str:
        """Generate tooltip text for system tray"""
        if self.overall_status == "healthy":
            return f"{base_text} - Healthy"
        elif self.overall_status == "degraded":
            if self.degraded_features:
                features = ", ".join(self.degraded_features)
                return f"{base_text} - Degraded ({features})"
            else:
                return f"{base_text} - Degraded"
        else:
            return f"{base_text} - Critical Issues"


class DependencyValidator:
    """
    Central dependency validator

    Validates system dependencies at installation and startup,
    providing detailed health reports with actionable fix hints.
    """

    def __init__(self, config: dict):
        """
        Initialize validator

        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('HealthCheck')

    def run_full_check(self, quick: bool = False) -> HealthReport:
        """
        Run complete health check

        Args:
            quick: If True, skip optional/slow checks

        Returns:
            HealthReport with status of all components
        """
        self.logger.info("Running dependency validation...")

        components = []

        # Critical checks (always run)
        components.append(self.check_python_packages())
        components.append(self.check_audio_device())
        components.append(self.check_gpu_cuda())

        # Optional checks (can be skipped in quick mode)
        if not quick:
            components.append(self.check_model_files())
            components.append(self.check_git_lfs())
            components.append(self.check_whisper_cache())

            # External services (only if enabled)
            if self.config.get('voice', {}).get('claude_mode', False):
                components.append(self.check_ollama())
                components.append(self.check_n8n())

        # Determine overall status
        overall_status = "healthy"
        degraded_features = []

        for component in components:
            # Log component status
            if component.status == "healthy":
                self.logger.info(f"{component.name}: {component.message}")
            elif component.status == "degraded":
                self.logger.warning(f"{component.name}: {component.message}")
                if component.fix_hint:
                    self.logger.warning(f"   Fix: {component.fix_hint}")
            elif component.status in ["unavailable", "critical"]:
                self.logger.warning(f"{component.name}: {component.message}")
                if component.fix_hint:
                    self.logger.warning(f"   Fix: {component.fix_hint}")

                # Track degraded features
                if "TTS" in component.name or "Git LFS" in component.name:
                    if "TTS" not in degraded_features:
                        degraded_features.append("TTS")
                elif component.name in ["Ollama", "N8N"]:
                    if "Voice Assistant" not in degraded_features:
                        degraded_features.append("Voice Assistant")

            # Update overall status
            if component.status == "critical" and component.required:
                overall_status = "critical"
            elif component.status in ["degraded", "unavailable"] and overall_status != "critical":
                overall_status = "degraded"

        return HealthReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            components=components,
            degraded_features=degraded_features
        )

    def check_python_packages(self) -> ComponentStatus:
        """Verify critical Python packages are installed"""
        critical_packages = [
            'faster_whisper',
            'sounddevice',
            'soundfile',
            'pyperclip',
            'pyautogui',
            'pynput',
            'pystray',
            'PIL',
            'yaml',
            'numpy',
            'aiohttp',
            'requests'
        ]

        missing = []
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            return ComponentStatus(
                name="Python Packages",
                status="critical",
                required=True,
                message=f"Missing packages: {', '.join(missing)}",
                fix_hint="Run: poetry install"
            )

        return ComponentStatus(
            name="Python Packages",
            status="healthy",
            required=True,
            message=f"All packages installed ({len(critical_packages)}/{len(critical_packages)})"
        )

    def check_model_files(self) -> ComponentStatus:
        """Verify TTS model files exist and are valid"""
        if not self.config.get('tts', {}).get('enabled', False):
            return ComponentStatus(
                name="TTS Model Files",
                status="healthy",
                enabled=False,
                message="TTS disabled in config"
            )

        try:
            model_path = self.config['tts']['kokoro']['model_path']
            voices_path = self.config['tts']['kokoro']['voices_path']

            model_file = Path(model_path)
            voices_file = Path(voices_path)

            # Check both files exist
            if not model_file.exists():
                return ComponentStatus(
                    name="TTS Model Files",
                    status="unavailable",
                    required=False,
                    message=f"Model file not found: {model_path}",
                    fix_hint="Run: git lfs pull"
                )

            if not voices_file.exists():
                return ComponentStatus(
                    name="TTS Model Files",
                    status="unavailable",
                    required=False,
                    message=f"Voices file not found: {voices_path}",
                    fix_hint="Run: git lfs pull"
                )

            # Check if files are LFS pointers (very small files)
            model_size = model_file.stat().st_size
            voices_size = voices_file.stat().st_size

            if model_size < 1000:  # LFS pointers are tiny
                # Check if it starts with LFS marker
                with open(model_file, 'rb') as f:
                    header = f.read(20)
                    if header.startswith(b'version https://'):
                        return ComponentStatus(
                            name="TTS Model Files",
                            status="unavailable",
                            required=False,
                            message="Model files are LFS pointers (not downloaded)",
                            fix_hint="Run: git lfs pull"
                        )

            total_size_mb = (model_size + voices_size) / 1e6

            return ComponentStatus(
                name="TTS Model Files",
                status="healthy",
                required=False,
                message=f"Model files present ({total_size_mb:.0f}MB)",
                details={
                    "model_path": str(model_file),
                    "voices_path": str(voices_file),
                    "total_size_mb": round(total_size_mb, 1)
                }
            )

        except Exception as e:
            return ComponentStatus(
                name="TTS Model Files",
                status="degraded",
                required=False,
                message=f"Check failed: {e}"
            )

    def check_git_lfs(self) -> ComponentStatus:
        """Verify Git LFS is installed"""
        try:
            result = subprocess.run(
                ['git', 'lfs', 'version'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return ComponentStatus(
                    name="Git LFS",
                    status="unavailable",
                    required=False,
                    message="Git LFS not installed",
                    fix_hint="Run: git lfs install"
                )

            # Extract version from output
            version = result.stdout.strip().split('\n')[0] if result.stdout else "unknown"

            return ComponentStatus(
                name="Git LFS",
                status="healthy",
                required=False,
                message=f"Installed: {version}",
                details={"version": version}
            )

        except FileNotFoundError:
            return ComponentStatus(
                name="Git LFS",
                status="unavailable",
                required=False,
                message="Git or Git LFS not found in PATH",
                fix_hint="Install Git LFS: https://git-lfs.github.com/"
            )
        except Exception as e:
            return ComponentStatus(
                name="Git LFS",
                status="degraded",
                required=False,
                message=f"Check failed: {e}"
            )

    def check_gpu_cuda(self) -> ComponentStatus:
        """Verify GPU/CUDA availability when required"""
        config_device = self.config.get('whisper', {}).get('device', 'cpu')

        if config_device == 'cpu':
            return ComponentStatus(
                name="GPU/CUDA",
                status="healthy",
                enabled=False,
                message="CPU mode configured (GPU not needed)"
            )

        try:
            # Try torch first (if available)
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "unknown"

                return ComponentStatus(
                    name="GPU/CUDA",
                    status="healthy",
                    required=True,
                    message=f"CUDA available: {gpu_name}",
                    details={
                        "device": gpu_name,
                        "cuda_version": cuda_version
                    }
                )
            else:
                # CUDA not available but config requires it
                return ComponentStatus(
                    name="GPU/CUDA",
                    status="critical",
                    required=True,
                    message="device=cuda in config but CUDA not available",
                    fix_hint="Install NVIDIA drivers or change config to device=cpu"
                )

        except ImportError:
            # Torch not available, CTranslate2 will handle CUDA automatically
            return ComponentStatus(
                name="GPU/CUDA",
                status="degraded",
                required=False,
                message="Cannot verify CUDA (torch not installed), CTranslate2 will auto-detect"
            )

    def check_audio_device(self) -> ComponentStatus:
        """Verify audio input device is accessible"""
        try:
            import sounddevice as sd

            # Get default input device
            device_info = sd.query_devices(kind='input')

            # Try to open a very short test stream
            with sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype='float32',
                blocksize=1024
            ):
                pass  # Just test if we can open

            return ComponentStatus(
                name="Audio Device",
                status="healthy",
                required=True,
                message=f"Microphone: {device_info['name']}",
                details={"device": device_info['name']}
            )

        except Exception as e:
            return ComponentStatus(
                name="Audio Device",
                status="critical",
                required=True,
                message=f"Cannot access microphone: {e}",
                fix_hint="Check Windows microphone permissions and default device"
            )

    def check_ollama(self) -> ComponentStatus:
        """Verify Ollama service is running and accessible"""
        # Only check if voice mode enabled
        if not self.config.get('voice', {}).get('claude_mode', False):
            return ComponentStatus(
                name="Ollama",
                status="healthy",
                enabled=False,
                message="Voice mode disabled (Ollama not needed)"
            )

        provider = self.config.get('voice', {}).get('llm', {}).get('provider')
        if provider != 'ollama':
            return ComponentStatus(
                name="Ollama",
                status="healthy",
                enabled=False,
                message=f"Using provider: {provider}"
            )

        try:
            import requests

            base_url = self.config['voice']['llm']['ollama']['base_url']

            # Quick ping with short timeout
            response = requests.get(f"{base_url}/api/tags", timeout=1)

            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]

                return ComponentStatus(
                    name="Ollama",
                    status="healthy",
                    required=False,
                    message=f"Running with {len(models)} model(s)",
                    details={
                        "url": base_url,
                        "models": model_names
                    }
                )
            else:
                raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            return ComponentStatus(
                name="Ollama",
                status="unavailable",
                required=False,
                message=f"Not accessible: {e}",
                fix_hint="Start Ollama service: ollama serve"
            )

    def check_n8n(self) -> ComponentStatus:
        """Verify N8N webhook is accessible"""
        provider = self.config.get('voice', {}).get('llm', {}).get('provider')

        if provider != 'n8n_toolcalling':
            return ComponentStatus(
                name="N8N",
                status="healthy",
                enabled=False,
                message="N8N provider not configured"
            )

        try:
            import requests

            webhook_url = self.config['voice']['llm']['n8n_toolcalling']['webhook_url']

            # Try HEAD request with timeout
            response = requests.head(webhook_url, timeout=2)

            return ComponentStatus(
                name="N8N",
                status="healthy",
                required=False,
                message="Webhook accessible",
                details={"url": webhook_url}
            )

        except Exception as e:
            return ComponentStatus(
                name="N8N",
                status="unavailable",
                required=False,
                message=f"Webhook not accessible: {e}",
                fix_hint="Start N8N workflow and verify webhook URL"
            )

    def check_whisper_cache(self) -> ComponentStatus:
        """Verify Whisper model cache directory"""
        try:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

            if not cache_dir.exists():
                return ComponentStatus(
                    name="Whisper Cache",
                    status="degraded",
                    required=False,
                    message="Model cache not found (will download on first run)",
                    details={"cache_dir": str(cache_dir)}
                )

            # Check if model exists
            model_name = self.config.get('whisper', {}).get('model', 'large-v3')

            # Count model files in cache
            model_files = list(cache_dir.glob(f"*{model_name}*"))

            if model_files:
                return ComponentStatus(
                    name="Whisper Cache",
                    status="healthy",
                    required=False,
                    message=f"Model '{model_name}' cached",
                    details={"cache_dir": str(cache_dir)}
                )
            else:
                return ComponentStatus(
                    name="Whisper Cache",
                    status="degraded",
                    required=False,
                    message=f"Model '{model_name}' not cached (will download on first run)"
                )

        except Exception as e:
            return ComponentStatus(
                name="Whisper Cache",
                status="degraded",
                required=False,
                message=f"Check failed: {e}"
            )
