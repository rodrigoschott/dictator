"""
Configuration Wizard

Generates config.yaml based on user selections and system capabilities.
"""

from pathlib import Path
from typing import Dict, Any
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger("ConfigWizard")


class ConfigGenerator:
    """
    Generates Dictator configuration file

    Creates config.yaml with appropriate defaults based on:
    - User feature selections
    - System capabilities (GPU, TTS models, etc.)
    - Installation paths
    """

    # Default configuration template
    DEFAULT_CONFIG = {
        "whisper": {
            "model": "large-v3",
            "language": "pt",
            "device": "cuda"
        },
        "hotkey": {
            "type": "mouse",
            "mouse_button": "middle",
            "keyboard_trigger": "ctrl+alt+v",
            "mode": "push_to_talk",
            "vad_enabled": False,
            "vad_threshold": 0.002,
            "auto_stop_silence": 2.0,
            "max_duration": 60
        },
        "paste": {
            "auto_paste": True,
            "delay": 0.2
        },
        "tts": {
            "enabled": True,
            "engine": "kokoro-onnx",
            "volume": 0.8,
            "interrupt_on_speech": True,
            "kokoro": {
                "model_path": "kokoro-v1.0.onnx",
                "voices_path": "voices-v1.0.bin",
                "voice": "pf_dora",
                "language": "pt-br",
                "speed": 1.25
            }
        },
        "voice": {
            "claude_mode": False,
            "event_loop": {
                "local": True,
                "pubsub_buffer_size": 100
            },
            "llm": {
                "call_mode": "single",
                "provider": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "qwen3:14b"
                },
                "sentence_chunking": True,
                "streaming": True
            },
            "mode": "event_driven",
            "vad": {
                "enabled": False,
                "min_speech_duration_ms": 250,
                "model_ttl": -1,
                "silence_duration_ms": 700,
                "threshold": 0.3
            }
        },
        "overlay": {
            "enabled": True,
            "padding": 20,
            "position": "top-right",
            "size": 15
        },
        "service": {
            "auto_start": True,
            "log_file": "",
            "log_level": "INFO",
            "notifications": True
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1
        },
        "logging": {
            "level": "INFO",
            "structured": True,
            "run_retention": 5,
            "trace_main_loop": False,
            "trace_threads": False
        },
        "tray": {
            "enabled": True,
            "tooltip": "Dictator - Voice to Text"
        },
        "advanced": {
            "cache_dir": "",
            "temp_dir": ""
        }
    }

    def __init__(self):
        """Initialize config generator"""
        if not YAML_AVAILABLE:
            raise RuntimeError("pyyaml not available - required for config generation")

        self.config = self._deep_copy_dict(self.DEFAULT_CONFIG)

    def _deep_copy_dict(self, d: dict) -> dict:
        """Deep copy a dictionary"""
        import copy
        return copy.deepcopy(d)

    def set_whisper_model(self, model: str):
        """
        Set Whisper model

        Args:
            model: Model name (tiny, small, medium, large, large-v3)
        """
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v3"]
        if model not in valid_models:
            logger.warning(f"Invalid Whisper model: {model}, using large-v3")
            model = "large-v3"

        self.config["whisper"]["model"] = model
        logger.info(f"Whisper model set to: {model}")

    def set_language(self, language: str):
        """
        Set transcription language

        Args:
            language: Language code (pt, en, es, fr, etc.)
        """
        self.config["whisper"]["language"] = language
        logger.info(f"Language set to: {language}")

    def set_device(self, device: str):
        """
        Set computation device

        Args:
            device: "cuda" for GPU or "cpu" for CPU
        """
        if device not in ["cuda", "cpu"]:
            logger.warning(f"Invalid device: {device}, using cpu")
            device = "cpu"

        self.config["whisper"]["device"] = device
        logger.info(f"Device set to: {device}")

    def set_hotkey_type(self, hotkey_type: str, button_or_key: str = "side1"):
        """
        Set hotkey type and button/key

        Args:
            hotkey_type: "mouse" or "keyboard"
            button_or_key: Mouse button (side1/side2/middle) or keyboard combo
        """
        if hotkey_type not in ["mouse", "keyboard"]:
            logger.warning(f"Invalid hotkey type: {hotkey_type}, using mouse")
            hotkey_type = "mouse"

        self.config["hotkey"]["type"] = hotkey_type

        if hotkey_type == "mouse":
            self.config["hotkey"]["mouse_button"] = button_or_key
        else:
            self.config["hotkey"]["keyboard_trigger"] = button_or_key

        logger.info(f"Hotkey set to: {hotkey_type} - {button_or_key}")

    def set_recording_mode(self, mode: str):
        """
        Set recording mode

        Args:
            mode: "toggle" or "push_to_talk"
        """
        if mode not in ["toggle", "push_to_talk"]:
            logger.warning(f"Invalid mode: {mode}, using toggle")
            mode = "toggle"

        self.config["hotkey"]["mode"] = mode
        logger.info(f"Recording mode set to: {mode}")

    def set_auto_paste(self, enabled: bool):
        """
        Enable/disable auto-paste

        Args:
            enabled: True to auto-paste, False to clipboard only
        """
        self.config["paste"]["auto_paste"] = enabled
        logger.info(f"Auto-paste: {enabled}")

    def enable_tts(self, enabled: bool, model_path: str = "", voices_path: str = ""):
        """
        Enable/disable TTS

        Args:
            enabled: True to enable TTS
            model_path: Path to kokoro model (if custom)
            voices_path: Path to voices file (if custom)
        """
        self.config["tts"]["enabled"] = enabled

        if enabled and model_path:
            self.config["tts"]["kokoro"]["model_path"] = model_path
        if enabled and voices_path:
            self.config["tts"]["kokoro"]["voices_path"] = voices_path

        logger.info(f"TTS enabled: {enabled}")

    def set_tts_voice(self, voice: str):
        """
        Set TTS voice

        Args:
            voice: Voice ID (e.g., "pf_dora", "pm_alex")
        """
        self.config["tts"]["kokoro"]["voice"] = voice
        logger.info(f"TTS voice set to: {voice}")

    def set_tts_speed(self, speed: float):
        """
        Set TTS speed

        Args:
            speed: Speed multiplier (0.5 - 2.0)
        """
        speed = max(0.5, min(2.0, speed))  # Clamp to valid range
        self.config["tts"]["kokoro"]["speed"] = speed
        logger.info(f"TTS speed set to: {speed}")

    def enable_llm(self, enabled: bool, provider: str = "ollama"):
        """
        Enable/disable LLM integration

        Args:
            enabled: True to enable voice assistant mode
            provider: LLM provider (ollama, claude_direct, n8n_toolcalling)
        """
        self.config["voice"]["claude_mode"] = enabled

        if enabled:
            self.config["voice"]["llm"]["provider"] = provider

        logger.info(f"LLM enabled: {enabled} (provider: {provider})")

    def set_llm_model(self, model: str):
        """
        Set LLM model (for Ollama)

        Args:
            model: Model name (e.g., "qwen3:14b", "llama3:8b")
        """
        self.config["voice"]["llm"]["ollama"]["model"] = model
        logger.info(f"LLM model set to: {model}")

    def enable_vad(self, enabled: bool):
        """
        Enable/disable Voice Activity Detection

        Args:
            enabled: True to enable auto-stop on silence
        """
        self.config["hotkey"]["vad_enabled"] = enabled
        self.config["voice"]["vad"]["enabled"] = enabled
        logger.info(f"VAD enabled: {enabled}")

    def set_log_level(self, level: str):
        """
        Set logging level

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if level not in valid_levels:
            logger.warning(f"Invalid log level: {level}, using INFO")
            level = "INFO"

        self.config["logging"]["level"] = level
        self.config["service"]["log_level"] = level
        logger.info(f"Log level set to: {level}")

    def apply_feature_selection(self, features: Dict[str, Any]):
        """
        Apply feature selection from user choices

        Args:
            features: Dictionary of feature flags
                - gpu_enabled: bool
                - tts_enabled: bool
                - llm_enabled: bool
                - whisper_model: str
                - etc.
        """
        # GPU
        if "gpu_enabled" in features:
            device = "cuda" if features["gpu_enabled"] else "cpu"
            self.set_device(device)

        # TTS
        if "tts_enabled" in features:
            self.enable_tts(features["tts_enabled"])

        # LLM
        if "llm_enabled" in features:
            self.enable_llm(features["llm_enabled"])

        # Whisper model
        if "whisper_model" in features:
            self.set_whisper_model(features["whisper_model"])

        # Hotkey
        if "hotkey_type" in features:
            button = features.get("hotkey_button", "side1")
            self.set_hotkey_type(features["hotkey_type"], button)

        # Recording mode
        if "recording_mode" in features:
            self.set_recording_mode(features["recording_mode"])

        # Auto-paste
        if "auto_paste" in features:
            self.set_auto_paste(features["auto_paste"])

        # VAD
        if "vad_enabled" in features:
            self.enable_vad(features["vad_enabled"])

        logger.info("Feature selection applied to configuration")

    def generate_config_file(self, output_path: Path) -> bool:
        """
        Generate config.yaml file

        Args:
            output_path: Path where config.yaml will be written

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write YAML file
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )

            logger.info(f"Configuration file generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate config file: {e}")
            return False

    def get_config(self) -> dict:
        """
        Get current configuration as dictionary

        Returns:
            Configuration dictionary
        """
        return self._deep_copy_dict(self.config)

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate current configuration

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        if "whisper" not in self.config:
            errors.append("Missing 'whisper' section")

        if "hotkey" not in self.config:
            errors.append("Missing 'hotkey' section")

        # Validate whisper model
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v3"]
        if self.config["whisper"]["model"] not in valid_models:
            errors.append(f"Invalid whisper model: {self.config['whisper']['model']}")

        # Validate device
        if self.config["whisper"]["device"] not in ["cuda", "cpu"]:
            errors.append(f"Invalid device: {self.config['whisper']['device']}")

        # Validate hotkey type
        if self.config["hotkey"]["type"] not in ["mouse", "keyboard"]:
            errors.append(f"Invalid hotkey type: {self.config['hotkey']['type']}")

        is_valid = len(errors) == 0
        return is_valid, errors
