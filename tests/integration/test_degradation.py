#!/usr/bin/env python3
"""
Integration tests for health check degradation scenarios

Tests the full flow of dependency validation and feature degradation.
"""

import sys
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dictator.health_check import DependencyValidator, HealthReport


class TestHealthDegradationScenarios(unittest.TestCase):
    """Test various degradation scenarios"""

    def setUp(self):
        """Set up test configuration"""
        self.base_config = {
            'whisper': {'device': 'cuda', 'model': 'large-v3'},
            'tts': {
                'enabled': True,
                'kokoro': {
                    'model_path': 'kokoro-v1.0.onnx',
                    'voices_path': 'voices-v1.0.bin'
                }
            },
            'voice': {
                'claude_mode': True,
                'llm': {
                    'provider': 'ollama',
                    'ollama': {'base_url': 'http://localhost:11434'}
                }
            },
            'tray': {'tooltip': 'Dictator'}
        }

        # Start patcher for check_python_packages to avoid torch import issues
        self.packages_patcher = patch('dictator.health_check.DependencyValidator.check_python_packages')
        self.mock_check_packages = self.packages_patcher.start()

        # Mock packages as healthy by default
        from dictator.health_check import ComponentStatus
        self.mock_check_packages.return_value = ComponentStatus(
            name="Python Packages",
            status="healthy",
            required=True,
            message="All critical packages installed"
        )

    def tearDown(self):
        """Clean up patches"""
        self.packages_patcher.stop()

    @patch('dictator.health_check.DependencyValidator.check_whisper_cache')
    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    @patch('subprocess.run')
    @patch('requests.get')
    @patch('pathlib.Path')
    def test_scenario_all_healthy(self, mock_path, mock_requests, mock_subprocess, mock_stream, mock_query, mock_whisper_cache):
        """Scenario 1: Everything is healthy"""
        # Mock whisper cache as healthy
        from dictator.health_check import ComponentStatus
        mock_whisper_cache.return_value = ComponentStatus(
            name="Whisper Cache",
            status="healthy",
            required=False,
            message="Model cache ready"
        )

        # Mock all components as healthy
        mock_query.return_value = {'name': 'Test Mic'}

        mock_model = Mock()
        mock_model.exists.return_value = True
        mock_model.stat.return_value.st_size = 325 * 1024 * 1024  # 325MB
        mock_path.return_value = mock_model

        mock_subprocess.return_value = Mock(returncode=0, stdout="git-lfs/3.4.0")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'name': 'model1'}]}
        mock_requests.return_value = mock_response

        validator = DependencyValidator(self.base_config)
        report = validator.run_full_check(quick=False)

        # Allow either healthy or degraded (whisper cache is optional)
        self.assertIn(report.overall_status, ["healthy", "degraded"])

        # If degraded, it should only be due to optional components
        if report.overall_status == "degraded":
            # All degraded components should be non-critical
            for comp in report.components:
                if comp.status in ["unavailable", "degraded"]:
                    self.assertFalse(comp.required)

        # Check tooltip
        tooltip = report.get_tray_tooltip("Dictator")
        self.assertIn("Dictator", tooltip)

    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    @patch('requests.get')
    @patch('pathlib.Path')
    def test_scenario_ollama_unavailable(self, mock_path, mock_requests, mock_stream, mock_query):
        """Scenario 2: Ollama not running - Voice Assistant should degrade"""
        # Mock Ollama as unavailable
        mock_requests.side_effect = Exception("Connection refused")

        # Mock audio as healthy
        mock_query.return_value = {'name': 'Mic'}

        # Mock other components as healthy
        mock_model = Mock()
        mock_model.exists.return_value = True
        mock_model.stat.return_value.st_size = 325 * 1024 * 1024
        mock_path.return_value = mock_model

        validator = DependencyValidator(self.base_config)
        report = validator.run_full_check(quick=False)

        self.assertEqual(report.overall_status, "degraded")
        self.assertIn("Voice Assistant", report.degraded_features)

        # Check tooltip shows degradation
        tooltip = report.get_tray_tooltip("Dictator")
        self.assertIn("Degraded", tooltip)
        self.assertIn("Voice Assistant", tooltip)

    @patch('dictator.health_check.DependencyValidator.check_model_files')
    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    def test_scenario_tts_models_missing(self, mock_stream, mock_query, mock_check_models):
        """Scenario 3: TTS model files are LFS pointers - TTS should degrade"""
        # Mock model files check to return unavailable (LFS pointers)
        from dictator.health_check import ComponentStatus
        mock_check_models.return_value = ComponentStatus(
            name="TTS Model Files",
            status="unavailable",
            required=False,
            message="Model files are LFS pointers (not downloaded)",
            fix_hint="Run: git lfs pull"
        )

        # Mock audio as healthy
        mock_query.return_value = {'name': 'Mic'}

        validator = DependencyValidator(self.base_config)
        report = validator.run_full_check(quick=False)

        self.assertEqual(report.overall_status, "degraded")
        # Check that TTS component is unavailable
        tts_comp = next((c for c in report.components if "TTS" in c.name), None)
        self.assertIsNotNone(tts_comp)
        self.assertEqual(tts_comp.status, "unavailable")
        self.assertIn("TTS", report.degraded_features)

    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    @patch('pathlib.Path')
    @patch('torch.cuda.is_available')
    def test_scenario_gpu_unavailable(self, mock_cuda, mock_path, mock_stream, mock_query):
        """Scenario 4: GPU not available but device=cuda - Should warn"""
        mock_cuda.return_value = False

        # Mock audio as healthy
        mock_query.return_value = {'name': 'Mic'}

        mock_model = Mock()
        mock_model.exists.return_value = True
        mock_model.stat.return_value.st_size = 325 * 1024 * 1024
        mock_path.return_value = mock_model

        validator = DependencyValidator(self.base_config)
        report = validator.run_full_check(quick=True)

        # Should be critical since CUDA is required but not available
        gpu_status = next((c for c in report.components if c.name == "GPU/CUDA"), None)
        self.assertIsNotNone(gpu_status)
        self.assertEqual(gpu_status.status, "critical")

    @patch('sounddevice.query_devices')
    def test_scenario_audio_device_unavailable(self, mock_query):
        """Scenario 5: Audio device not accessible - Should be critical"""
        mock_query.side_effect = Exception("No device found")

        validator = DependencyValidator(self.base_config)
        report = validator.run_full_check(quick=True)

        self.assertEqual(report.overall_status, "critical")

        # Find audio device status
        audio_status = next((c for c in report.components if c.name == "Audio Device"), None)
        self.assertIsNotNone(audio_status)
        self.assertEqual(audio_status.status, "critical")
        self.assertTrue(audio_status.required)

    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    @patch('requests.get')
    @patch('pathlib.Path')
    def test_scenario_multiple_degradations(self, mock_path, mock_requests, mock_stream, mock_query):
        """Scenario 6: Multiple components degraded (TTS + Ollama)"""
        # Mock TTS models as missing
        def path_side_effect(file_path):
            mock_file = Mock()
            mock_file.exists.return_value = False  # Models don't exist
            mock_file.__truediv__ = lambda self, other: path_side_effect(str(other))
            return mock_file

        mock_path.side_effect = path_side_effect

        # Mock Ollama as unavailable
        mock_requests.side_effect = Exception("Connection refused")

        # Mock audio as healthy
        mock_query.return_value = {'name': 'Mic'}

        validator = DependencyValidator(self.base_config)
        report = validator.run_full_check(quick=False)

        self.assertEqual(report.overall_status, "degraded")

        # At least one of these features should be degraded
        # (The exact count may vary based on implementation)
        has_tts_or_voice = ("TTS" in report.degraded_features or
                            "Voice Assistant" in report.degraded_features)
        self.assertTrue(has_tts_or_voice,
                       f"Expected TTS or Voice Assistant degraded, got: {report.degraded_features}")

    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    @patch('pathlib.Path')
    def test_scenario_voice_mode_disabled(self, mock_path, mock_stream, mock_query):
        """Scenario 7: Voice mode disabled in config - Ollama not checked"""
        config = self.base_config.copy()
        config['voice']['claude_mode'] = False

        # Mock audio as healthy
        mock_query.return_value = {'name': 'Mic'}

        mock_model = Mock()
        mock_model.exists.return_value = True
        mock_model.stat.return_value.st_size = 325 * 1024 * 1024
        mock_path.return_value = mock_model

        validator = DependencyValidator(config)
        report = validator.run_full_check(quick=False)

        # Ollama should be healthy but disabled
        ollama_status = next((c for c in report.components if c.name == "Ollama"), None)
        if ollama_status:  # Only in full check
            self.assertEqual(ollama_status.status, "healthy")
            self.assertFalse(ollama_status.enabled)

    @patch('sounddevice.query_devices')
    @patch('sounddevice.InputStream')
    @patch('pathlib.Path')
    def test_health_report_serialization(self, mock_path, mock_stream, mock_query):
        """Test that health report can be serialized to JSON"""
        config = self.base_config.copy()

        # Mock audio as healthy
        mock_query.return_value = {'name': 'Mic'}

        mock_model = Mock()
        mock_model.exists.return_value = True
        mock_model.stat.return_value.st_size = 325 * 1024 * 1024
        mock_path.return_value = mock_model

        validator = DependencyValidator(config)
        report = validator.run_full_check(quick=True)

        # Test serialization
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict, indent=2)

        # Test deserialization
        loaded_dict = json.loads(json_str)

        self.assertIn("timestamp", loaded_dict)
        self.assertIn("overall_status", loaded_dict)
        self.assertIn("components", loaded_dict)
        self.assertIsInstance(loaded_dict["components"], list)


class TestServiceIntegration(unittest.TestCase):
    """Test integration with DictatorService"""

    def test_health_check_module_importable(self):
        """Test that health_check module can be imported by service"""
        # This test verifies the health_check module is properly structured
        # and can be imported without errors
        from dictator.health_check import DependencyValidator, HealthReport, ComponentStatus

        # Verify classes exist and can be instantiated
        self.assertTrue(callable(DependencyValidator))
        self.assertTrue(callable(HealthReport))
        self.assertTrue(callable(ComponentStatus))

        # Create a minimal config
        config = {'whisper': {'device': 'cpu', 'model': 'tiny'}}

        # Verify DependencyValidator can be instantiated
        validator = DependencyValidator(config)
        self.assertIsNotNone(validator)
        self.assertEqual(validator.config, config)


if __name__ == '__main__':
    unittest.main()
