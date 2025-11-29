#!/usr/bin/env python3
"""
Unit tests for health_check module

Tests individual validators and health report functionality.
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dictator.health_check import (
    ComponentStatus,
    HealthReport,
    DependencyValidator
)


class TestComponentStatus(unittest.TestCase):
    """Test ComponentStatus dataclass"""

    def test_component_status_creation(self):
        """Test creating a ComponentStatus"""
        status = ComponentStatus(
            name="Test Component",
            status="healthy",
            required=True,
            message="All good"
        )

        self.assertEqual(status.name, "Test Component")
        self.assertEqual(status.status, "healthy")
        self.assertTrue(status.required)
        self.assertEqual(status.message, "All good")

    def test_component_status_to_dict(self):
        """Test serialization to dict"""
        status = ComponentStatus(
            name="Test",
            status="healthy",
            required=True,
            message="OK",
            fix_hint="No fix needed",
            details={"version": "1.0"}
        )

        result = status.to_dict()

        self.assertIn("name", result)
        self.assertIn("status", result)
        self.assertIn("required", result)
        self.assertIn("message", result)
        self.assertIn("fix_hint", result)
        self.assertIn("details", result)


class TestHealthReport(unittest.TestCase):
    """Test HealthReport dataclass"""

    def test_health_report_creation(self):
        """Test creating a HealthReport"""
        components = [
            ComponentStatus("Component1", "healthy", True, message="OK"),
            ComponentStatus("Component2", "degraded", False, message="Warning")
        ]

        report = HealthReport(
            timestamp="2025-11-29T12:00:00",
            overall_status="degraded",
            components=components,
            degraded_features=["Feature1"]
        )

        self.assertEqual(report.overall_status, "degraded")
        self.assertEqual(len(report.components), 2)
        self.assertEqual(len(report.degraded_features), 1)

    def test_health_report_to_dict(self):
        """Test serialization to dict"""
        components = [
            ComponentStatus("Test", "healthy", True, message="OK")
        ]

        report = HealthReport(
            timestamp="2025-11-29T12:00:00",
            overall_status="healthy",
            components=components,
            degraded_features=[]
        )

        result = report.to_dict()

        self.assertIn("timestamp", result)
        self.assertIn("overall_status", result)
        self.assertIn("components", result)
        self.assertIn("degraded_features", result)
        self.assertIsInstance(result["components"], list)

    def test_get_tray_tooltip_healthy(self):
        """Test tooltip generation for healthy status"""
        report = HealthReport(
            timestamp="2025-11-29T12:00:00",
            overall_status="healthy",
            components=[],
            degraded_features=[]
        )

        tooltip = report.get_tray_tooltip("Dictator")
        self.assertIn("Healthy", tooltip)

    def test_get_tray_tooltip_degraded(self):
        """Test tooltip generation for degraded status"""
        report = HealthReport(
            timestamp="2025-11-29T12:00:00",
            overall_status="degraded",
            components=[],
            degraded_features=["TTS", "Voice Assistant"]
        )

        tooltip = report.get_tray_tooltip("Dictator")
        self.assertIn("Degraded", tooltip)
        self.assertIn("TTS", tooltip)

    def test_get_tray_tooltip_critical(self):
        """Test tooltip generation for critical status"""
        report = HealthReport(
            timestamp="2025-11-29T12:00:00",
            overall_status="critical",
            components=[],
            degraded_features=[]
        )

        tooltip = report.get_tray_tooltip("Dictator")
        self.assertIn("Critical", tooltip)


class TestDependencyValidator(unittest.TestCase):
    """Test DependencyValidator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'whisper': {'device': 'cpu', 'model': 'large-v3'},
            'tts': {
                'enabled': True,
                'kokoro': {
                    'model_path': 'kokoro-v1.0.onnx',
                    'voices_path': 'voices-v1.0.bin'
                }
            },
            'voice': {
                'claude_mode': False,
                'llm': {
                    'provider': 'ollama',
                    'ollama': {'base_url': 'http://localhost:11434'}
                }
            }
        }
        self.validator = DependencyValidator(self.config)

    def test_check_python_packages_all_present(self):
        """Test when all packages are installed"""
        # This will actually check real packages
        status = self.validator.check_python_packages()

        self.assertEqual(status.name, "Python Packages")
        # Status depends on actual installation
        self.assertIn(status.status, ["healthy", "critical"])

    @patch('dictator.health_check.Path')
    def test_check_model_files_not_found(self, mock_path):
        """Test when model files don't exist"""
        mock_file = Mock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file

        status = self.validator.check_model_files()

        self.assertEqual(status.name, "TTS Model Files")
        self.assertEqual(status.status, "unavailable")
        self.assertIn("not found", status.message)

    @patch('dictator.health_check.Path')
    def test_check_model_files_lfs_pointer(self, mock_path):
        """Test when model files are LFS pointers"""
        # Mock model file
        mock_model = Mock()
        mock_model.exists.return_value = True
        mock_model.stat.return_value.st_size = 100  # Very small (LFS pointer)

        # Mock voices file
        mock_voices = Mock()
        mock_voices.exists.return_value = True

        def path_side_effect(arg):
            if 'onnx' in str(arg):
                return mock_model
            return mock_voices

        mock_path.side_effect = path_side_effect

        # Mock open to return LFS pointer content
        with patch('builtins.open', unittest.mock.mock_open(read_data=b'version https://git-lfs')):
            status = self.validator.check_model_files()

        self.assertEqual(status.status, "unavailable")
        self.assertIn("LFS pointer", status.message)

    @patch('dictator.health_check.subprocess.run')
    def test_check_git_lfs_installed(self, mock_run):
        """Test when Git LFS is installed"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "git-lfs/3.4.0"
        mock_run.return_value = mock_result

        status = self.validator.check_git_lfs()

        self.assertEqual(status.name, "Git LFS")
        self.assertEqual(status.status, "healthy")

    @patch('dictator.health_check.subprocess.run')
    def test_check_git_lfs_not_installed(self, mock_run):
        """Test when Git LFS is not installed"""
        mock_run.side_effect = FileNotFoundError()

        status = self.validator.check_git_lfs()

        self.assertEqual(status.status, "unavailable")
        self.assertIn("not found", status.message)

    def test_check_gpu_cuda_cpu_mode(self):
        """Test when CPU mode is configured"""
        self.config['whisper']['device'] = 'cpu'

        status = self.validator.check_gpu_cuda()

        self.assertEqual(status.status, "healthy")
        self.assertFalse(status.enabled)
        self.assertIn("CPU mode", status.message)

    def test_check_ollama_disabled(self):
        """Test when voice mode is disabled"""
        self.config['voice']['claude_mode'] = False

        status = self.validator.check_ollama()

        self.assertEqual(status.status, "healthy")
        self.assertFalse(status.enabled)

    def test_run_full_check(self):
        """Test running full health check"""
        report = self.validator.run_full_check(quick=True)

        self.assertIsInstance(report, HealthReport)
        self.assertIsInstance(report.timestamp, str)
        self.assertIn(report.overall_status, ["healthy", "degraded", "critical"])
        self.assertIsInstance(report.components, list)
        self.assertGreater(len(report.components), 0)

    def test_run_full_check_quick_mode(self):
        """Test quick mode skips optional checks"""
        report_quick = self.validator.run_full_check(quick=True)
        report_full = self.validator.run_full_check(quick=False)

        # Quick mode should have fewer components
        self.assertLessEqual(len(report_quick.components), len(report_full.components))


if __name__ == '__main__':
    unittest.main()
