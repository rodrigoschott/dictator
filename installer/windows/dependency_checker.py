"""
Dependency Checker

Pre-flight system validation for installer.
Checks all requirements before installation begins.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.validation import SystemRequirements, validate_path_writable

logger = logging.getLogger("DependencyChecker")


@dataclass
class CheckResult:
    """Result of a single dependency check"""
    name: str
    passed: bool
    required: bool  # If false, this is optional/recommended
    status: str  # "passed", "warning", "failed"
    message: str
    details: Dict = field(default_factory=dict)
    fix_hint: str = ""


class DependencyChecker:
    """
    Pre-flight dependency checker

    Validates all system requirements before installation:
    - Operating system compatibility
    - Python version
    - RAM and disk space
    - Network connectivity
    - GPU availability (optional)
    - Administrator rights
    - Path permissions
    """

    def __init__(self, install_dir: Path):
        """
        Initialize dependency checker

        Args:
            install_dir: Planned installation directory
        """
        self.install_dir = install_dir
        self.results: List[CheckResult] = []

    def check_all(self) -> Tuple[bool, List[CheckResult]]:
        """
        Run all dependency checks

        Returns:
            (all_passed, results)
        """
        logger.info("Running pre-flight dependency checks...")

        self.results = []

        # Critical checks
        self._check_os()
        self._check_python_version()
        self._check_admin_rights()
        self._check_ram()
        self._check_disk_space()
        self._check_path_writable()

        # Optional but recommended checks
        self._check_network()
        self._check_gpu()

        # Determine overall result
        critical_failures = [r for r in self.results if not r.passed and r.required]
        all_passed = len(critical_failures) == 0

        if all_passed:
            logger.info(" All pre-flight checks passed")
        else:
            logger.error(f" {len(critical_failures)} critical check(s) failed")

        return all_passed, self.results

    def _check_os(self):
        """Check operating system compatibility"""
        passed, message, details = SystemRequirements.check_os()

        result = CheckResult(
            name="Operating System",
            passed=passed,
            required=True,
            status="passed" if passed else "failed",
            message=message,
            details=details,
            fix_hint="Windows 10 or later is required" if not passed else ""
        )
        self.results.append(result)
        logger.info(f"OS Check: {message}")

    def _check_python_version(self):
        """Check Python version compatibility"""
        passed, message, details = SystemRequirements.check_python_version()

        result = CheckResult(
            name="Python Version",
            passed=passed,
            required=True,
            status="passed" if passed else "failed",
            message=message,
            details=details,
            fix_hint="Install Python 3.10-3.13 from https://www.python.org" if not passed else ""
        )
        self.results.append(result)
        logger.info(f"Python Check: {message}")

    def _check_admin_rights(self):
        """Check administrator/root privileges"""
        passed, message, details = SystemRequirements.check_admin_rights()

        result = CheckResult(
            name="Administrator Rights",
            passed=passed,
            required=True,
            status="passed" if passed else "failed",
            message=message,
            details=details,
            fix_hint="Right-click installer and select 'Run as Administrator'" if not passed else ""
        )
        self.results.append(result)
        logger.info(f"Admin Check: {message}")

    def _check_ram(self):
        """Check available RAM"""
        passed, message, details = SystemRequirements.check_ram()

        # RAM check is warning-only if below recommended but above minimum
        total_gb = details.get('total_gb', 0)
        required = total_gb < SystemRequirements.MIN_RAM_GB

        status = "passed"
        if not passed and required:
            status = "failed"
        elif not passed:
            status = "warning"

        result = CheckResult(
            name="RAM",
            passed=passed or not required,
            required=required,
            status=status,
            message=message,
            details=details,
            fix_hint=f"At least {SystemRequirements.MIN_RAM_GB}GB RAM required" if required else ""
        )
        self.results.append(result)
        logger.info(f"RAM Check: {message}")

    def _check_disk_space(self):
        """Check available disk space"""
        passed, message, details = SystemRequirements.check_disk_space(self.install_dir)

        # Disk check is warning-only if below recommended but above minimum
        free_gb = details.get('free_gb', 0)
        required = free_gb < SystemRequirements.MIN_DISK_GB

        status = "passed"
        if not passed and required:
            status = "failed"
        elif not passed:
            status = "warning"

        result = CheckResult(
            name="Disk Space",
            passed=passed or not required,
            required=required,
            status=status,
            message=message,
            details=details,
            fix_hint=f"At least {SystemRequirements.MIN_DISK_GB}GB free space required" if required else ""
        )
        self.results.append(result)
        logger.info(f"Disk Check: {message}")

    def _check_path_writable(self):
        """Check if installation path is writable"""
        writable, error_msg = validate_path_writable(self.install_dir)

        message = f"[OK] Installation path writable" if writable else f"[ERRO] {error_msg}"

        result = CheckResult(
            name="Installation Path",
            passed=writable,
            required=True,
            status="passed" if writable else "failed",
            message=message,
            details={"path": str(self.install_dir)},
            fix_hint="Choose a different installation directory or run as Administrator" if not writable else ""
        )
        self.results.append(result)
        logger.info(f"Path Check: {message}")

    def _check_network(self):
        """Check network connectivity (optional)"""
        passed, message, details = SystemRequirements.check_network_connectivity(timeout=5)

        result = CheckResult(
            name="Network Connectivity",
            passed=passed,
            required=False,  # Network is optional (can install offline with pre-downloaded models)
            status="passed" if passed else "warning",
            message=message,
            details=details,
            fix_hint="Network required to download model files. You can skip model download and add them later." if not passed else ""
        )
        self.results.append(result)
        logger.info(f"Network Check: {message}")

    def _check_gpu(self):
        """Check GPU availability (optional)"""
        passed, message, details = SystemRequirements.check_gpu_availability()

        # GPU is always optional - CPU mode is available
        result = CheckResult(
            name="GPU Acceleration",
            passed=True,  # Always pass since CPU mode works
            required=False,
            status="passed" if passed else "warning",
            message=message,
            details=details,
            fix_hint="Install NVIDIA drivers from https://www.nvidia.com/Download/index.aspx for better performance" if not passed else ""
        )
        self.results.append(result)
        logger.info(f"GPU Check: {message}")

    def get_summary(self) -> Dict:
        """
        Get summary of all checks

        Returns:
            Dictionary with summary statistics
        """
        total = len(self.results)
        passed = len([r for r in self.results if r.passed])
        warnings = len([r for r in self.results if r.status == "warning"])
        failed = len([r for r in self.results if r.status == "failed"])
        critical_failed = len([r for r in self.results if not r.passed and r.required])

        return {
            "total_checks": total,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "critical_failures": critical_failed,
            "can_proceed": critical_failed == 0
        }

    def get_failed_checks(self) -> List[CheckResult]:
        """Get list of failed critical checks"""
        return [r for r in self.results if not r.passed and r.required]

    def get_warning_checks(self) -> List[CheckResult]:
        """Get list of warning checks"""
        return [r for r in self.results if r.status == "warning"]

    def format_results(self) -> str:
        """
        Format results as human-readable text

        Returns:
            Formatted string of all check results
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  PRE-FLIGHT DEPENDENCY CHECK")
        lines.append("=" * 70)
        lines.append("")

        # Group by status
        passed_checks = [r for r in self.results if r.status == "passed"]
        warning_checks = [r for r in self.results if r.status == "warning"]
        failed_checks = [r for r in self.results if r.status == "failed"]

        # Passed
        if passed_checks:
            lines.append("[PASSED]")
            for result in passed_checks:
                lines.append(f"  {result.message}")
            lines.append("")

        # Warnings
        if warning_checks:
            lines.append("[WARNINGS]")
            for result in warning_checks:
                lines.append(f"  {result.message}")
                if result.fix_hint:
                    lines.append(f"    → {result.fix_hint}")
            lines.append("")

        # Failures
        if failed_checks:
            lines.append("[FAILED]")
            for result in failed_checks:
                lines.append(f"  {result.message}")
                if result.fix_hint:
                    lines.append(f"    → Fix: {result.fix_hint}")
            lines.append("")

        # Summary
        summary = self.get_summary()
        lines.append("=" * 70)
        lines.append(f"Summary: {summary['passed']}/{summary['total_checks']} passed")

        if summary['can_proceed']:
            lines.append("[RESULT] [OK] Ready to install")
        else:
            lines.append(f"[RESULT] [ERRO] Cannot proceed - {summary['critical_failures']} critical failure(s)")

        lines.append("=" * 70)

        return "\n".join(lines)
