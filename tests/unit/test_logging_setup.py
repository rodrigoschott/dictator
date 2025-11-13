#!/usr/bin/env python3
"""Tests for the logging bootstrap utilities.

Plan
====
1. Verify per-run directory creation and state tracking via `bootstrap_logging`.
2. Ensure structured logging writes JSON lines and run-level log file.
3. Confirm retention trimming removes older run directories.
4. Validate environment overrides for run directory and tracing flags.
"""

import importlib
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

logging_setup = importlib.import_module("dictator.logging_setup")

TestFunc = Callable[[], None]


def _reset_logging_state() -> None:
    """Reset global logging state after each test."""

    logging.shutdown()
    logging_setup._CURRENT_STATE = None  # type: ignore[attr-defined]


@contextmanager
def _temp_workdir() -> Iterator[Path]:
    """Create a temporary working directory and chdir into it."""

    with tempfile.TemporaryDirectory() as tmp:
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp)
            yield Path(tmp)
        finally:
            os.chdir(original_cwd)


@contextmanager
def _modified_environ(**updates: str) -> Iterator[None]:
    """Temporarily set environment variables."""

    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _flush_log_handlers() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        try:
            handler.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_bootstrap_creates_run_dir_and_files() -> None:
    with _temp_workdir():
        config = {
            "service": {"log_level": "INFO"},
            "logging": {"structured": False, "run_retention": 5},
        }

        state = logging_setup.bootstrap_logging(config)
        run_dir = state.run_dir

        assert run_dir.exists(), "Run directory was not created"
        assert logging_setup.get_current_run_dir() == run_dir

        log_file = run_dir / "dictator.log"
        assert log_file.exists(), "Primary log file missing"

        _reset_logging_state()


def test_structured_logging_writes_json() -> None:
    message = "Structured logging test"

    with _temp_workdir():
        config = {
            "service": {"log_level": "INFO"},
            "logging": {"structured": True, "run_retention": 5},
        }
        state = logging_setup.bootstrap_logging(config)

        logger = logging.getLogger("DictatorService")
        logger.info(message)
        _flush_log_handlers()

        _reset_logging_state()

        json_path = state.run_dir / "dictator.jsonl"
        assert json_path.exists(), "Structured JSON log file missing"

        lines = json_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines, "Structured log file is empty"

        payload = json.loads(lines[-1])
        assert payload["message"] == message
        assert payload["level"] == "INFO"

        # Ensure plain log also recorded
        plain_path = state.run_dir / "dictator.log"
        assert message in plain_path.read_text(encoding="utf-8")


def test_retention_enforces_limit() -> None:
    with _temp_workdir():
        logs_dir = Path("logs")
        logs_dir.mkdir()

        # Pre-create older runs
        for suffix in ["20240101-000000", "20240101-010000", "20240101-020000"]:
            (logs_dir / f"run-{suffix}").mkdir()

        state = logging_setup.bootstrap_logging(
            {
                "service": {"log_level": "INFO"},
                "logging": {"run_retention": 2},
            }
        )

        remaining = sorted(str(p.name) for p in logs_dir.iterdir())
        assert len(remaining) == 2, f"Expected 2 runs retained, got {remaining}"
        assert state.run_dir.name in remaining
        assert "run-20240101-000000" not in remaining

        _reset_logging_state()


def test_env_override_for_run_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        override_dir = Path(tmp) / "custom-run"
        with _modified_environ(DICTATOR_RUN_DIR=str(override_dir)):
            state = logging_setup.bootstrap_logging({"service": {"log_level": "INFO"}})
            assert state.run_dir == override_dir.resolve()
            assert override_dir.exists()

            _reset_logging_state()


def test_trace_flags_follow_environment() -> None:
    with _temp_workdir():
        with _modified_environ(
            DICTATOR_TRACE_MAIN_LOOP="1",
            DICTATOR_TRACE_THREADS="true",
        ):
            state = logging_setup.bootstrap_logging({"service": {"log_level": "INFO"}})
            assert state.trace_main_loop is True
            assert state.trace_threads is True
            assert logging_setup.is_main_loop_tracing_enabled() is True
            assert logging_setup.is_thread_tracing_enabled() is True

            _reset_logging_state()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

TESTS: List[Tuple[str, TestFunc]] = [
    ("bootstrap_creates_run_dir_and_files", test_bootstrap_creates_run_dir_and_files),
    ("structured_logging_writes_json", test_structured_logging_writes_json),
    ("retention_enforces_limit", test_retention_enforces_limit),
    ("env_override_for_run_dir", test_env_override_for_run_dir),
    ("trace_flags_follow_environment", test_trace_flags_follow_environment),
]


def main() -> int:
    print("🧪 Logging setup validation tests\n")

    failures = 0
    for name, func in TESTS:
        print(f"▶ Running {name}...")
        try:
            func()
        except AssertionError as exc:
            failures += 1
            print(f"  ❌ FAIL: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected
            failures += 1
            print(f"  ❌ ERROR: {exc}")
        else:
            print("  ✅ PASS")
        print()

    print("📊 Summary: {} passed, {} failed".format(len(TESTS) - failures, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
