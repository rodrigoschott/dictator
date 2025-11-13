#!/usr/bin/env python3
"""Tests for DictatorService logging integration.

Plan
====
1. Ensure `setup_logging` stores logging state and skips thread monitor when
   tracing is disabled.
2. Verify that enabling thread tracing instantiates and starts a monitor with
   the configured interval.
"""

import ast
import importlib
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

logging_setup = importlib.import_module("dictator.logging_setup")

SERVICE_PATH = REPO_ROOT / "src" / "dictator" / "service.py"

_SERVICE_NAMESPACE = {
    "logging_setup": logging_setup,
    "logging": logging,
    "ThreadMonitor": None,
}


def _load_service_logging_harness() -> type:
    source = SERVICE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SERVICE_PATH))

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DictatorService"
    )

    body_nodes: List[ast.stmt] = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"setup_logging", "_resolve_thread_monitor_interval"}
    ]

    harness_class = ast.ClassDef(
        name="ServiceLoggingHarness",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=body_nodes,
    )

    module = ast.Module(body=[harness_class], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SERVICE_PATH), "exec"), _SERVICE_NAMESPACE)
    return _SERVICE_NAMESPACE["ServiceLoggingHarness"]


ServiceLoggingHarness = _load_service_logging_harness()


@contextmanager
def _temp_workdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp)
            yield Path(tmp)
        finally:
            os.chdir(original_cwd)


def _reset_logging_state() -> None:
    logging.shutdown()
    logging_setup._CURRENT_STATE = None  # type: ignore[attr-defined]


def _set_thread_monitor(cls: Any) -> None:
    _SERVICE_NAMESPACE["ThreadMonitor"] = cls


def _make_service(config: dict) -> Any:
    svc = ServiceLoggingHarness()
    svc.config = config
    svc.thread_monitor = None
    svc.logging_state = None
    svc.run_dir = None
    svc.logger = logging.getLogger("DictatorService")
    return svc


def test_setup_logging_without_trace() -> None:
    config = {
        "service": {"log_level": "INFO"},
        "logging": {"trace_threads": False},
        "profiling": {"thread_monitor_interval": 5},
    }

    with _temp_workdir():
        _set_thread_monitor(None)
        svc = _make_service(config)
        svc.setup_logging()

        assert svc.logging_state is not None
        assert svc.logging_state.trace_threads is False
        assert svc.thread_monitor is None
        assert svc.run_dir and svc.run_dir.exists()

        _reset_logging_state()


class DummyMonitor:
    def __init__(self, interval_seconds: float, logger: logging.Logger, run_dir: Optional[Path]):
        self.interval = interval_seconds
        self.logger = logger
        self.run_dir = run_dir
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float | None = None) -> None:
        self.stopped = True


def test_setup_logging_with_trace() -> None:
    config = {
        "service": {"log_level": "DEBUG"},
        "logging": {"trace_threads": True},
        "profiling": {"thread_monitor_interval": 7},
    }

    with _temp_workdir():
        _set_thread_monitor(DummyMonitor)
        svc = _make_service(config)
        svc.setup_logging()

        assert isinstance(svc.thread_monitor, DummyMonitor)
        assert svc.thread_monitor.started is True  # type: ignore[union-attr]
        assert svc.thread_monitor.interval == 7  # type: ignore[union-attr]
        assert svc.logging_state and svc.logging_state.trace_threads is True

        _reset_logging_state()
        _set_thread_monitor(None)


TestFunc = Callable[[], None]
TESTS: tuple[TestFunc, ...] = (
    test_setup_logging_without_trace,
    test_setup_logging_with_trace,
)


def main() -> int:
    print("🧪 DictatorService logging tests\n")
    failures = 0
    for func in TESTS:
        print(f"▶ Running {func.__name__}...")
        try:
            func()
        except AssertionError as exc:
            print(f"  ❌ FAIL: {exc}")
            failures += 1
        except Exception as exc:  # pragma: no cover
            print(f"  ❌ ERROR: {exc}")
            failures += 1
        else:
            print("  ✅ PASS")
        print()

    print(f"📊 Summary: {len(TESTS) - failures} passed, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
