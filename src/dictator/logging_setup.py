"""Centralized logging bootstrap and lightweight timing helpers."""

from __future__ import annotations

import json
import logging
import logging.config
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


_RUN_DIR_ENV = "DICTATOR_RUN_DIR"
_TRACE_MAIN_LOOP_ENV = "DICTATOR_TRACE_MAIN_LOOP"
_TRACE_THREADS_ENV = "DICTATOR_TRACE_THREADS"

_CURRENT_STATE: Optional["LoggingState"] = None


@dataclass(slots=True)
class LoggingState:
    """Holds runtime logging options derived from configuration."""

    run_dir: Path
    structured: bool
    trace_main_loop: bool
    trace_threads: bool


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logging files."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - short doc
        data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            data["stack"] = self.formatStack(record.stack_info)

        return json.dumps(data, ensure_ascii=True)


class TimingContext:
    """Context manager to measure and optionally log slow operations."""

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        *,
        enabled: bool = True,
        warn_threshold_ms: float | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self._logger = logger
        self._label = label
        self._enabled = enabled
        self._warn_threshold = warn_threshold_ms
        self._level = log_level
        self._start = 0.0

    def __enter__(self) -> "TimingContext":
        if self._enabled:
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if not self._enabled:
            return None

        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if self._warn_threshold is None or elapsed_ms >= self._warn_threshold:
            self._logger.log(
                self._level,
                "%s completed in %.2f ms",
                self._label,
                elapsed_ms,
            )
        return None


def bootstrap_logging(config: dict[str, Any]) -> LoggingState:
    """Configure logging based on config dict and environment overrides."""

    logging_cfg = config.get("logging", {}) or {}
    service_cfg = config.get("service", {}) or {}

    level_name = str(logging_cfg.get("level") or service_cfg.get("log_level") or "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    structured = bool(logging_cfg.get("structured", False))

    trace_main_loop = _env_flag(
        _TRACE_MAIN_LOOP_ENV,
        logging_cfg.get("trace_main_loop", False),
    )
    trace_threads = _env_flag(
        _TRACE_THREADS_ENV,
        logging_cfg.get("trace_threads", False),
    )

    retention = int(logging_cfg.get("run_retention", 5))
    run_dir = ensure_run_directory(retention)

    log_file_name = service_cfg.get("log_file") or "dictator.log"
    log_file_path = run_dir / Path(log_file_name).name

    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "plain",
            "filename": str(log_file_path),
            "encoding": "utf-8",
        },
    }

    if structured:
        handlers["json"] = {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": str(run_dir / "dictator.jsonl"),
            "encoding": "utf-8",
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "plain": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                },
                "json": {
                    "()": JsonFormatter,
                },
            },
            "handlers": handlers,
            "root": {
                "level": level,
                "handlers": list(handlers.keys()),
            },
        }
    )

    state = LoggingState(
        run_dir=run_dir,
        structured=structured,
        trace_main_loop=trace_main_loop,
        trace_threads=trace_threads,
    )

    global _CURRENT_STATE
    _CURRENT_STATE = state

    logging.getLogger(__name__).debug("Logging initialized in %s", run_dir)
    return state


def ensure_run_directory(retention: int) -> Path:
    """Create per-run log directory and enforce retention."""

    base = Path("logs")
    base.mkdir(exist_ok=True)

    env_override = os.environ.get(_RUN_DIR_ENV)
    if env_override:
        run_dir = Path(env_override).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        (d for d in base.iterdir() if d.is_dir() and d.name.startswith("run-")),
        key=lambda path: path.name,
    )

    if retention > 0:
        while len(run_dirs) > retention:
            oldest = run_dirs.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)

    return run_dir


def get_current_run_dir() -> Optional[Path]:
    """Return the active log run directory if logging was bootstrapped."""

    return _CURRENT_STATE.run_dir if _CURRENT_STATE else None


def is_main_loop_tracing_enabled() -> bool:
    """Flag indicating whether main loop tracing is active."""

    return bool(_CURRENT_STATE and _CURRENT_STATE.trace_main_loop)


def is_thread_tracing_enabled() -> bool:
    """Flag indicating whether thread monitoring is active."""

    return bool(_CURRENT_STATE and _CURRENT_STATE.trace_threads)


def _env_flag(name: str, default: Any) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}
