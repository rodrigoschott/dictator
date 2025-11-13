"""Background thread monitor for logging thread and CPU statistics."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is an optional dependency
    psutil = None  # type: ignore[assignment]


class ThreadMonitor:
    """Periodic reporter for thread counts and CPU usage."""

    def __init__(self, interval_seconds: float, logger: logging.Logger, run_dir: Optional[Path] = None) -> None:
        self.interval_seconds = interval_seconds
        self.logger = logger
        self.run_dir = run_dir

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ThreadMonitor",
            daemon=True,
        )

    def start(self) -> None:
        """Start the monitoring thread if psutil is available."""

        if psutil is None:
            self.logger.warning("Thread monitor disabled: psutil not installed")
            return
        if self._thread.is_alive():
            return
        self.logger.debug("Starting thread monitor (interval=%.1fs)", self.interval_seconds)
        self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        """Signal the monitor to stop."""

        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run(self) -> None:
        if psutil is None:  # pragma: no cover - guard for safety
            return

        process = psutil.Process()
        while not self._stop_event.is_set():
            start = time.perf_counter()
            try:
                self._snapshot(process)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning("Thread monitor could not gather stats: %s", exc)
            elapsed = time.perf_counter() - start
            wait_time = max(self.interval_seconds - elapsed, 0.5)
            self._stop_event.wait(wait_time)

    def _snapshot(self, process) -> None:
        threads = process.threads()
        thread_count = len(threads)
        id_to_name = {thread.ident: thread.name for thread in threading.enumerate()}

        sorted_threads = sorted(
            threads,
            key=lambda info: info.user_time + info.system_time,
            reverse=True,
        )[:5]

        snapshot = [
            {
                "id": info.id,
                "name": id_to_name.get(info.id, "unknown"),
                "cpu_ms": round((info.user_time + info.system_time) * 1000, 2),
            }
            for info in sorted_threads
        ]

        self.logger.info(
            "Thread snapshot: total=%d, top=%s",
            thread_count,
            snapshot,
        )

        if self.run_dir:
            report_path = self.run_dir / "thread_monitor.latest.json"
            try:
                report_path.write_text(json.dumps(snapshot), encoding="utf-8")
            except Exception:  # pragma: no cover - diagnostics best-effort
                pass


__all__ = ["ThreadMonitor"]
