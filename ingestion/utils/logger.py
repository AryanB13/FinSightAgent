"""
logger.py — Centralised logging configuration for the ingestion pipeline.

Usage in every module:
    from ingestion.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

_LOG_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "logs")
_LOG_FILE: str = os.path.join(_LOG_DIR, "ingestion.log")

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level start time used by log_progress() to compute elapsed/ETA.
_PIPELINE_START_TIME: float = time.time()


def _ensure_log_dir() -> None:
    """Creates the logs/ directory if it does not exist."""
    os.makedirs(_LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a Logger for *name* configured with:
    - StreamHandler (stdout) at INFO level — progress messages for normal use.
    - FileHandler  (logs/ingestion.log) at DEBUG level — full trace for debugging.

    Idempotent: calling get_logger() with the same name twice returns the same
    Logger instance without adding duplicate handlers.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger already exists.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler (INFO) ────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(console_handler)

    # ── File handler (DEBUG) ──────────────────────────────────────────────────
    try:
        _ensure_log_dir()
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(file_handler)
    except OSError as exc:
        # If we cannot write to the log file (e.g. read-only filesystem),
        # degrade gracefully to console-only logging rather than crashing.
        logger.warning("Could not open log file %s: %s — logging to console only.", _LOG_FILE, exc)

    # Prevent propagation to the root logger to avoid duplicate output
    # when the calling code also configures basicConfig at the top level.
    logger.propagate = False

    return logger


def reset_pipeline_timer() -> None:
    """
    Resets the global start time used by log_progress().
    Call this at the beginning of a full pipeline run so elapsed/ETA
    calculations are relative to the current run, not process startup.
    """
    global _PIPELINE_START_TIME
    _PIPELINE_START_TIME = time.time()


def log_progress(
    logger: logging.Logger,
    step: str,
    current: int,
    total: int,
) -> None:
    """
    Logs a standardised progress message at INFO level:
        "[step] 3/9 (33.3%) — elapsed: 12.4s, ETA: 24.8s"

    Args:
        logger:  Logger returned by get_logger().
        step:    Label for the current stage, e.g. "Ingesting PDF".
        current: Current item index (1-based).
        total:   Total number of items.
    """
    elapsed = time.time() - _PIPELINE_START_TIME
    pct = (current / total * 100) if total > 0 else 0.0
    eta = (elapsed / current * (total - current)) if current > 0 else 0.0
    logger.info(
        "[%s] %d/%d (%.1f%%) — elapsed: %.1fs, ETA: %.1fs",
        step,
        current,
        total,
        pct,
        elapsed,
        eta,
    )
