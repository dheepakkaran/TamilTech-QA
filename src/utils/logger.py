"""Centralised loguru-based logging.

All scripts should import :func:`get_logger` instead of using ``print``.
The first call configures the global sink; subsequent calls just return
a bound logger.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "20 MB",
) -> None:
    """Configure the global loguru sink.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path for a file sink. The directory is created.
        rotation: Loguru rotation policy for the file sink.

    Returns:
        None.

    Example:
        >>> setup_logging("DEBUG", log_file="outputs/run.log")
    """
    global _CONFIGURED
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=5,
            encoding="utf-8",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} - {message}"
            ),
        )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None):
    """Return a loguru logger bound to ``name`` (or the root logger).

    Args:
        name: Optional logger name to bind. Useful for filtering.

    Returns:
        A loguru logger instance.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("hello")  # doctest: +SKIP
    """
    if not _CONFIGURED:
        setup_logging()
    return _logger.bind(name=name) if name else _logger
