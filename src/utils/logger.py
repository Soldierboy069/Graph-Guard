"""
GraphGuard: Explainable GNN for Intrusion Detection
Utility: Logging setup
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> logging.Logger:
    """
    Create and configure a named logger with console and optional file output.

    Args:
        name: Logger name (typically __name__ of calling module).
        log_dir: Directory to write rotating log file. If None, file logging is skipped.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Log message format string.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        # Avoid adding duplicate handlers if called multiple times
        return logger

    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{name.replace('.', '_')}.log"
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=3
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
