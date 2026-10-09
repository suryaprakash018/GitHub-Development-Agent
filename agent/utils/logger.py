"""Structured logging setup for console output and local log persistence."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "dev_agent",
    log_level: str = "INFO",
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configures and returns a structured logger with console and optional file handlers."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if directory specified)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / "agent.log"
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
