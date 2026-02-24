"""
Structured logging configuration for SymptomPal.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the application."""
    logger = logging.getLogger("symptompal")

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the application logger (singleton)."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def log_request(endpoint: str, user_id: str, **extra) -> None:
    """Log an incoming API request."""
    logger = get_logger()
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.info(f"REQUEST | endpoint={endpoint} user_id={user_id} {extra_str}".strip())


def log_response(endpoint: str, user_id: str, status: str = "success", **extra) -> None:
    """Log an API response."""
    logger = get_logger()
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.info(f"RESPONSE | endpoint={endpoint} user_id={user_id} status={status} {extra_str}".strip())


def log_error(endpoint: str, error: str, user_id: str = "unknown", **extra) -> None:
    """Log an error."""
    logger = get_logger()
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.error(f"ERROR | endpoint={endpoint} user_id={user_id} error={error} {extra_str}".strip())


def log_warning(message: str, **extra) -> None:
    """Log a warning."""
    logger = get_logger()
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.warning(f"WARNING | {message} {extra_str}".strip())


def log_debug(message: str, **extra) -> None:
    """Log a debug message."""
    logger = get_logger()
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.debug(f"DEBUG | {message} {extra_str}".strip())
