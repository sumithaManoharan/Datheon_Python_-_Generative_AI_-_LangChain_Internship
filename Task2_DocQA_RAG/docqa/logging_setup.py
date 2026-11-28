"""Logging setup shared by the CLI and the rest of the package."""

import logging
import sys

PACKAGE_LOGGER = "docqa"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Attach one stderr handler to the package logger.

    Safe to call more than once: the handler is only added the first time,
    so repeated calls never duplicate log lines. Modules should create their
    own logger with ``logging.getLogger(__name__)``; it inherits this setup.

    Args:
        level: A standard level name such as "INFO" or "DEBUG".
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    # Keep records out of the root logger so they are not printed twice.
    logger.propagate = False
