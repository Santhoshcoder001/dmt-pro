"""Logging configuration helpers for the ABC-HP project."""

from __future__ import annotations

import logging


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure a consistent logging format for CLI and API usage."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
