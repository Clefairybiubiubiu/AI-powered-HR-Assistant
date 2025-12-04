"""
Lightweight logger helper for utility modules.
"""
from __future__ import annotations

import logging
from typing import Optional

_LOGGER_CONFIGURED = False


def _configure_logger(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    format_string = fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=format_string)
    _LOGGER_CONFIGURED = True


def get_logger(name: str = "resume_matcher") -> logging.Logger:
    """
    Return a module-specific logger. Configuration happens lazily so that
    importing this helper never overrides the main logging setup.
    """
    _configure_logger()
    return logging.getLogger(name)


