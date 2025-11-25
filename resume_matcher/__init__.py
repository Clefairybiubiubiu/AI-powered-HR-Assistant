"""Resume-JD Matcher package."""
from .config import AppConfig, config
from .logging_config import get_logger, setup_logging

__version__ = "2.0.0"
__all__ = ['AppConfig', 'config', 'get_logger', 'setup_logging']

