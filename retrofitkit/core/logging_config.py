"""
Centralized structured logging configuration for POLYMORPH-LITE.

Provides consistent logging across all components with support for:
- JSON and console formatting
- Multiple log levels
- Component-specific loggers
- Sentry integration (optional)
"""
import os
import sys
import logging
import structlog
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

# Create logs directory
LOGS_DIR = Path(os.getenv("P4_LOGS_DIR", "logs"))
LOGS_DIR.mkdir(exist_ok=True)


def add_timestamp(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add ISO timestamp to log entries."""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_component(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add component name from logger name."""
    event_dict["component"] = logger.name
    return event_dict


def setup_logging(
    log_level: str = None,
    log_format: str = None,
    enable_sentry: bool = False,
    sentry_dsn: str = None
):
    """
    Configure structured logging for the entire application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('json' or 'console')
        enable_sentry: Whether to enable Sentry error tracking
        sentry_dsn: Sentry DSN for error tracking
    """
    # Get configuration from environment if not provided
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_format = log_format or os.getenv("LOG_FORMAT", "console")

    # Map string levels to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    numeric_level = level_map.get(log_level.upper(), logging.INFO)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_component,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Choose renderer based on format
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure Sentry if enabled
    if enable_sentry and sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )

            sentry_sdk.init(
                dsn=sentry_dsn,
                integrations=[sentry_logging],
                traces_sample_rate=0.1,
                environment=os.getenv("ENVIRONMENT", "development")
            )

            log = get_logger("logging_config")
            log.info("sentry_initialized", dsn=sentry_dsn[:20] + "...")

        except ImportError:
            log = get_logger("logging_config")
            log.warning("sentry_not_available",
                       message="sentry_sdk not installed, skipping Sentry integration")

    log = get_logger("logging_config")
    log.info("logging_configured",
             level=log_level,
             format=log_format,
             sentry_enabled=enable_sentry)


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically module name)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def log(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# File handler for persistent logs
def add_file_handler(filename: str = None, level: str = "INFO"):
    """
    Add file handler for persistent logging.

    Args:
        filename: Log file name (default: polymorph-{date}.log)
        level: Minimum log level for file output
    """
    if filename is None:
        filename = f"polymorph-{datetime.now().strftime('%Y-%m-%d')}.log"

    log_path = LOGS_DIR / filename

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    logging.getLogger().addHandler(file_handler)

    log = get_logger("logging_config")
    log.info("file_handler_added", log_path=str(log_path), level=level)


# Initialize logging on import (can be reconfigured later)
if __name__ != "__main__":
    setup_logging()
