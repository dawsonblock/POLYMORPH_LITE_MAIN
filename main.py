#!/usr/bin/env python3
"""
POLYMORPH-LITE Main Entry Point

This is the main application entry point that initializes and starts the FastAPI server.
It handles:
- Environment configuration
- Database initialization
- Logging setup
- Application startup
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from retrofitkit.core.logging_config import setup_logging, get_logger


def main():
    """Main entry point for the application."""

    # Setup logging first
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "console")
    enable_sentry = os.getenv("ENABLE_SENTRY", "false").lower() == "true"
    sentry_dsn = os.getenv("SENTRY_DSN")

    setup_logging(
        log_level=log_level,
        log_format=log_format,
        enable_sentry=enable_sentry,
        sentry_dsn=sentry_dsn
    )

    log = get_logger("main")

    # Get configuration
    environment = os.getenv("POLYMORPH_ENV", os.getenv("ENVIRONMENT", "development"))
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))

    log.info(
        "starting_polymorph",
        environment=environment,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level
    )

    # Database migration check
    if os.getenv("RUN_MIGRATIONS", "false").lower() == "true":
        log.info("running_database_migrations")
        try:
            from alembic.config import Config
            from alembic import command

            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")
            log.info("database_migrations_completed")
        except Exception as e:
            log.error("database_migration_failed", error=str(e))
            if environment == "production":
                sys.exit(1)

    # Start server
    try:
        uvicorn.run(
            "retrofitkit.api.server:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,  # reload mode doesn't support multiple workers
            log_level=log_level.lower(),
            access_log=True,
            use_colors=True if log_format == "console" else False,
        )
    except KeyboardInterrupt:
        log.info("shutting_down", reason="keyboard_interrupt")
    except Exception as e:
        log.error("startup_failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
