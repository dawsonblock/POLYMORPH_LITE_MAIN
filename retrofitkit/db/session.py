"""
Database session management and configuration for POLYMORPH-LITE.

Provides:
- Settings class for environment configuration
- SQLAlchemy engine and session factory
- FastAPI dependency for database sessions
"""

import os
from typing import Generator
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # Environment
        self.polymorph_env = os.getenv("POLYMORPH_ENV", "development")

        # Database
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://polymorph:polymorph_pass@localhost:5432/polymorph_db"
        )
        self.echo = os.getenv("P4_DATABASE_ECHO", "False").lower() == "true"
        self.pool_size = int(os.getenv("P4_DATABASE_POOL_SIZE", "5"))
        self.max_overflow = int(os.getenv("P4_DATABASE_MAX_OVERFLOW", "10"))

        # Fallback to SQLite for local development if not PostgreSQL
        if "postgresql" not in self.database_url and "sqlite" not in self.database_url:
            db_dir = os.getenv("P4_DATA_DIR", "data")
            os.makedirs(db_dir, exist_ok=True)
            self.database_url = f"sqlite:///{os.path.join(db_dir, 'polymorph.db')}"

        # Security
        self.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-change-in-production")

        # Redis
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")

        # AI Service
        self.ai_service_url = os.getenv("AI_SERVICE_URL", "http://localhost:3000")
        self.pmm_service_url = os.getenv("PMM_SERVICE_URL", "http://localhost:3000")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create SQLAlchemy engine
settings = get_settings()

connect_args = {}
if "sqlite" in settings.database_url:
    connect_args["check_same_thread"] = False

if settings.database_url == "sqlite:///:memory:":
    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        poolclass=StaticPool,
        echo=settings.echo
    )
else:
    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        pool_pre_ping=True,  # Verify connections before using
        echo=settings.echo,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow
    )

from sqlalchemy import event

# Enable foreign keys for SQLite
if "sqlite" in settings.database_url:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# Create SessionLocal class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.
    
    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from contextlib import contextmanager
from retrofitkit.core.error_codes import ErrorCode
import logging

logger = logging.getLogger(__name__)

@contextmanager
def safe_db_commit(db: Session):
    """
    Context manager for atomic DB transactions.
    
    Usage:
        with safe_db_commit(db):
            db.add(new_item)
            # commit is automatic
            
    On error:
        - Rolls back transaction
        - Logs error
        - Re-raises exception
    """
    try:
        yield
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB Transaction Failed: {e}", extra={"error_code": ErrorCode.DB_INTEGRITY_ERROR})
        raise e
