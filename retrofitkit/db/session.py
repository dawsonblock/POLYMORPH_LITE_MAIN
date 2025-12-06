"""
Database session management and configuration for POLYMORPH-LITE.

Provides:
- Settings class for environment configuration
- SQLAlchemy AsyncEngine and AsyncSession factory
- FastAPI dependency for asynchronous database sessions
"""

import os
import logging
from typing import AsyncGenerator
from functools import lru_cache
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # Environment
        self.polymorph_env = os.getenv("POLYMORPH_ENV", "development")

        # Database
        self.raw_database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://polymorph:polymorph_pass@localhost:5432/polymorph_db"
        )
        
        # Ensure driver is async
        self.database_url = self._fix_driver(self.raw_database_url)
        
        self.echo = os.getenv("P4_DATABASE_ECHO", "False").lower() == "true"
        self.pool_size = int(os.getenv("P4_DATABASE_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("P4_DATABASE_MAX_OVERFLOW", "10"))

        # Fallback to SQLite for local development
        if "postgresql" not in self.database_url and "sqlite" not in self.database_url:
             # Default fallback if URL is malformed or empty
            db_dir = os.getenv("P4_DATA_DIR", "data")
            os.makedirs(db_dir, exist_ok=True)
            self.database_url = f"sqlite+aiosqlite:///{os.path.join(db_dir, 'polymorph.db')}"

        # Security
        self.secret_key = os.getenv("SECRET_KEY")
        if not self.secret_key or "change-in-production" in self.secret_key:
             if self.polymorph_env == "production":
                 logger.critical("PRODUCTION SECURITY RISK: WEAK SECRET KEY DETECTED")
                 # In strict mode we might raise here, but for now just warn loudly
             if not self.secret_key:
                 self.secret_key = "dev-secret-key-change-in-production"

        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-change-in-production")

        # Redis
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")

        # AI Service
        self.ai_service_url = os.getenv("AI_SERVICE_URL", "http://localhost:3000")
        self.pmm_service_url = os.getenv("PMM_SERVICE_URL", "http://localhost:3000")

    def _fix_driver(self, url: str) -> str:
        """Ensure the URL uses an async driver."""
        if url.startswith("postgresql://"):
             return url.replace("postgresql://", "postgresql+asyncpg://")
        if url.startswith("postgresql+psycopg2://"):
             return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
        if url.startswith("sqlite://"):
             return url.replace("sqlite://", "sqlite+aiosqlite://")
        return url

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create SQLAlchemy engine
settings = get_settings()

connect_args = {}
if "sqlite" in settings.database_url:
    connect_args["check_same_thread"] = False

# Create Async Engine
# Note: creating engine is synchronous, connecting is async
if settings.database_url == "sqlite+aiosqlite:///:memory:":
    pool_class = NullPool
else:
    pool_class = None # Use default

engine = create_async_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=settings.echo,
    pool_size=settings.pool_size if not "sqlite" in settings.database_url else None,
    max_overflow=settings.max_overflow if not "sqlite" in settings.database_url else None,
    future=True
)

from sqlalchemy import event

# Enable foreign keys for SQLite
if "sqlite" in settings.database_url:
    # For aiosqlite we need to listen on the sync driver connection
    # This is tricky with async, usually handled by strictly running SQL on connect
    pass 

# Create Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async database session.
    
    Usage:
        @app.get("/items")
        async def read_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


from contextlib import asynccontextmanager
from retrofitkit.core.error_codes import ErrorCode

@asynccontextmanager
async def safe_db_commit(db: AsyncSession):
    """
    Async context manager for atomic DB transactions.
    
    Usage:
        async with safe_db_commit(db):
            db.add(new_item)
            # commit is automatic
    """
    try:
        yield
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"DB Transaction Failed: {e}", extra={"error_code": ErrorCode.DB_INTEGRITY_ERROR})
        raise e
