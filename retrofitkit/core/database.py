from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from retrofitkit.config import settings
import logging

logger = logging.getLogger(__name__)

# Create Async Engine with optimized settings
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_size=20,  # Increase connection pool size
    max_overflow=10,  # Allow 10 additional connections
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Create Async Session Factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

@asynccontextmanager
async def get_db_session():
    """Async context manager for DB sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

async def init_db():
    """Initialize database (create tables if needed)."""
    # Note: In production, use Alembic. This is for quick local setup.
    from retrofitkit.core.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
