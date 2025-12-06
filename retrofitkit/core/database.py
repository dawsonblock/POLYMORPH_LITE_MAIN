
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
import logging

from retrofitkit.db.session import engine, AsyncSessionLocal, get_db
from retrofitkit.db.base import Base

# Re-export for compatibility
get_db_session = get_db

logger = logging.getLogger(__name__)

async def init_db():
    """Initialize database (create tables if needed)."""
    # Note: In production, use Alembic. This is for quick local setup.
    async with engine.begin() as conn:
        # Import all models so they are registered in metadata
        # import retrofitkit.db.models  # Ensure models are loaded
        # Note: If models are scattered, we might need to import them here.
        # However, Base.metadata should be populated if modules are imported.
        # For now, we assume the caller app has imported models or they are in db/models/__init__.py
        
        # We'll explicitly import the main models module to be safe
        try:
             import retrofitkit.db.models
        except ImportError:
             pass

        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized (metadata.create_all completed)")

