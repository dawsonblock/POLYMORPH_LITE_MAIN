import pytest
import asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient, ASGITransport

from retrofitkit.config import settings
# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
settings.ENV = "testing"

from retrofitkit.core.database import get_db_session
from retrofitkit.db.base import Base
from retrofitkit.db.models.user import User
import retrofitkit.db.models  # Register all models
from retrofitkit.api.server import app

from sqlalchemy.pool import StaticPool

engine = create_async_engine(
    TEST_DATABASE_URL, 
    echo=False, 
    future=True,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async with TestingSessionLocal() as session:
        yield session
        await session.rollback() # Rollback transaction
        
    # Drop tables to ensure clean state for next test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with overridden DB dependency."""
    
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()

@pytest.fixture
def mock_user():
    return {"email": "test@example.com", "role": "admin"}