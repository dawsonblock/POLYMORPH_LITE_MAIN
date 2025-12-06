"""
Tests for API Hardening (Audit Log & Roles).
"""
import pytest
import asyncio
from fastapi import FastAPI, Request, Depends
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from retrofitkit.api.middleware.audit_log import AuditLogMiddleware
from retrofitkit.api.auth.roles import require_role, Role
from retrofitkit.db.base import Base
import retrofitkit.db.models  # Register all models

# Test database setup
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, poolclass=StaticPool, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

# --- Setup Test App ---
def create_test_app():
    app = FastAPI()
    
    @app.middleware("http")
    async def mock_auth_middleware(request: Request, call_next):
        # Mock Auth: Set user in state based on header
        role = request.headers.get("X-Role", "operator")
        request.state.user = {"email": "test@example.com", "role": role}
        response = await call_next(request)
        return response

    @app.post("/protected")
    def protected_route(user = Depends(require_role([Role.ADMIN]))):
        return {"message": "success"}

    @app.post("/operator")
    def operator_route(user = Depends(require_role([Role.OPERATOR]))):
        return {"message": "success"}
        
    @app.post("/state-change")
    def state_change(data: dict):
        return {"status": "changed"}

    return app

@pytest.fixture
async def setup_db():
    """Initialize test database."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def async_client(setup_db):
    """Create async test client."""
    app = create_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
class TestRBAC:
    async def test_admin_access(self, async_client):
        response = await async_client.post("/protected", headers={"X-Role": "admin"})
        assert response.status_code == 200

    async def test_operator_denied_admin(self, async_client):
        response = await async_client.post("/protected", headers={"X-Role": "operator"})
        assert response.status_code == 403

    async def test_operator_access(self, async_client):
        response = await async_client.post("/operator", headers={"X-Role": "operator"})
        assert response.status_code == 200

    async def test_invalid_role(self, async_client):
        response = await async_client.post("/operator", headers={"X-Role": "hacker"})
        assert response.status_code == 403

@pytest.mark.asyncio
class TestAuditLog:
    async def test_audit_log_generation(self, async_client):
        response = await async_client.post("/state-change", json={"key": "value", "password": "secret"}, headers={"X-Role": "admin"})
        assert response.status_code == 200
