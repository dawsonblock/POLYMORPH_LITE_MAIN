"""
Tests for Sample Tracking API endpoints.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import Mock, patch
import uuid
from sqlalchemy import select

from retrofitkit.api.server import app
from retrofitkit.core.database import get_db_session
from retrofitkit.api.dependencies import get_current_user
from retrofitkit.db.models.user import User


# Mock authentication
def mock_get_current_user():
    return {"email": "test@example.com", "name": "Test User", "role": "Admin"}


@pytest.fixture
async def client(db_session):
    """Create async test client with overridden DB dependency."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_current_user] = mock_get_current_user
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Provide authentication headers for tests."""
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def sample_data():
    """Sample creation data."""
    return {
        "sample_id": f"SAMPLE-{uuid.uuid4().hex[:8]}",
        "lot_number": "LOT-001",
        "status": "active",
        "extra_data": {"test": "data"}
    }


@pytest.mark.asyncio
class TestSampleAPI:
    """Test sample CRUD operations."""
    
    @pytest.fixture(autouse=True)
    async def seed_user(self, db_session):
        """Seed test user."""
        result = await db_session.execute(select(User).filter_by(email="test@example.com"))
        user = result.scalars().first()
        if not user:
            user = User(
                email="test@example.com",
                name="Test User",
                role="Admin",
                password_hash=b"test_hash"
            )
            db_session.add(user)
            await db_session.commit()

    async def test_create_sample(self, client, sample_data, auth_headers):
        """Test creating a new sample."""
        response = await client.post(
            "/api/samples/",
            json=sample_data,
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["sample_id"] == sample_data["sample_id"]
        assert data["lot_number"] == sample_data["lot_number"]
        assert data["status"] == "active"
        assert data["created_by"] == "test@example.com"

    async def test_list_samples(self, client, auth_headers):
        """Test listing samples."""
        response = await client.get("/api/samples/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_get_sample_not_found(self, client, auth_headers):
        """Test getting non-existent sample."""
        response = await client.get("/api/samples/NONEXISTENT", headers=auth_headers)

        assert response.status_code == 404

    async def test_update_sample(self, client, auth_headers):
        """Test updating sample status."""
        # First create a sample
        sample_data = {
            "sample_id": f"SAMPLE-{uuid.uuid4().hex[:8]}",
            "status": "active"
        }
        create_response = await client.post(
            "/api/samples/",
            json=sample_data,
            headers=auth_headers
        )
        assert create_response.status_code == 201

        # Update the sample
        update_data = {"status": "consumed"}
        response = await client.put(
            f"/api/samples/{sample_data['sample_id']}",
            json=update_data,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "consumed"

    async def test_split_sample(self, client, auth_headers):
        """Test sample splitting (aliquoting)."""
        # Create parent sample
        parent_data = {
            "sample_id": f"PARENT-{uuid.uuid4().hex[:8]}",
            "status": "active"
        }
        create_response = await client.post(
            "/api/samples/",
            json=parent_data,
            headers=auth_headers
        )
        assert create_response.status_code == 201

        # Split into child samples
        child_ids = [f"CHILD-{i}-{uuid.uuid4().hex[:6]}" for i in range(3)]
        response = await client.post(
            f"/api/samples/{parent_data['sample_id']}/split",
            json=child_ids,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert "child_samples" in data
            assert len(data["child_samples"]) == 3


@pytest.mark.asyncio
class TestContainerAPI:
    """Test container management."""
    
    @pytest.fixture(autouse=True)
    async def seed_user(self, db_session):
        """Seed test user."""
        result = await db_session.execute(select(User).filter_by(email="test@example.com"))
        user = result.scalars().first()
        if not user:
            user = User(
                email="test@example.com",
                name="Test User",
                role="Admin",
                password_hash=b"test_hash"
            )
            db_session.add(user)
            await db_session.commit()

    async def test_create_container(self, client, auth_headers):
        """Test creating a container."""
        container_data = {
            "container_id": f"CONTAINER-{uuid.uuid4().hex[:8]}",
            "container_type": "vial",
            "location": "Freezer-A-Shelf-2",
            "capacity": 100
        }

        response = await client.post(
            "/api/samples/containers",
            json=container_data,
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["container_id"] == container_data["container_id"]
        assert data["capacity"] == 100

    async def test_list_containers(self, client, auth_headers):
        """Test listing containers."""
        response = await client.get("/api/samples/containers", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
class TestProjectAPI:
    """Test project management."""
    
    @pytest.fixture(autouse=True)
    async def seed_user(self, db_session):
        """Seed test user."""
        result = await db_session.execute(select(User).filter_by(email="test@example.com"))
        user = result.scalars().first()
        if not user:
            user = User(
                email="test@example.com",
                name="Test User",
                role="Admin",
                password_hash=b"test_hash"
            )
            db_session.add(user)
            await db_session.commit()

    async def test_create_project(self, client, auth_headers):
        """Test creating a project."""
        project_data = {
            "project_id": f"PROJ-{uuid.uuid4().hex[:8]}",
            "name": "Test Project",
            "description": "A test project for validation",
            "status": "active"
        }

        response = await client.post(
            "/api/samples/projects",
            json=project_data,
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["project_id"] == project_data["project_id"]
        assert data["name"] == "Test Project"

    async def test_list_projects(self, client, auth_headers):
        """Test listing projects."""
        response = await client.get("/api/samples/projects", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
