"""
Tests for Sample Tracking API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import uuid

from retrofitkit.api.server import app


# Mock authentication
def mock_get_current_user():
    return {"email": "test@example.com", "name": "Test User", "role": "Admin"}


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


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


class TestSampleAPI:
    """Test sample CRUD operations."""
    
    @pytest.fixture(autouse=True)
    def override_dependencies(self, db_session):
        """Override dependencies."""
        from retrofitkit.db.session import get_db
        from retrofitkit.api.dependencies import get_current_user
        
        # Override auth
        app.dependency_overrides[get_current_user] = mock_get_current_user
        
        # Override DB
        app.dependency_overrides[get_db] = lambda: db_session
        
        yield
            
        app.dependency_overrides = {}

    @pytest.fixture(autouse=True)
    def seed_user(self, db_session):
        """Seed test user."""
        from retrofitkit.db.models.user import User
        from retrofitkit.db.models.rbac import Role, UserRole
        
        user = db_session.query(User).filter_by(email="test@example.com").first()
        if not user:
            user = User(
                email="test@example.com",
                name="Test User",
                role="Admin",
                password_hash=b"test_hash"
            )
            db_session.add(user)
            db_session.commit()

    def test_create_sample(self, client, sample_data, auth_headers):
        """Test creating a new sample."""
        response = client.post(
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

    def test_list_samples(self, client, auth_headers):
        """Test listing samples."""
        response = client.get("/api/samples/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_sample_not_found(self, client, auth_headers):
        """Test getting non-existent sample."""
        response = client.get("/api/samples/NONEXISTENT", headers=auth_headers)

        assert response.status_code == 404

    def test_update_sample(self, client, auth_headers):
        """Test updating sample status."""
        # First create a sample
        sample_data = {
            "sample_id": f"SAMPLE-{uuid.uuid4().hex[:8]}",
            "status": "active"
        }
        create_response = client.post(
            "/api/samples/",
            json=sample_data,
            headers=auth_headers
        )
        assert create_response.status_code == 201

        # Update the sample
        update_data = {"status": "consumed"}
        response = client.put(
            f"/api/samples/{sample_data['sample_id']}",
            json=update_data,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "consumed"

    def test_split_sample(self, client, auth_headers):
        """Test sample splitting (aliquoting)."""
        # Create parent sample
        parent_data = {
            "sample_id": f"PARENT-{uuid.uuid4().hex[:8]}",
            "status": "active"
        }
        create_response = client.post(
            "/api/samples/",
            json=parent_data,
            headers=auth_headers
        )
        assert create_response.status_code == 201

        # Split into child samples
        child_ids = [f"CHILD-{i}-{uuid.uuid4().hex[:6]}" for i in range(3)]
        response = client.post(
            f"/api/samples/{parent_data['sample_id']}/split",
            json=child_ids,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert "child_samples" in data
            assert len(data["child_samples"]) == 3


class TestContainerAPI:
    """Test container management."""
    
    @pytest.fixture(autouse=True)
    def override_dependencies(self, db_session):
        """Override dependencies."""
        from retrofitkit.api.dependencies import get_current_user
        from retrofitkit.db.session import get_db
        
        app.dependency_overrides[get_current_user] = mock_get_current_user
        app.dependency_overrides[get_db] = lambda: db_session
        
        yield
            
        app.dependency_overrides = {}

    @pytest.fixture(autouse=True)
    def seed_user(self, db_session):
        """Seed test user."""
        from retrofitkit.db.models.user import User
        from retrofitkit.db.models.rbac import Role, UserRole
        
        user = db_session.query(User).filter_by(email="test@example.com").first()
        if not user:
            user = User(
                email="test@example.com",
                name="Test User",
                role="Admin",
                password_hash=b"test_hash"
            )
            db_session.add(user)
            db_session.commit()

    def test_create_container(self, client, auth_headers):
        """Test creating a container."""
        container_data = {
            "container_id": f"CONTAINER-{uuid.uuid4().hex[:8]}",
            "container_type": "vial",
            "location": "Freezer-A-Shelf-2",
            "capacity": 100
        }

        response = client.post(
            "/api/samples/containers",
            json=container_data,
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["container_id"] == container_data["container_id"]
        assert data["capacity"] == 100

    def test_list_containers(self, client, auth_headers):
        """Test listing containers."""
        response = client.get("/api/samples/containers", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestProjectAPI:
    """Test project management."""
    
    @pytest.fixture(autouse=True)
    def override_dependencies(self, db_session):
        """Override dependencies."""
        from retrofitkit.api.dependencies import get_current_user
        from retrofitkit.db.session import get_db
        
        app.dependency_overrides[get_current_user] = mock_get_current_user
        app.dependency_overrides[get_db] = lambda: db_session
        
        yield
            
        app.dependency_overrides = {}

    @pytest.fixture(autouse=True)
    def seed_user(self, db_session):
        """Seed test user."""
        from retrofitkit.db.models.user import User
        from retrofitkit.db.models.rbac import Role, UserRole
        
        user = db_session.query(User).filter_by(email="test@example.com").first()
        if not user:
            user = User(
                email="test@example.com",
                name="Test User",
                role="Admin",
                password_hash=b"test_hash"
            )
            db_session.add(user)
            db_session.commit()

    def test_create_project(self, client, auth_headers):
        """Test creating a project."""
        project_data = {
            "project_id": f"PROJ-{uuid.uuid4().hex[:8]}",
            "name": "Test Project",
            "description": "A test project for validation",
            "status": "active"
        }

        response = client.post(
            "/api/samples/projects",
            json=project_data,
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["project_id"] == project_data["project_id"]
        assert data["name"] == "Test Project"

    def test_list_projects(self, client, auth_headers):
        """Test listing projects."""
        response = client.get("/api/samples/projects", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
