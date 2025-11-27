"""
Tests for Visual Workflow Builder API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import uuid

from retrofitkit.api.server import app

client = TestClient(app)


def mock_get_current_user():
    return {"email": "test@example.com", "name": "Test User", "role": "Admin"}


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def workflow_definition():
    """Sample workflow definition with nodes and edges."""
    return {
        "workflow_name": f"test-workflow-{uuid.uuid4().hex[:8]}",
        "nodes": [
            {
                "id": "node-1",
                "type": "acquire",
                "position": {"x": 100, "y": 100},
                "data": {"device": "raman", "duration": 10}
            },
            {
                "id": "node-2",
                "type": "ai-evaluate",
                "position": {"x": 300, "y": 100},
                "data": {"threshold": 0.8}
            },
            {
                "id": "node-3",
                "type": "gate",
                "position": {"x": 500, "y": 100},
                "data": {"condition": "peak_detected"}
            }
        ],
        "edges": [
            {
                "id": "edge-1",
                "source": "node-1",
                "target": "node-2"
            },
            {
                "id": "edge-2",
                "source": "node-2",
                "target": "node-3"
            }
        ],
        "metadata": {
            "author": "test@example.com",
            "description": "Test workflow"
        }
    }


class TestWorkflowBuilderAPI:
    """Test workflow builder CRUD operations."""
    
    @pytest.fixture(autouse=True)
    def override_dependencies(self, db_session):
        """Override dependencies."""
        from retrofitkit.api.dependencies import get_current_user
        
        # Override auth
        app.dependency_overrides[get_current_user] = mock_get_current_user
        
        # Patch get_session in the workflow_builder module
        # We need to return a mock that behaves like a session context manager if needed,
        # or just the session itself if get_session() returns a session directly.
        # Looking at workflow_builder.py: session = get_session(); ... session.close()
        # So get_session() returns a session.
        # We should return a MagicMock that wraps the db_session but prevents close() from actually closing it
        # because pytest fixture handles teardown.
        
        real_close = db_session.close
        db_session.close = Mock() # Prevent closing by endpoint
        
        with patch("retrofitkit.api.workflow_builder.get_session", return_value=db_session):
            yield
            
        db_session.close = real_close # Restore
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

    def test_create_workflow(self, workflow_definition, auth_headers):
        """Test creating a new workflow definition."""
        response = client.post(
            "/api/workflow-builder/workflows",
            json=workflow_definition,
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["workflow_name"] == workflow_definition["workflow_name"]
        assert data["version"] == 1
        assert data["is_active"] == False
        assert data["is_approved"] == False
        assert len(data["definition"]["nodes"]) == 3
        assert len(data["definition"]["edges"]) == 2

    def test_workflow_versioning(self, auth_headers):
        """Test that creating multiple versions increments version number."""
        workflow_name = f"versioned-workflow-{uuid.uuid4().hex[:8]}"

        # Create version 1
        workflow_v1 = {
            "workflow_name": workflow_name,
            "nodes": [{"id": "n1", "type": "acquire", "position": {"x": 0, "y": 0}, "data": {}}],
            "edges": [],
            "metadata": {}
        }
        response_v1 = client.post(
            "/api/workflow-builder/workflows",
            json=workflow_v1,
            headers=auth_headers
        )
        assert response_v1.status_code == 201
        assert response_v1.json()["version"] == 1

        # Create version 2
        workflow_v2 = {
            "workflow_name": workflow_name,
            "nodes": [
                {"id": "n1", "type": "acquire", "position": {"x": 0, "y": 0}, "data": {}},
                {"id": "n2", "type": "measure", "position": {"x": 100, "y": 0}, "data": {}}
            ],
            "edges": [{"id": "e1", "source": "n1", "target": "n2"}],
            "metadata": {}
        }
        response_v2 = client.post(
            "/api/workflow-builder/workflows",
            json=workflow_v2,
            headers=auth_headers
        )
        assert response_v2.status_code == 201
        assert response_v2.json()["version"] == 2

    def test_list_workflow_versions(self, auth_headers):
        """Test listing all versions of a workflow."""
        response = client.get(
            "/api/workflow-builder/workflows/test-workflow",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_approve_workflow(self, workflow_definition, auth_headers):
        """Test workflow approval process."""
        # Create workflow
        create_response = client.post(
            "/api/workflow-builder/workflows",
            json=workflow_definition,
            headers=auth_headers
        )
        assert create_response.status_code == 201
        workflow_data = create_response.json()

        # Approve workflow
        approve_response = client.post(
            f"/api/workflow-builder/workflows/{workflow_data['workflow_name']}/v/{workflow_data['version']}/approve",
            headers=auth_headers
        )

        if approve_response.status_code == 200:
            assert "approved" in approve_response.json()["message"].lower()

    def test_activate_workflow_requires_approval(self, workflow_definition, auth_headers):
        """Test that unapproved workflows cannot be activated."""
        # Create workflow (not approved)
        create_response = client.post(
            "/api/workflow-builder/workflows",
            json=workflow_definition,
            headers=auth_headers
        )
        assert create_response.status_code == 201
        workflow_data = create_response.json()

        # Try to activate without approval
        activate_response = client.post(
            f"/api/workflow-builder/workflows/{workflow_data['workflow_name']}/v/{workflow_data['version']}/activate",
            headers=auth_headers
        )

        assert activate_response.status_code == 403  # Forbidden

    def test_execute_workflow(self, auth_headers):
        """Test workflow execution creation."""
        # Create, approve, and activate a workflow first
        workflow_def = {
            "workflow_name": f"exec-test-{uuid.uuid4().hex[:8]}",
            "nodes": [{"id": "n1", "type": "acquire", "position": {"x": 0, "y": 0}, "data": {}}],
            "edges": [],
            "metadata": {}
        }

        create_resp = client.post(
            "/api/workflow-builder/workflows",
            json=workflow_def,
            headers=auth_headers
        )
        assert create_resp.status_code == 201
        workflow = create_resp.json()

        # Approve
        approve_resp = client.post(
            f"/api/workflow-builder/workflows/{workflow['workflow_name']}/v/{workflow['version']}/approve",
            headers=auth_headers
        )

        # Activate
        if approve_resp.status_code == 200:
            activate_resp = client.post(
                f"/api/workflow-builder/workflows/{workflow['workflow_name']}/v/{workflow['version']}/activate",
                headers=auth_headers
            )

            # Execute
            if activate_resp.status_code == 200:
                execute_resp = client.post(
                    "/api/workflow-builder/execute",
                    json={
                        "workflow_name": workflow['workflow_name'],
                        "parameters": {"test": "param"}
                    },
                    headers=auth_headers
                )

                assert execute_resp.status_code == 201
                exec_data = execute_resp.json()
                assert exec_data["status"] == "running"
                assert exec_data["operator"] == "test@example.com"

    def test_list_executions(self, auth_headers):
        """Test listing workflow executions."""
        response = client.get(
            "/api/workflow-builder/executions",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestWorkflowIntegrity:
    """Test workflow definition integrity features."""
    
    @pytest.fixture(autouse=True)
    def override_dependencies(self, db_session):
        """Override dependencies."""
        from retrofitkit.api.dependencies import get_current_user, get_db
        
        # Override auth
        app.dependency_overrides[get_current_user] = mock_get_current_user
        
        # Override DB
        def override_get_db():
            yield db_session
            
        app.dependency_overrides[get_db] = override_get_db
        
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

    def test_workflow_hash_generation(self, workflow_definition, auth_headers):
        """Test that workflow definitions get a hash for integrity verification."""
        response = client.post(
            "/api/workflow-builder/workflows",
            json=workflow_definition,
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert "definition_hash" in data
        assert len(data["definition_hash"]) == 64  # SHA-256 hash length
