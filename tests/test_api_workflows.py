"""
Tests for workflow API endpoints.

This module tests:
- Workflow upload and storage
- Workflow listing and retrieval
- Workflow execution
- Workflow deletion
- Safety policy management
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from retrofitkit.api.workflows import router, _workflows


@pytest.fixture
def app():
    """Create test app with workflows router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    # Clear workflows before each test
    _workflows.clear()
    return TestClient(app)


@pytest.fixture
def sample_workflow_yaml():
    """Sample workflow YAML for testing."""
    return """
id: "test_workflow"
name: "Test Workflow"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.1
    children: []
"""


class TestListWorkflows:
    """Test cases for GET /workflows/."""

    def test_list_workflows_empty(self, client):
        """Test listing workflows when none exist."""
        response = client.get("/workflows/")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_workflows_with_uploaded(self, client, sample_workflow_yaml):
        """Test listing workflows after upload."""
        # Upload workflow
        client.post("/workflows/", json={"yaml_content": sample_workflow_yaml})

        # List workflows
        response = client.get("/workflows/")
        assert response.status_code == 200
        workflows = response.json()
        assert len(workflows) == 1
        assert workflows[0]["id"] == "test_workflow"
        assert workflows[0]["name"] == "Test Workflow"


class TestUploadWorkflow:
    """Test cases for POST /workflows/."""

    def test_upload_valid_workflow(self, client, sample_workflow_yaml):
        """Test uploading a valid workflow."""
        response = client.post(
            "/workflows/",
            json={"yaml_content": sample_workflow_yaml}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "test_workflow"
        assert "uploaded successfully" in data["message"]

    def test_upload_invalid_yaml(self, client):
        """Test uploading invalid YAML."""
        response = client.post(
            "/workflows/",
            json={"yaml_content": "invalid: yaml: content: ["}
        )

        assert response.status_code == 400
        assert "Invalid workflow YAML" in response.json()["detail"]

    def test_upload_workflow_missing_required_fields(self, client):
        """Test uploading workflow missing required fields."""
        incomplete_yaml = """
id: "incomplete"
name: "Incomplete Workflow"
"""
        response = client.post(
            "/workflows/",
            json={"yaml_content": incomplete_yaml}
        )

        assert response.status_code == 400


class TestGetWorkflow:
    """Test cases for GET /workflows/{id}."""

    def test_get_existing_workflow(self, client, sample_workflow_yaml):
        """Test getting details for existing workflow."""
        # Upload workflow
        client.post("/workflows/", json={"yaml_content": sample_workflow_yaml})

        # Get workflow
        response = client.get("/workflows/test_workflow")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_workflow"
        assert data["name"] == "Test Workflow"
        assert "steps" in data
        assert "wait" in data["steps"]

    def test_get_nonexistent_workflow_returns_404(self, client):
        """Test that getting nonexistent workflow returns 404."""
        response = client.get("/workflows/nonexistent")
        assert response.status_code == 404


class TestExecuteWorkflow:
    """Test cases for POST /workflows/{id}/execute."""

    def test_execute_simple_workflow(self, client, sample_workflow_yaml):
        """Test executing a simple workflow."""
        # Upload workflow
        client.post("/workflows/", json={"yaml_content": sample_workflow_yaml})

        # Execute workflow
        response = client.post(
            "/workflows/test_workflow/execute",
            json={}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["workflow_id"] == "test_workflow"
        assert result["success"] is True
        assert "wait" in result["steps_executed"]
        assert result["duration_seconds"] >= 0.1

    def test_execute_with_context(self, client, sample_workflow_yaml):
        """Test executing workflow with initial context."""
        client.post("/workflows/", json={"yaml_content": sample_workflow_yaml})

        response = client.post(
            "/workflows/test_workflow/execute",
            json={"initial_value": 42}
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_execute_nonexistent_workflow_returns_404(self, client):
        """Test executing nonexistent workflow returns 404."""
        response = client.post(
            "/workflows/nonexistent/execute",
            json={}
        )

        assert response.status_code == 404


class TestDeleteWorkflow:
    """Test cases for DELETE /workflows/{id}."""

    def test_delete_existing_workflow(self, client, sample_workflow_yaml):
        """Test deleting an existing workflow."""
        # Upload workflow
        client.post("/workflows/", json={"yaml_content": sample_workflow_yaml})

        # Delete workflow
        response = client.delete("/workflows/test_workflow")
        assert response.status_code == 200
        assert "deleted" in response.json()["message"]

        # Verify it's gone
        response = client.get("/workflows/test_workflow")
        assert response.status_code == 404

    def test_delete_nonexistent_workflow_returns_404(self, client):
        """Test deleting nonexistent workflow returns 404."""
        response = client.delete("/workflows/nonexistent")
        assert response.status_code == 404


class TestSafetyPolicies:
    """Test cases for safety policy management."""

    def test_list_safety_policies(self, client):
        """Test listing active safety policies."""
        response = client.get("/workflows/safety/policies")
        assert response.status_code == 200
        data = response.json()
        assert "policies" in data
        assert isinstance(data["policies"], list)

    def test_disable_safety_policy(self, client):
        """Test disabling a safety policy."""
        # Get initial policies
        response = client.get("/workflows/safety/policies")
        initial_policies = response.json()["policies"]

        if len(initial_policies) > 0:
            policy_name = initial_policies[0]

            # Disable policy
            response = client.post(f"/workflows/safety/policies/{policy_name}/disable")
            assert response.status_code == 200

            # Verify it's removed
            response = client.get("/workflows/safety/policies")
            remaining_policies = response.json()["policies"]
            assert policy_name not in remaining_policies
