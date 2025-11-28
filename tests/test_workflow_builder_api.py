import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from retrofitkit.api.server import app
from retrofitkit.db.models.workflow import WorkflowVersion
from retrofitkit.db.models.user import User
from retrofitkit.db.models.rbac import Role, UserRole

client = TestClient(app)

@pytest.fixture
def mock_db_session():
    with patch("retrofitkit.api.workflow_builder.get_session") as mock_get_session:
        session = MagicMock()
        mock_get_session.return_value = session
        yield session

from retrofitkit.api.dependencies import get_current_user

@pytest.fixture
def mock_current_user():
    user = {"email": "test@example.com"}
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    app.dependency_overrides = {}

def test_activate_workflow_insufficient_permissions(mock_db_session, mock_current_user):
    # Mock user with only "scientist" role
    mock_user_obj = MagicMock()
    mock_user_obj.roles = [MagicMock(role=MagicMock(name="scientist"))]
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_user_obj
    
    response = client.post("/api/workflow-builder/workflows/TestWorkflow/v/1/activate")
    
    assert response.status_code == 403
    assert "Insufficient permissions" in response.json()["detail"]

def test_activate_workflow_success(mock_db_session, mock_current_user):
    # Mock user with "admin" role
    mock_user_obj = MagicMock()
    role_mock = MagicMock()
    role_mock.name = "admin"
    user_role_mock = MagicMock()
    user_role_mock.role = role_mock
    mock_user_obj.roles = [user_role_mock]
    
    # Mock workflow
    mock_workflow = MagicMock()
    mock_workflow.is_approved = True
    mock_workflow.id = "123"
    
    # Setup query side effects
    def query_side_effect(model):
        query = MagicMock()
        if model == User:
            query.filter.return_value.first.return_value = mock_user_obj
        elif model == WorkflowVersion:
            query.filter.return_value.first.return_value = mock_workflow
            query.filter.return_value.update.return_value = None
        return query
    
    mock_db_session.query.side_effect = query_side_effect
    
    response = client.post("/api/workflow-builder/workflows/TestWorkflow/v/1/activate")
    
    assert response.status_code == 200
    assert "activated" in response.json()["message"]

def test_execute_workflow_unapproved_returns_403(mock_db_session, mock_current_user):

    mock_workflow = MagicMock()
    mock_workflow.is_approved = False
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_workflow

    response = client.post(
        "/api/workflow-builder/execute",
        json={
            "workflow_name": "TestWorkflow",
            "workflow_version": 1,
            "parameters": {},
        },
    )

    assert response.status_code == 403
    assert "must be approved" in response.json()["detail"].lower()


def test_execute_workflow_missing_workflow_returns_404(mock_db_session, mock_current_user):

    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = client.post(
        "/api/workflow-builder/execute",
        json={
            "workflow_name": "MissingWorkflow",
            "parameters": {},
        },
    )

    assert response.status_code == 404
    assert "workflow" in response.json()["detail"].lower()


def test_execute_workflow_happy_path_creates_execution(mock_db_session, mock_current_user):
    # Approved workflow with a simple acquire → measure graph
    graph = {
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "data": {},
            },
            {
                "id": "n-acquire",
                "type": "acquire",
                "data": {"voltage": 2.5, "device_id": "daq"},
            },
            {
                "id": "n-measure",
                "type": "measure",
                "data": {"timeout": 120, "device_id": "raman"},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "n-acquire"},
            {"id": "e2", "source": "n-acquire", "target": "n-measure"},
        ],
    }

    mock_workflow = MagicMock()
    mock_workflow.workflow_name = "TestWorkflow"
    mock_workflow.version = 1
    mock_workflow.definition = graph
    mock_workflow.is_approved = True
    mock_workflow.id = uuid.uuid4()

    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_workflow

    # Ensure created execution objects have the fields required by the
    # response model (id as UUID4 and started_at as datetime).
    def add_side_effect(obj):
        if hasattr(obj, "run_id"):
            obj.id = uuid.uuid4()
            obj.started_at = datetime.now(timezone.utc)

    mock_db_session.add.side_effect = add_side_effect

    response = client.post(
        "/api/workflow-builder/execute",
        json={
            "workflow_name": "TestWorkflow",
            "workflow_version": 1,
            "parameters": {"n-acquire": {"voltage": 3.3}},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["operator"] == mock_current_user["email"]
    assert body["results"]["recipe_generated"] is True
    assert body["results"]["steps_count"] == 2
    assert body["results"]["step_types"] == ["bias_set", "wait_for_raman"]


def test_execute_workflow_invokes_orchestrator_when_not_testing_env(
    mock_db_session, mock_current_user, monkeypatch
):
    # Approved workflow with simple graph, same as happy path
    graph = {
        "nodes": [
            {"id": "start", "type": "start", "data": {}},
            {
                "id": "n-acquire",
                "type": "acquire",
                "data": {"voltage": 2.5, "device_id": "daq"},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "n-acquire"},
        ],
    }

    mock_workflow = MagicMock()
    mock_workflow.workflow_name = "TestWorkflow"
    mock_workflow.version = 1
    mock_workflow.definition = graph
    mock_workflow.is_approved = True
    mock_workflow.id = uuid.uuid4()

    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_workflow

    def add_side_effect(obj):
        if hasattr(obj, "run_id"):
            obj.id = uuid.uuid4()
            obj.started_at = datetime.now(timezone.utc)

    mock_db_session.add.side_effect = add_side_effect

    # Ensure environment is not 'testing' for this test so orchestrator path is used
    monkeypatch.setenv("P4_ENVIRONMENT", "development")

    # Patch orchestrator.run to avoid hitting real hardware and to return a known ID
    from retrofitkit.api import routes

    async def fake_run(recipe, operator_email, simulation=False):
        return "RUN-ORC-TEST"

    monkeypatch.setattr(routes.orc, "run", fake_run)

    response = client.post(
        "/api/workflow-builder/execute",
        json={
            "workflow_name": "TestWorkflow",
            "workflow_version": 1,
            "parameters": {},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["operator"] == mock_current_user["email"]
    assert body["results"]["recipe_generated"] is True
    assert body["results"]["orchestrator_run_id"] == "RUN-ORC-TEST"
