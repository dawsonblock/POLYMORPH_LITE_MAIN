import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from datetime import datetime

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
