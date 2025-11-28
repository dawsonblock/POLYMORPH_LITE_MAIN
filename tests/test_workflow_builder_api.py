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


def test_abort_workflow_missing_execution_returns_404(mock_db_session, mock_current_user):
    # No execution found for given run_id
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = client.post(
        "/api/workflow-builder/executions/RUN-UNKNOWN/abort",
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_abort_workflow_with_invalid_status_returns_400(mock_db_session, mock_current_user):
    # Execution exists but is already completed
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-COMPLETED"
    mock_execution.status = "completed"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    response = client.post(
        "/api/workflow-builder/executions/RUN-COMPLETED/abort",
    )

    assert response.status_code == 400
    assert "cannot abort" in response.json()["detail"].lower()


def test_abort_workflow_signals_orchestrator_when_not_testing_env(
    mock_db_session, mock_current_user, monkeypatch
):
    # Execution is running and has an associated orchestrator_run_id
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-BUILDER"
    mock_execution.status = "running"
    mock_execution.results = {"orchestrator_run_id": "RUN-ORC-123"}
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    # Ensure environment is not 'testing' so orchestrator path is used
    monkeypatch.setenv("P4_ENVIRONMENT", "development")

    from retrofitkit.api import routes

    called = {"run_id": None}

    async def fake_abort_execution(run_id: str) -> bool:
        called["run_id"] = run_id
        return True

    monkeypatch.setattr(routes.orc, "abort_execution", fake_abort_execution)

    response = client.post(
        "/api/workflow-builder/executions/RUN-BUILDER/abort",
    )

    assert response.status_code == 200
    assert "aborted" in response.json()["message"].lower()
    # Orchestrator should be called with the underlying orchestrator run id
    assert called["run_id"] == "RUN-ORC-123"


def test_pause_workflow_missing_execution_returns_404(mock_db_session, mock_current_user):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = client.post("/api/workflow-builder/executions/RUN-UNKNOWN/pause")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_pause_workflow_with_invalid_status_returns_400(mock_db_session, mock_current_user):
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-COMPLETED"
    mock_execution.status = "completed"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    response = client.post("/api/workflow-builder/executions/RUN-COMPLETED/pause")

    assert response.status_code == 400
    assert "cannot pause" in response.json()["detail"].lower()


def test_pause_workflow_signals_orchestrator_when_not_testing_env(
    mock_db_session, mock_current_user, monkeypatch
):
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-BUILDER"
    mock_execution.status = "running"
    mock_execution.results = {"orchestrator_run_id": "RUN-ORC-PAUSE"}
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    monkeypatch.setenv("P4_ENVIRONMENT", "development")

    from retrofitkit.api import routes

    called = {"run_id": None}

    async def fake_pause_execution(run_id: str) -> bool:
        called["run_id"] = run_id
        return True

    monkeypatch.setattr(routes.orc, "pause_execution", fake_pause_execution)

    response = client.post("/api/workflow-builder/executions/RUN-BUILDER/pause")

    assert response.status_code == 200
    assert "paused" in response.json()["message"].lower()
    assert called["run_id"] == "RUN-ORC-PAUSE"


def test_resume_workflow_missing_execution_returns_404(mock_db_session, mock_current_user):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = client.post("/api/workflow-builder/executions/RUN-UNKNOWN/resume")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_resume_workflow_with_invalid_status_returns_400(mock_db_session, mock_current_user):
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-RUNNING"
    mock_execution.status = "running"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    response = client.post("/api/workflow-builder/executions/RUN-RUNNING/resume")

    assert response.status_code == 400
    assert "cannot resume" in response.json()["detail"].lower()


def test_resume_workflow_signals_orchestrator_when_not_testing_env(
    mock_db_session, mock_current_user, monkeypatch
):
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-BUILDER"
    mock_execution.status = "paused"
    mock_execution.results = {"orchestrator_run_id": "RUN-ORC-RESUME"}
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    monkeypatch.setenv("P4_ENVIRONMENT", "development")

    from retrofitkit.api import routes

    called = {"run_id": None}

    async def fake_resume_execution(run_id: str) -> bool:
        called["run_id"] = run_id
        return True

    monkeypatch.setattr(routes.orc, "resume_execution", fake_resume_execution)

    response = client.post("/api/workflow-builder/executions/RUN-BUILDER/resume")

    assert response.status_code == 200
    assert "resumed" in response.json()["message"].lower()
    assert called["run_id"] == "RUN-ORC-RESUME"


def test_list_workflow_executions_with_filters(mock_db_session, mock_current_user):
    # Prepare mocked executions
    exec1 = MagicMock()
    exec1.run_id = "RUN-1"
    exec1.id = uuid.uuid4()
    exec1.workflow_version_id = uuid.uuid4()
    exec1.started_at = datetime.now(timezone.utc)
    exec1.completed_at = None
    exec1.status = "running"
    exec1.operator = "op@example.com"
    exec1.results = {}
    exec1.error_message = None

    # Configure query chain
    query = MagicMock()
    mock_db_session.query.return_value = query
    query.join.return_value = query
    query.filter.return_value = query
    query.order_by.return_value.limit.return_value.offset.return_value.all.return_value = [
        exec1
    ]

    params = {
        "workflow_name": "TestWorkflow",
        "status": "running",
        "operator": "op@example.com",
    }

    response = client.get("/api/workflow-builder/executions", params=params)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] == "RUN-1"


def test_get_workflow_execution_not_found(mock_db_session, mock_current_user):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = client.get("/api/workflow-builder/executions/RUN-UNKNOWN")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_workflow_execution_happy_path(mock_db_session, mock_current_user):
    exec1 = MagicMock()
    exec1.run_id = "RUN-1"
    exec1.id = uuid.uuid4()
    exec1.workflow_version_id = uuid.uuid4()
    exec1.started_at = datetime.now(timezone.utc)
    exec1.completed_at = None
    exec1.status = "running"
    exec1.operator = mock_current_user["email"]
    exec1.results = {"foo": "bar"}
    exec1.error_message = None

    mock_db_session.query.return_value.filter.return_value.first.return_value = exec1

    response = client.get("/api/workflow-builder/executions/RUN-1")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "RUN-1"
    assert body["status"] == "running"
    assert body["operator"] == mock_current_user["email"]
    assert body["results"]["foo"] == "bar"


def test_list_workflow_executions_with_date_filters(mock_db_session, mock_current_user):
    exec1 = MagicMock()
    exec1.run_id = "RUN-1"
    exec1.id = uuid.uuid4()
    exec1.workflow_version_id = uuid.uuid4()
    exec1.started_at = datetime.now(timezone.utc)
    exec1.completed_at = None
    exec1.status = "completed"
    exec1.operator = "user@example.com"
    exec1.results = {}
    exec1.error_message = None

    query = MagicMock()
    mock_db_session.query.return_value = query
    query.join.return_value = query
    query.filter.return_value = query
    query.order_by.return_value.limit.return_value.offset.return_value.all.return_value = [
        exec1
    ]

    started_after = datetime(2025, 1, 1, tzinfo=timezone.utc)
    started_before = datetime(2025, 1, 2, tzinfo=timezone.utc)

    params = {
        "started_after": started_after.isoformat(),
        "started_before": started_before.isoformat(),
    }

    response = client.get("/api/workflow-builder/executions", params=params)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] == "RUN-1"


def test_ui_recent_executions_endpoint(mock_db_session, mock_current_user):
    exec1 = MagicMock()
    exec1.run_id = "RUN-1"
    exec1.id = uuid.uuid4()
    exec1.workflow_version_id = uuid.uuid4()
    exec1.started_at = datetime.now(timezone.utc)
    exec1.completed_at = None
    exec1.status = "running"
    exec1.operator = "user@example.com"
    exec1.results = {}
    exec1.error_message = None
    exec1.workflow_version = MagicMock(workflow_name="TestWorkflow", version="1")

    exec2 = MagicMock()
    exec2.run_id = "RUN-2"
    exec2.id = uuid.uuid4()
    exec2.workflow_version_id = uuid.uuid4()
    exec2.started_at = datetime.now(timezone.utc)
    exec2.completed_at = None
    exec2.status = "completed"
    exec2.operator = "user@example.com"
    exec2.results = {}
    exec2.error_message = None
    exec2.workflow_version = MagicMock(workflow_name="TestWorkflow", version="1")

    query = MagicMock()
    mock_db_session.query.return_value = query
    query.join.return_value = query
    query.filter.return_value = query
    query.order_by.return_value.limit.return_value.all.return_value = [
        exec1,
        exec2,
    ]

    params = {"workflow_name": "TestWorkflow", "limit": 2}

    response = client.get("/api/workflow-builder/ui/recent-executions", params=params)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {item["run_id"] for item in body} == {"RUN-1", "RUN-2"}
    for item in body:
        assert item["workflow_name"] == "TestWorkflow"
        assert item["workflow_version"] == "1"



def test_workflow_execution_summary_endpoint(mock_db_session, mock_current_user):
    exec1 = MagicMock()
    exec1.status = "running"
    exec1.started_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    exec1.completed_at = datetime(2025, 1, 1, 0, 5, 0, tzinfo=timezone.utc)  # 300s

    exec2 = MagicMock()
    exec2.status = "completed"
    exec2.started_at = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    exec2.completed_at = datetime(2025, 1, 1, 1, 10, 0, tzinfo=timezone.utc)  # 600s

    exec3 = MagicMock()
    exec3.status = "completed"
    exec3.started_at = datetime(2025, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    exec3.completed_at = datetime(2025, 1, 1, 2, 20, 0, tzinfo=timezone.utc)  # 1200s

    query = MagicMock()
    mock_db_session.query.return_value = query
    query.join.return_value.filter.return_value.all.return_value = [
        exec1,
        exec2,
        exec3,
    ]

    response = client.get(
        "/api/workflow-builder/workflows/TestWorkflow/executions/summary"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["workflow_name"] == "TestWorkflow"
    assert body["total"] == 3
    assert body["by_status"]["running"] == 1
    assert body["by_status"]["completed"] == 2

    # Average duration over runs with both started_at and completed_at
    # Durations: 300, 600, 1200 -> average = 700
    assert round(body["average_duration_seconds"]) == 700

    # Last run is exec3 (latest started_at)
    assert body["last_run_status"] == "completed"
    assert body["last_run_started_at"].startswith(
        exec3.started_at.strftime("%Y-%m-%dT%H:%M:%S")
    )


def test_ui_workflow_summary_card_endpoint(mock_db_session, mock_current_user):
    exec1 = MagicMock()
    exec1.status = "running"
    exec1.started_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    exec1.completed_at = datetime(2025, 1, 1, 0, 5, 0, tzinfo=timezone.utc)

    exec2 = MagicMock()
    exec2.status = "completed"
    exec2.started_at = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    exec2.completed_at = datetime(2025, 1, 1, 1, 10, 0, tzinfo=timezone.utc)

    exec3 = MagicMock()
    exec3.status = "completed"
    exec3.started_at = datetime(2025, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    exec3.completed_at = datetime(2025, 1, 1, 2, 20, 0, tzinfo=timezone.utc)

    query = MagicMock()
    mock_db_session.query.return_value = query
    query.join.return_value.filter.return_value.all.return_value = [
        exec1,
        exec2,
        exec3,
    ]

    response = client.get(
        "/api/workflow-builder/ui/workflows/TestWorkflow/card"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["workflow_name"] == "TestWorkflow"
    assert body["total_runs"] == 3
    assert body["running_count"] == 1
    assert body["completed_count"] == 2
    # Average duration ~ 700 seconds
    assert round(body["average_duration_seconds"]) == 700
    assert body["last_run_status"] == "completed"


def test_ui_workflow_dashboard_cards(mock_db_session, mock_current_user, monkeypatch):
    # Distinct workflow names returned from WorkflowVersion
    query = MagicMock()
    mock_db_session.query.return_value = query
    query.distinct.return_value.all.return_value = [("WF-A",), ("WF-B",)]

    from retrofitkit.api import workflow_builder as wb

    async def fake_get_workflow_execution_summary(workflow_name: str):
        if workflow_name == "WF-A":
            return wb.WorkflowExecutionSummaryResponse(
                workflow_name="WF-A",
                total=3,
                by_status={"running": 1, "completed": 2},
                average_duration_seconds=100.0,
                last_run_status="completed",
                last_run_started_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                last_run_completed_at=datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc),
            )
        else:
            return wb.WorkflowExecutionSummaryResponse(
                workflow_name="WF-B",
                total=1,
                by_status={"failed": 1},
                average_duration_seconds=None,
                last_run_status="failed",
                last_run_started_at=None,
                last_run_completed_at=None,
            )

    monkeypatch.setattr(
        wb, "get_workflow_execution_summary", fake_get_workflow_execution_summary
    )

    response = client.get("/api/workflow-builder/ui/workflows/cards")

    assert response.status_code == 200
    body = response.json()
    assert {card["workflow_name"] for card in body} == {"WF-A", "WF-B"}

    wf_a = next(card for card in body if card["workflow_name"] == "WF-A")
    assert wf_a["total_runs"] == 3
    assert wf_a["running_count"] == 1
    assert wf_a["completed_count"] == 2
    assert wf_a["failed_count"] == 0
    assert wf_a["aborted_count"] == 0
    assert wf_a["average_duration_seconds"] == 100.0
    assert wf_a["last_run_status"] == "completed"

    wf_b = next(card for card in body if card["workflow_name"] == "WF-B")
    assert wf_b["total_runs"] == 1
    assert wf_b["running_count"] == 0
    assert wf_b["completed_count"] == 0
    assert wf_b["failed_count"] == 1
    assert wf_b["aborted_count"] == 0
    assert wf_b["average_duration_seconds"] is None
    assert wf_b["last_run_status"] == "failed"
