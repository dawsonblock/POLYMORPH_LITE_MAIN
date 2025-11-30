import uuid
from datetime import datetime, timezone

import pytest
import httpx
from httpx import ASGITransport
from unittest.mock import MagicMock, patch, AsyncMock

from retrofitkit.api.server import app
from retrofitkit.db.models.workflow import WorkflowVersion
from retrofitkit.db.models.user import User
from retrofitkit.db.models.rbac import Role, UserRole



@pytest.fixture
def mock_db_session():
    from retrofitkit.db.session import get_db
    session = MagicMock()
    app.dependency_overrides[get_db] = lambda: session
    yield session
    app.dependency_overrides = {}

from retrofitkit.api.dependencies import get_current_user

@pytest.fixture
def mock_current_user():
    user = {"email": "test@example.com"}
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_activate_workflow_insufficient_permissions(mock_db_session, mock_current_user, client):
    # Mock user with only "scientist" role
    mock_user_obj = MagicMock()
    mock_user_obj.roles = [MagicMock(role=MagicMock(name="scientist"))]
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_user_obj
    
    response = await client.post("/api/workflow-builder/workflows/TestWorkflow/v/1/activate")
    
    assert response.status_code == 403
    assert "Insufficient permissions" in response.json()["error"]["message"]

@pytest.mark.asyncio
async def test_activate_workflow_success(mock_db_session, mock_current_user, client):
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
    
    response = await client.post("/api/workflow-builder/workflows/TestWorkflow/v/1/activate")
    
    assert response.status_code == 200
    assert "activated" in response.json()["message"]

@pytest.mark.asyncio
async def test_execute_workflow_unapproved_returns_403(mock_db_session, mock_current_user, client):

    mock_workflow = MagicMock()
    mock_workflow.is_approved = False
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_workflow

    response = await client.post(
        "/api/workflow-builder/execute",
        json={
            "workflow_name": "TestWorkflow",
            "workflow_version": 1,
            "parameters": {},
        },
    )

    assert response.status_code == 403
    assert "must be approved" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_execute_workflow_missing_workflow_returns_404(mock_db_session, mock_current_user, client):

    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = await client.post(
        "/api/workflow-builder/execute",
        json={
            "workflow_name": "MissingWorkflow",
            "parameters": {},
        },
    )

    assert response.status_code == 404
    assert "workflow" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_execute_workflow_happy_path_creates_execution(mock_db_session, mock_current_user, client):
    # Mock active workflow
    mock_workflow = MagicMock()
    mock_workflow.id = uuid.uuid4()
    mock_workflow.is_approved = True
    mock_workflow.workflow_name = "TestWorkflow"
    mock_workflow.definition = {"nodes": [{"id": "1", "type": "start", "data": {}}], "edges": []}
    
    # Mock query results
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_workflow
    
    # Mock refresh to populate started_at
    def refresh_side_effect(obj):
        obj.started_at = datetime.now(timezone.utc)
        if not obj.id:
            obj.id = uuid.uuid4()
    mock_db_session.refresh.side_effect = refresh_side_effect
    
    payload = {
        "workflow_name": "TestWorkflow",
        "parameters": {"voltage": 5.0}
    }
    
    response = await client.post(
        "/api/workflow-builder/execute",
        json=payload
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ready"
    assert "run_id" in data


@pytest.mark.asyncio
async def test_execute_workflow_with_metadata(mock_db_session, mock_current_user, client):
    # Mock active workflow
    mock_workflow = MagicMock()
    mock_workflow.id = uuid.uuid4()
    mock_workflow.is_approved = True
    mock_workflow.definition = {"nodes": [{"id": "1", "type": "start", "data": {}}], "edges": []}
    mock_workflow.workflow_name = "TestWorkflow"
    mock_workflow.version = 1
    
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_workflow
    
    # Mock refresh to populate started_at
    def refresh_side_effect(obj):
        obj.started_at = datetime.now(timezone.utc)
        if not obj.id:
            obj.id = uuid.uuid4()
    mock_db_session.refresh.side_effect = refresh_side_effect
    
    payload = {
        "workflow_name": "TestWorkflow",
        "parameters": {},
        "metadata": {"project": "ProjectX", "priority": "high"}
    }
    
    response = await client.post(
        "/api/workflow-builder/execute",
        json=payload
    )
    
    assert response.status_code == 201
    # Verify metadata was stored (mock check would be better but checking response for now)
    # Since we mock the DB, we can't easily check the stored object without more complex mocking
    # But we can check that the request didn't fail


@pytest.mark.asyncio
async def test_execute_workflow_invokes_orchestrator_when_not_testing_env(
    mock_db_session, mock_current_user, monkeypatch, client
):
    # Mock environment to NOT be testing
    monkeypatch.setenv("P4_ENVIRONMENT", "production")
    
    # Mock active workflow
    mock_workflow = MagicMock()
    mock_workflow.id = uuid.uuid4()
    mock_workflow.is_approved = True
    mock_workflow.definition = {"nodes": [{"id": "1", "type": "start", "data": {}}], "edges": []}
    mock_workflow.workflow_name = "TestWorkflow"
    mock_workflow.version = 1
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_workflow
    
    # Mock orchestrator
    fake_run = AsyncMock()
    # We need to mock where 'orc' is imported in workflow_builder.py
    # It is imported inside the function: from retrofitkit.api.routes import orc
    # So we need to mock retrofitkit.api.routes.orc
    
    # Create a mock for the module attribute
    mock_orc = MagicMock()
    mock_orc.run = fake_run
    
    # We need to patch 'retrofitkit.api.server.orc'
    import retrofitkit.api.server
    monkeypatch.setattr(retrofitkit.api.server, "orc", mock_orc)
    
    # Mock refresh to populate started_at
    def refresh_side_effect(obj):
        obj.started_at = datetime.now(timezone.utc)
        if not obj.id:
            obj.id = uuid.uuid4()
    mock_db_session.refresh.side_effect = refresh_side_effect

    payload = {
        "workflow_name": "TestWorkflow",
        "parameters": {}
    }
    
    response = await client.post(
        "/api/workflow-builder/execute",
        json=payload
    )
    
    assert response.status_code == 201
    # Verify orchestrator was called
    assert fake_run.called
    assert fake_run.call_count == 1


@pytest.mark.asyncio
async def test_abort_workflow_missing_execution_returns_404(mock_db_session, mock_current_user, client):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    response = await client.post(
        "/api/workflow-builder/executions/invalid-id/abort"
    )
    
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_abort_workflow_with_invalid_status_returns_400(mock_db_session, mock_current_user, client):
    # Execution exists but is already completed
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-COMPLETED"
    mock_execution.status = "completed"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    response = await client.post(
        "/api/workflow-builder/executions/RUN-COMPLETED/abort",
    )

    assert response.status_code == 400
    assert "cannot abort" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_abort_workflow_signals_orchestrator_when_not_testing_env(
    mock_db_session, mock_current_user, monkeypatch, client
):
    # Execution is running and has an associated orchestrator_run_id
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-BUILDER"
    mock_execution.status = "running"
    mock_execution.results = {"orchestrator_run_id": "RUN-ORC-123"}
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    # Ensure environment is not 'testing' so orchestrator path is used
    monkeypatch.setenv("P4_ENVIRONMENT", "development")

    from retrofitkit.api import server

    called = {"run_id": None}

    async def fake_abort_execution(run_id: str) -> bool:
        called["run_id"] = run_id
        return True

    monkeypatch.setattr(server.orc, "abort_execution", fake_abort_execution)

    response = await client.post(
        "/api/workflow-builder/executions/RUN-BUILDER/abort",
    )

    assert response.status_code == 200
    assert "aborted" in response.json()["message"].lower()
    # Orchestrator should be called with the underlying orchestrator run id
    assert called["run_id"] == "RUN-ORC-123"


@pytest.mark.asyncio
async def test_pause_workflow_missing_execution_returns_404(mock_db_session, mock_current_user, client):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = await client.post("/api/workflow-builder/executions/RUN-UNKNOWN/pause")

    assert response.status_code == 404
    assert "not found" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_pause_workflow_with_invalid_status_returns_400(mock_db_session, mock_current_user, client):
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-COMPLETED"
    mock_execution.status = "completed"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    response = await client.post("/api/workflow-builder/executions/RUN-COMPLETED/pause")

    assert response.status_code == 400
    assert "cannot pause" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_pause_workflow_signals_orchestrator_when_not_testing_env(
    mock_db_session, mock_current_user, client, monkeypatch
):
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-BUILDER"
    mock_execution.status = "running"
    mock_execution.results = {"orchestrator_run_id": "RUN-ORC-PAUSE"}
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    monkeypatch.setenv("P4_ENVIRONMENT", "development")

    from retrofitkit.api import server

    called = {"run_id": None}

    async def fake_pause_execution(run_id: str) -> bool:
        called["run_id"] = run_id
        return True

    monkeypatch.setattr(server.orc, "pause_execution", fake_pause_execution)

    response = await client.post("/api/workflow-builder/executions/RUN-BUILDER/pause")

    assert response.status_code == 200
    assert "paused" in response.json()["message"].lower()
    assert called["run_id"] == "RUN-ORC-PAUSE"


@pytest.mark.asyncio
async def test_resume_workflow_missing_execution_returns_404(mock_db_session, mock_current_user, client):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = await client.post("/api/workflow-builder/executions/RUN-UNKNOWN/resume")

    assert response.status_code == 404
    assert "not found" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_resume_workflow_with_invalid_status_returns_400(mock_db_session, mock_current_user, client):
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-RUNNING"
    mock_execution.status = "running"
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    response = await client.post("/api/workflow-builder/executions/RUN-RUNNING/resume")

    assert response.status_code == 400
    assert "cannot resume" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_resume_workflow_signals_orchestrator_when_not_testing_env(
    mock_db_session, mock_current_user, client, monkeypatch
):
    mock_execution = MagicMock()
    mock_execution.run_id = "RUN-BUILDER"
    mock_execution.status = "paused"
    mock_execution.results = {"orchestrator_run_id": "RUN-ORC-RESUME"}
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_execution

    monkeypatch.setenv("P4_ENVIRONMENT", "development")

    from retrofitkit.api import server

    called = {"run_id": None}

    async def fake_resume_execution(run_id: str) -> bool:
        called["run_id"] = run_id
        return True

    monkeypatch.setattr(server.orc, "resume_execution", fake_resume_execution)

    response = await client.post("/api/workflow-builder/executions/RUN-BUILDER/resume")

    assert response.status_code == 200
    assert "resumed" in response.json()["message"].lower()
    assert called["run_id"] == "RUN-ORC-RESUME"


@pytest.mark.asyncio
async def test_list_workflow_executions_with_filters(mock_db_session, mock_current_user, client):
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
    exec1.run_metadata = {}

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

    response = await client.get("/api/workflow-builder/executions", params=params)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] == "RUN-1"


@pytest.mark.asyncio
async def test_list_workflow_executions_with_metadata_filter(mock_db_session, mock_current_user, client):
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
    exec1.run_metadata = {}
    exec1.run_metadata = {"batch": "B-42"}

    query = MagicMock()
    mock_db_session.query.return_value = query
    query.filter.return_value = query
    query.order_by.return_value.limit.return_value.offset.return_value.all.return_value = [
        exec1
    ]

    params = {"metadata_key": "batch", "metadata_value": "B-42"}

    response = await client.get("/api/workflow-builder/executions", params=params)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] == "RUN-1"
    assert body[0]["run_metadata"]["batch"] == "B-42"


@pytest.mark.asyncio
async def test_get_workflow_execution_not_found(mock_db_session, mock_current_user, client):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = await client.get("/api/workflow-builder/executions/RUN-UNKNOWN")

    assert response.status_code == 404
    assert "not found" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_get_workflow_execution_happy_path(mock_db_session, mock_current_user, client):
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
    exec1.run_metadata = {}

    mock_db_session.query.return_value.filter.return_value.first.return_value = exec1

    response = await client.get("/api/workflow-builder/executions/RUN-1")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "RUN-1"
    assert body["status"] == "running"
    assert body["operator"] == mock_current_user["email"]
    assert body["results"]["foo"] == "bar"


@pytest.mark.asyncio
async def test_list_workflow_executions_with_date_filters(mock_db_session, mock_current_user, client):
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
    exec1.run_metadata = {}

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

    response = await client.get("/api/workflow-builder/executions", params=params)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] == "RUN-1"


@pytest.mark.asyncio
async def test_ui_recent_executions_endpoint(mock_db_session, mock_current_user, client):
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
    exec1.run_metadata = {"batch": "B1"}
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
    exec2.run_metadata = {"batch": "B2"}
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

    response = await client.get("/api/workflow-builder/ui/recent-executions", params=params)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {item["run_id"] for item in body} == {"RUN-1", "RUN-2"}
    for item in body:
        assert item["workflow_name"] == "TestWorkflow"
        assert item["workflow_version"] == "1"
    assert {item["run_metadata"]["batch"] for item in body} == {"B1", "B2"}



@pytest.mark.asyncio
async def test_workflow_execution_summary_endpoint(mock_db_session, mock_current_user, client):
    exec1 = MagicMock()
    exec1.status = "running"
    exec1.started_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    exec1.completed_at = datetime(2025, 1, 1, 0, 5, 0, tzinfo=timezone.utc)  # 300s
    exec1.operator = "op1@example.com"

    exec2 = MagicMock()
    exec2.status = "completed"
    exec2.started_at = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    exec2.completed_at = datetime(2025, 1, 1, 1, 10, 0, tzinfo=timezone.utc)  # 600s
    exec2.operator = "op1@example.com"

    exec3 = MagicMock()
    exec3.status = "completed"
    exec3.started_at = datetime(2025, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    exec3.completed_at = datetime(2025, 1, 1, 2, 20, 0, tzinfo=timezone.utc)  # 1200s
    exec3.operator = "op2@example.com"

    query = MagicMock()
    mock_db_session.query.return_value = query
    query.join.return_value.filter.return_value.all.return_value = [
        exec1,
        exec2,
        exec3,
    ]

    response = await client.get(
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

    # Extended analytics
    assert round(body["median_duration_seconds"]) == 600
    assert round(body["p95_duration_seconds"]) == 1200
    assert body["success_rate"] == pytest.approx(2 / 3)
    assert body["runs_per_operator"] == {
        "op1@example.com": 2,
        "op2@example.com": 1,
    }


@pytest.mark.asyncio
async def test_ui_workflow_summary_card_endpoint(mock_db_session, mock_current_user, client):
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

    response = await client.get(
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

    # Card also exposes extended analytics
    assert round(body["median_duration_seconds"]) == 600
    assert round(body["p95_duration_seconds"]) == 1200
    assert body["success_rate"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_ui_workflow_dashboard_cards(mock_db_session, mock_current_user, client, monkeypatch):
    # Distinct workflow names returned from WorkflowVersion
    query = MagicMock()
    mock_db_session.query.return_value = query
    query.distinct.return_value.all.return_value = [("WF-A",), ("WF-B",)]

    from retrofitkit.api import workflow_builder as wb

    async def fake_get_workflow_execution_summary(workflow_name: str, session=None):
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

    response = await client.get("/api/workflow-builder/ui/workflows/cards")

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


@pytest.mark.asyncio
async def test_rerun_workflow_execution_missing_original_returns_404(
    mock_db_session, mock_current_user, client
):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = await client.post(
        "/api/workflow-builder/executions/RUN-MISSING/rerun",
        json={"parameters_override": {}, "metadata_override": {}},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_rerun_workflow_execution_unapproved_workflow_returns_403(
    mock_db_session, mock_current_user, client
):
    original = MagicMock()
    original.run_id = "RUN-1"
    original.workflow_version = MagicMock()
    original.workflow_version.is_approved = False

    mock_db_session.query.return_value.filter.return_value.first.return_value = original

    response = await client.post(
        "/api/workflow-builder/executions/RUN-1/rerun",
        json={"parameters_override": {}, "metadata_override": {}},
    )

    assert response.status_code == 403
    assert "approved" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_rerun_workflow_execution_happy_path(
    mock_db_session, mock_current_user, monkeypatch, client
):
    from retrofitkit.api import workflow_builder as wb

    # Original execution with config snapshot and metadata
    original = MagicMock()
    original.run_id = "RUN-ORIG"

    workflow = MagicMock()
    workflow.workflow_name = "TestWorkflow"
    workflow.version = "1"
    workflow.is_approved = True
    original.workflow_version = workflow

    snapshot = MagicMock()
    snapshot.config_data = {"workflow_parameters": {"foo": "bar", "x": 1}}
    original.config_snapshot = snapshot
    original.run_metadata = {"batch": "B1", "sample_id": "S1"}

    mock_db_session.query.return_value.filter.return_value.first.return_value = original

    captured = {}

    async def fake_execute_workflow(exec_req, user, session=None):
        captured["req"] = exec_req

        # Minimal object compatible with WorkflowExecutionResponse
        result = MagicMock()
        result.id = uuid.uuid4()
        result.run_id = "RUN-RERUN"
        result.workflow_version_id = uuid.uuid4()
        result.started_at = datetime.now(timezone.utc)
        result.completed_at = None
        result.status = "ready"
        result.operator = user["email"]
        result.results = {"recipe_generated": True}
        result.error_message = None
        result.run_metadata = exec_req.metadata
        return result

    monkeypatch.setattr(wb, "execute_workflow", fake_execute_workflow)

    response = await client.post(
        "/api/workflow-builder/executions/RUN-ORIG/rerun",
        json={
            "parameters_override": {"x": 2},
            "metadata_override": {"batch": "B2"},
        },
    )

    assert response.status_code == 201
    body = response.json()
    # New run id from fake executor
    assert body["run_id"] == "RUN-RERUN"
    # Parameters were merged before delegating
    assert captured["req"].parameters == {"foo": "bar", "x": 2}
    # Metadata was merged and exposed on the response
    assert body["run_metadata"] == {"batch": "B2", "sample_id": "S1"}
