import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from datetime import datetime
import json
import hashlib
import uuid

from retrofitkit.api.server import app
from retrofitkit.core.app import AppContext, Config, SystemCfg, SecurityCfg, DAQCfg, RamanCfg, GatingCfg, SafetyCfg
from retrofitkit.db.models.workflow import ConfigSnapshot, WorkflowExecution, WorkflowVersion
from retrofitkit.database.models import AuditEvent as AuditLog

client = TestClient(app)

@pytest.fixture
def mock_app_context(monkeypatch):
    monkeypatch.setenv("P4_SYSTEM_NAME", "TestApp")
    monkeypatch.setenv("P4_ENVIRONMENT", "testing")
    config = Config(
        system=SystemCfg(name="TestApp", environment="testing", timezone="UTC", data_dir="/tmp", logs_dir="/tmp"),
        security=SecurityCfg(password_policy={}, two_person_signoff=False, jwt_exp_minutes=60, rsa_private_key="key", rsa_public_key="pub"),
        daq=DAQCfg(backend="simulator", ni={}, redpitaya={}, simulator={}),
        raman=RamanCfg(provider="stub", simulator={}, vendor={}),
        gating=GatingCfg(rules=[]),
        safety=SafetyCfg(estop_line=0, door_line=1, watchdog_seconds=1.0)
    )
    with patch("retrofitkit.core.app.AppContext.load") as mock_load:
        mock_load.return_value = MagicMock(config=config)
        yield mock_load

@pytest.fixture
def mock_db_session():
    with patch("retrofitkit.api.compliance.get_session") as mock_get_session:
        session = MagicMock()
        mock_get_session.return_value = session
        # Mock Alembic revision
        session.execute.return_value.scalar.return_value = "12345abcdef"
        
        # Mock refresh to set ID
        def mock_refresh(obj):
            if hasattr(obj, 'id') and obj.id is None:
                obj.id = uuid.uuid4()
        session.refresh.side_effect = mock_refresh
        
        yield session

from retrofitkit.api.dependencies import get_current_user

@pytest.fixture
def mock_current_user():
    from types import SimpleNamespace
    user = SimpleNamespace(email="test@example.com", role="admin")
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    app.dependency_overrides = {}

def test_create_config_snapshot(mock_app_context, mock_db_session, mock_current_user):
    response = client.post(
        "/api/compliance/config/snapshot",
        params={"reason": "Test Snapshot"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["reason"] == "Test Snapshot"
    assert data["created_by"] == "test@example.com"
    assert "config_hash" in data
    assert len(data["config_hash"]) == 64  # SHA256 length
    
    # Verify DB interaction
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    
    # Inspect the added object
    added_snapshot = mock_db_session.add.call_args[0][0]
    added_snapshot.id = uuid.uuid4() # Mock DB ID generation
    assert isinstance(added_snapshot, ConfigSnapshot)
    assert added_snapshot.reason == "Test Snapshot"
    assert added_snapshot.config_data["system"]["app_name"] == "TestApp"
    assert added_snapshot.config_data["database"]["alembic_revision"] == "12345abcdef"


def test_get_run_details_not_found(mock_db_session, mock_current_user):
    # No execution found
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    response = client.get("/api/compliance/run/RUN-UNKNOWN")

    assert response.status_code == 404
    assert "not found" in response.json()["error"]["message"].lower()


def test_get_run_details_happy_path(mock_db_session, mock_current_user):
    run_id = "RUN-123"

    # Mock execution
    execution = MagicMock()
    execution.run_id = run_id
    execution.status = "completed"
    execution.operator = "test@example.com"
    execution.started_at = datetime(2025, 1, 1)
    execution.completed_at = datetime(2025, 1, 1, 1, 0, 0)
    execution.workflow_version_id = uuid.uuid4()
    execution.config_snapshot_id = uuid.uuid4()

    # Mock workflow
    workflow = MagicMock()
    workflow.workflow_name = "TestWorkflow"
    workflow.version = "1"
    workflow.definition_hash = "abc123"

    # Mock config snapshot
    config = MagicMock()
    config.snapshot_id = "SNAP-1"
    config.timestamp = datetime(2025, 1, 1, 0, 0, 0)
    config.config_hash = "deadbeef" * 8
    config.created_by = "test@example.com"
    config.reason = "Test Run"

    # Mock audit entry
    audit_entry = MagicMock()
    audit_entry.id = 1
    audit_entry.ts = 1234567890.0
    audit_entry.event = "RUN_COMPLETED"
    audit_entry.actor = "test@example.com"
    audit_entry.subject = run_id
    audit_entry.details = "{}"
    audit_entry.hash = "h1"
    audit_entry.prev_hash = "h0"

    # Configure query side effects based on model
    def query_side_effect(model):
        query = MagicMock()
        if model == WorkflowExecution:
            query.filter.return_value.first.return_value = execution
        elif model == WorkflowVersion:
            query.filter.return_value.first.return_value = workflow
        elif model == ConfigSnapshot:
            query.filter.return_value.first.return_value = config
        elif model == AuditLog:
            query.filter.return_value.order_by.return_value.all.return_value = [audit_entry]
        return query

    mock_db_session.query.side_effect = query_side_effect

    response = client.get(f"/api/compliance/run/{run_id}")

    assert response.status_code == 200
    body = response.json()

    assert body["run_id"] == run_id
    assert body["status"] == "completed"
    assert body["operator"] == "test@example.com"
    assert body["workflow_name"] == "TestWorkflow"
    assert body["workflow_version"] == "1"
    assert body["workflow_hash"] == "abc123"

    assert body["config_snapshot"]["snapshot_id"] == "SNAP-1"
    assert body["config_snapshot"]["config_hash"] == config.config_hash

    assert len(body["audit_entries"]) == 1
    assert body["audit_entries"][0]["event"] == "RUN_COMPLETED"
