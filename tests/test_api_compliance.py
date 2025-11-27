import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from datetime import datetime
import json
import hashlib
import uuid

from retrofitkit.api.server import app
from retrofitkit.core.app import AppContext, Config, SystemCfg, SecurityCfg, DAQCfg, RamanCfg, GatingCfg, SafetyCfg
from retrofitkit.db.models.workflow import ConfigSnapshot

client = TestClient(app)

@pytest.fixture
def mock_app_context():
    config = Config(
        system=SystemCfg(name="TestApp", mode="test", timezone="UTC", data_dir="/tmp", logs_dir="/tmp"),
        security=SecurityCfg(password_policy={}, two_person_signoff=False, jwt_exp_minutes=60, rsa_private_key="key", rsa_public_key="pub"),
        daq=DAQCfg(backend="simulator", ni={}, redpitaya={}, simulator={}),
        raman=RamanCfg(provider="stub", simulator={}, vendor={}),
        gating=GatingCfg(rules=[]),
        safety=SafetyCfg(interlocks={"enabled": True}, watchdog_seconds=1.0)
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

from retrofitkit.compliance.tokens import get_current_user

@pytest.fixture
def mock_current_user():
    user = {"email": "test@example.com", "role": "admin"}
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
