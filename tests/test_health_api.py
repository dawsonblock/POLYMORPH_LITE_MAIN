import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from retrofitkit.api.health import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)

@pytest.fixture
def mock_db_path(tmp_path):
    # Create a dummy db
    import sqlite3
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    db_path = db_dir / "system.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.close()
    
    with patch.dict(os.environ, {"P4_DATA_DIR": str(db_dir)}):
        yield db_path

def test_database_health_check(mock_db_path):
    # Test healthy database
    response = client.get("/health/components")
    assert response.status_code == 200
    data = response.json()
    
    db_health = next(c for c in data if c["name"] == "Database")
    assert db_health["status"] == "healthy"
    assert "path" in db_health["details"]

def test_database_health_check_failure():
    # Test missing database
    with patch.dict(os.environ, {"P4_DATA_DIR": "/non/existent/path"}):
        response = client.get("/health/components")
        assert response.status_code == 200
        data = response.json()
        
        db_health = next(c for c in data if c["name"] == "Database")
        assert db_health["status"] == "error"
        assert "not found" in db_health["error_message"]
