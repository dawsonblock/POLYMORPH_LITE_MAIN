import pytest
import asyncio
from fastapi.testclient import TestClient
from retrofitkit.api.server import app
from retrofitkit.core.registry import registry
from retrofitkit.core.app import AppContext
from retrofitkit.database.models import Base, engine
from retrofitkit.db.session import get_db

@pytest.fixture
def client(db_session):
    """Create test client with DB session override."""
    app.dependency_overrides[get_db] = lambda: db_session
    yield TestClient(app)
    app.dependency_overrides = {}

# Mock data for auth
LOGIN_PAYLOAD = {
    "username": "operator@example.com",
    "password": "password123" # Mock auth accepts any password in dev/test usually, or we mock the auth provider
}

@pytest.mark.asyncio
async def test_golden_run_headless_gui(client):
    """
    Simulate a full user session:
    1. Login
    2. Check Status
    3. Submit Run
    4. Verify Audit Trail
    """
    # 1. Login (Mocking auth if needed, but let's try real endpoint if configured)
    # In test env, we might need to override dependency or use a known user.
    # Assuming 'operator@example.com' exists or we can mock the token.
    
    # For this integration test, let's assume we can get a token via a helper or mock.
    # But wait, we are running against the FastAPI app instance directly.
    # We can override the get_current_user dependency for simplicity in this test
    # to avoid dealing with real JWT generation/validation complexities here.
    
    from retrofitkit.api.security import get_current_user as get_security_user
    from retrofitkit.api.dependencies import get_current_user as get_dependency_user
    from types import SimpleNamespace
    
    # Mock user as dict for legacy security
    mock_user_dict = {
        "email": "operator@example.com",
        "role": "Operator",
        "permissions": ["run:execute", "audit:read"]
    }
    
    # Mock user as object for new dependencies
    mock_user_obj = SimpleNamespace(
        email="operator@example.com",
        role="Operator",
        permissions=["run:execute", "audit:read"]
    )
    
    app.dependency_overrides[get_security_user] = lambda: mock_user_dict
    app.dependency_overrides[get_dependency_user] = lambda: mock_user_obj
    
    # 2. Check Status
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "orchestrator" in data
    assert data["user"]["email"] == "operator@example.com"
    
    # 3. Submit Run
    # We need a valid recipe. Let's create a temporary one or use a simple dict.
    # The API expects a path. In simulation, we can pass a dummy path if we mock the loader,
    # or write a real temp file.
    
    import tempfile
    import yaml
    import os
    
    import uuid
    recipe_data = {
        "id": str(uuid.uuid4()),
        "name": "Integration Test Recipe",
        "steps": [
            {"type": "wait", "params": {"seconds": 0.1}},
            {"type": "compute", "params": {}}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(recipe_data, tmp)
        recipe_path = tmp.name
        
    try:
        # Request Run (Approval flow)
        # For simplicity, let's assume auto-approval in dev or we mock Approvals.
        # But the API enforces it.
        # Let's use the /api/run endpoint directly if we can bypass approval for tests,
        # OR we simulate the approval.
        
        # Step 3a: Request
        resp = client.post("/api/request_run", json={"recipe_path": recipe_path, "simulation": True})
        assert resp.status_code == 200
        req_id = resp.json()["request_id"]
        
        # Step 3b: Approve (as Operator)
        # The requester must also approve explicitly in this implementation
        app.dependency_overrides[get_security_user] = lambda: mock_user_dict
        app.dependency_overrides[get_dependency_user] = lambda: mock_user_obj
        resp = client.post("/api/approve", json={"request_id": req_id})
        assert resp.status_code == 200

        # Step 3c: Approve (as QA)
        mock_qa_obj = SimpleNamespace(
            email="qa@example.com",
            role="QA",
            permissions=["run:approve"]
        )
        mock_qa_dict = {
            "email": "qa@example.com",
            "role": "QA",
            "permissions": ["run:approve"]
        }
        app.dependency_overrides[get_security_user] = lambda: mock_qa_dict
        app.dependency_overrides[get_dependency_user] = lambda: mock_qa_obj
        
        resp = client.post("/api/approve", json={"request_id": req_id})
        assert resp.status_code == 200
        
        # Step 3d: Execute (as Operator)
        app.dependency_overrides[get_security_user] = lambda: mock_user_dict
        app.dependency_overrides[get_dependency_user] = lambda: mock_user_obj
        
        # Note: /api/run is async, TestClient handles async endpoints but the Orchestrator.run is async.
        # TestClient uses Starlette's TestClient which runs in a thread. 
        # Orchestrator.run spawns a task or awaits? 
        # In routes.py: await orc.run(...)
        # So it waits for completion? 
        # WorkflowExecutor.execute is async. 
        # If it waits, this call might block for 0.1s (wait step).
        
        resp = client.post("/api/run", json={"recipe_path": recipe_path, "simulation": True})
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        assert run_id is not None
        
        # 4. Verify Audit Trail
        # Call compliance endpoint
        resp = client.get("/api/compliance/audit/verify-chain")
        assert resp.status_code == 200
        audit_data = resp.json()

        if not audit_data["is_valid"]:
            print("\nAudit Verification Failed!")
            print("Errors:", audit_data.get("errors"))
            print("Entries:", audit_data.get("total_entries"))

        assert audit_data["is_valid"] is True
        # We expect at least RUN_REQUEST, RUN_APPROVE, RUN_START, STEP_COMPLETE x2, RUN_COMPLETE
        assert audit_data["total_entries"] >= 5 
        
    finally:
        if os.path.exists(recipe_path):
            os.remove(recipe_path)
        app.dependency_overrides = {}
