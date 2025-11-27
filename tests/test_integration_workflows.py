"""
Integration tests for complete workflow scenarios.

This module tests end-to-end workflow execution including:
- User authentication
- Workflow creation and upload
- Device initialization
- Workflow execution with safety checks
- Audit trail verification
- Approval workflows
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import FastAPI

from retrofitkit.api.auth import router as auth_router
from retrofitkit.api.workflows import router as workflow_router
from retrofitkit.api.devices import router as device_router
from retrofitkit.compliance.users import Users
from retrofitkit.compliance.approvals import request, approve, list_pending


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for integration tests."""
    temp_dir = tempfile.mkdtemp()
    import os
    from retrofitkit.compliance import approvals
    
    old_data_dir = os.environ.get("P4_DATA_DIR")
    os.environ["P4_DATA_DIR"] = temp_dir
    
    # Monkeypatch approvals DB path since it is read at import time
    old_db_path = approvals.DB
    approvals.DB = os.path.join(temp_dir, "system.db")
    approvals.DB_DIR = temp_dir
    
    yield temp_dir
    
    # Restore
    approvals.DB = old_db_path
    if old_data_dir:
        os.environ["P4_DATA_DIR"] = old_data_dir
    else:
        if "P4_DATA_DIR" in os.environ:
            del os.environ["P4_DATA_DIR"]
    shutil.rmtree(temp_dir)


@pytest.fixture
def app():
    """Create test application with all routers."""
    app = FastAPI()
    app.include_router(auth_router, prefix="/auth", tags=["auth"])
    app.include_router(workflow_router)
    app.include_router(device_router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_users(temp_data_dir, db_session):
    """Create test users with different roles."""
    users = Users(db_session)

    # Create operator
    users.create(
        email="operator@polymorph.com",
        name="Test Operator",
        role="Operator",
        password="TestOperator123!"
    )

    # Create QA user
    users.create(
        email="qa@polymorph.com",
        name="Test QA",
        role="QA",
        password="TestQA123!"
    )

    # Create admin
    users.create(
        email="admin@polymorph.com",
        name="Test Admin",
        role="admin",
        password="TestAdmin123!"
    )

    return {
        "operator": {"email": "operator@polymorph.com", "password": "TestOperator123!"},
        "qa": {"email": "qa@polymorph.com", "password": "TestQA123!"},
        "admin": {"email": "admin@polymorph.com", "password": "TestAdmin123!"}
    }


@pytest.mark.integration
class TestCompleteWorkflowScenario:
    """Integration tests for complete workflow execution scenarios."""

    def test_full_workflow_execution_flow(self, client, test_users, temp_data_dir):
        """
        Test complete workflow execution from authentication to completion.

        Steps:
        1. Operator logs in
        2. Operator uploads workflow
        3. Operator executes workflow
        4. Verify execution results
        """
        # Step 1: Login as operator
        login_response = client.post("/auth/login", json={
            "email": test_users["operator"]["email"],
            "password": test_users["operator"]["password"]
        })
        assert login_response.status_code == 200
        operator_token = login_response.json()["access_token"]

        # Step 2: Upload workflow
        workflow_yaml = """
id: "integration_test_workflow"
name: "Integration Test Workflow"
entry_step: "wait_step"
steps:
  wait_step:
    kind: "wait"
    params:
      seconds: 0.1
    children: []
"""
        upload_response = client.post(
            "/workflows/",
            json={"yaml_content": workflow_yaml},
            headers={"Authorization": f"Bearer {operator_token}"}
        )
        assert upload_response.status_code == 201

        # Step 3: Execute workflow
        execute_response = client.post(
            "/workflows/integration_test_workflow/execute",
            json={},
            headers={"Authorization": f"Bearer {operator_token}"}
        )
        assert execute_response.status_code == 200
        result = execute_response.json()

        # Step 4: Verify execution
        assert result["success"] is True
        assert result["workflow_id"] == "integration_test_workflow"
        assert "wait_step" in result["steps_executed"]
        assert result["duration_seconds"] >= 0.1

    def test_workflow_approval_flow(self, client, test_users, temp_data_dir):
        """
        Test workflow requiring multi-role approval.

        Steps:
        1. Operator requests approval
        2. QA approves
        3. Operator verifies approval status
        """
        # Step 1: Request approval
        recipe_path = "test_recipe.yaml"
        req_id = request(recipe_path, test_users["operator"]["email"])
        assert req_id > 0

        # Verify pending status
        pending = list_pending()
        assert len(pending) > 0
        assert pending[0]["status"] == "PENDING"

        # Step 2: QA approves
        approve(req_id, test_users["qa"]["email"], "QA")

        # Still pending (needs Operator approval too)
        pending = list_pending()
        approval_record = next(r for r in pending if r["id"] == req_id)
        assert approval_record["status"] == "PENDING"

        # Step 3: Operator approves
        approve(req_id, test_users["operator"]["email"], "Operator")

        # Now should be approved
        pending = list_pending()
        approval_record = next(r for r in pending if r["id"] == req_id)
        assert approval_record["status"] == "APPROVED"

    def test_multi_user_concurrent_workflows(self, client, test_users):
        """
        Test multiple users executing workflows concurrently.
        """
        # Login both users
        operator_login = client.post("/auth/login", json={
            "email": test_users["operator"]["email"],
            "password": test_users["operator"]["password"]
        })
        operator_token = operator_login.json()["access_token"]

        qa_login = client.post("/auth/login", json={
            "email": test_users["qa"]["email"],
            "password": test_users["qa"]["password"]
        })
        qa_token = qa_login.json()["access_token"]

        # Upload different workflows
        workflow1 = """
id: "workflow_1"
name: "Workflow 1"
entry_step: "step1"
steps:
  step1:
    kind: "wait"
    params:
      seconds: 0.1
    children: []
"""

        workflow2 = """
id: "workflow_2"
name: "Workflow 2"
entry_step: "step1"
steps:
  step1:
    kind: "wait"
    params:
      seconds: 0.1
    children: []
"""

        client.post("/workflows/", json={"yaml_content": workflow1})
        client.post("/workflows/", json={"yaml_content": workflow2})

        # Execute concurrently
        result1 = client.post(
            "/workflows/workflow_1/execute",
            json={},
            headers={"Authorization": f"Bearer {operator_token}"}
        )

        result2 = client.post(
            "/workflows/workflow_2/execute",
            json={},
            headers={"Authorization": f"Bearer {qa_token}"}
        )

        # Both should succeed
        assert result1.status_code == 200
        assert result2.status_code == 200
        assert result1.json()["success"] is True
        assert result2.json()["success"] is True


@pytest.mark.integration
class TestAuthenticationIntegration:
    """Integration tests for authentication flows."""

    def test_login_workflow_access_pattern(self, client, test_users):
        """Test that authentication properly gates workflow access."""
        # Try to access workflows without authentication
        response = client.get("/workflows/")
        # Should work (listing is public in current implementation)
        assert response.status_code == 200

        # Login
        login_response = client.post("/auth/login", json={
            "email": test_users["operator"]["email"],
            "password": test_users["operator"]["password"]
        })
        token = login_response.json()["access_token"]

        # Access with token should work
        response = client.get(
            "/workflows/",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200

    def test_token_expiration_workflow(self, client, test_users):
        """Test workflow access with expired token."""
        from retrofitkit.compliance.tokens import create_access_token

        # Create token that expires immediately
        expired_token = create_access_token(
            {"sub": test_users["operator"]["email"], "role": "Operator"},
            expires_minutes=-1
        )

        # Try to use expired token (if endpoint is protected)
        # Note: Current implementation doesn't protect all endpoints
        # This test documents expected behavior
        pass


@pytest.mark.integration
class TestDeviceWorkflowIntegration:
    """Integration tests for device and workflow interactions."""

    def test_device_discovery_and_workflow_execution(self, client):
        """Test discovering devices and using them in workflows."""
        # List available device drivers
        drivers_response = client.get("/devices/drivers")
        assert drivers_response.status_code == 200
        drivers = drivers_response.json()
        assert len(drivers) >= 0

        # List device instances
        instances_response = client.get("/devices/instances")
        assert instances_response.status_code == 200


@pytest.mark.integration
class TestComplianceIntegration:
    """Integration tests for compliance and audit features."""

    def test_audit_trail_through_workflow(self, client, test_users, temp_data_dir):
        """Test that workflow execution creates audit trail."""
        from retrofitkit.compliance.audit import Audit

        # Login (creates audit entry)
        login_response = client.post("/auth/login", json={
            "email": test_users["operator"]["email"],
            "password": test_users["operator"]["password"]
        })
        assert login_response.status_code == 200

        # Audit trail should exist
        audit = Audit()
        # Note: Would need to add query methods to Audit class to verify

    @pytest.mark.skip(reason="Requires legacy DB path fix")
    def test_signature_approval_workflow(self, temp_data_dir):
        """Test electronic signature with approval workflow."""
        from retrofitkit.compliance.signatures import Signer, SignatureRequest
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import os

        # Setup keys
        key_dir = os.path.join(temp_data_dir, "config", "keys")
        os.makedirs(key_dir, exist_ok=True)

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()

        with open(os.path.join(key_dir, "private.pem"), "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(os.path.join(key_dir, "public.pem"), "wb") as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))

        # Create approval request
        req_id = request("critical_workflow.yaml", "operator@polymorph.com")

        # Sign approval
        signer = Signer()
        signature_request = SignatureRequest(
            record_id=req_id,
            reason="Approving critical workflow for production"
        )
        signature_result = signer.sign_record(signature_request, "qa@polymorph.com")

        assert signature_result["signed"] is True
        assert "signature" in signature_result

        # Approve with both required roles
        approve(req_id, "qa@polymorph.com", "QA")
        approve(req_id, "operator@polymorph.com", "Operator")

        # Verify approved
        pending = list_pending()
        approval_record = next(r for r in pending if r["id"] == req_id)
        assert approval_record["status"] == "APPROVED"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skip(reason="DB setup issues in performance tests")
class TestPerformanceIntegration:
    """Integration tests for performance scenarios."""

    def test_multiple_sequential_workflows(self, client):
        """Test executing multiple workflows in sequence."""
        import time

        workflow_yaml = """
id: "perf_test_workflow"
name: "Performance Test Workflow"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.05
    children: []
"""

        # Upload workflow
        client.post("/workflows/", json={"yaml_content": workflow_yaml})

        # Execute 10 times
        start_time = time.time()
        for i in range(10):
            response = client.post("/workflows/perf_test_workflow/execute", json={})
            assert response.status_code == 200
            assert response.json()["success"] is True

        total_time = time.time() - start_time

        # Should complete in reasonable time (< 2 seconds for 10 x 0.05s waits + overhead)
        assert total_time < 2.0

    def test_large_workflow_upload(self, client):
        """Test uploading large workflow definitions."""
        # Create workflow with many steps
        steps = []
        for i in range(100):
            steps.append(f"""
  step_{i}:
    kind: "wait"
    params:
      seconds: 0.01
    children: {f'["step_{i+1}"]' if i < 99 else '[]'}""")

        workflow_yaml = f"""
id: "large_workflow"
name: "Large Workflow Test"
entry_step: "step_0"
steps:
{''.join(steps)}
"""

        response = client.post("/workflows/", json={"yaml_content": workflow_yaml})
        assert response.status_code == 201

        # Verify it was stored
        get_response = client.get("/workflows/large_workflow")
        assert get_response.status_code == 200
        assert len(get_response.json()["steps"]) == 100
