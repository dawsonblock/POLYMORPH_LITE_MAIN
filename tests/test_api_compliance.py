"""
Tests for Enhanced Compliance API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import uuid

from retrofitkit.api.server import app

client = TestClient(app)


def mock_get_current_user():
    return {"email": "compliance@example.com", "name": "Compliance Officer", "role": "QA"}


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token"}


class TestAuditTrailVerification:
    """Test audit trail integrity verification."""

    @patch("retrofitkit.api.compliance.get_current_user", return_value=mock_get_current_user())
    def test_verify_audit_chain(self, mock_auth, auth_headers):
        """Test audit chain verification endpoint."""
        response = client.get(
            "/api/compliance/audit/verify-chain",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "is_valid" in data
        assert "total_entries" in data
        assert "verified_entries" in data
        assert "chain_start_hash" in data
        assert "chain_end_hash" in data
        assert "errors" in data

        # Valid chain should have no errors
        if data["is_valid"]:
            assert len(data["errors"]) == 0

    def test_export_audit_trail(self, auth_headers):
        """Test audit trail export."""
        response = client.get(
            "/api/compliance/audit/export",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "export_date" in data
        assert "exported_by" in data
        assert "entries" in data
        assert isinstance(data["entries"], list)

    def test_export_audit_trail_with_filters(self, auth_headers):
        """Test audit trail export with date filtering."""
        response = client.get(
            "/api/compliance/audit/export",
            params={
                "start_date": "2025-01-01T00:00:00",
                "actor": "test@example.com"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["start_date"] == "2025-01-01T00:00:00"
        assert data["filters"]["actor"] == "test@example.com"


class TestPDFReportGeneration:
    """Test PDF compliance report generation."""

    def test_generate_run_report_pdf(self, auth_headers):
        """Test PDF report generation for a run."""
        # Note: This will fail if run doesn't exist, but tests the endpoint
        response = client.get(
            "/api/compliance/reports/run/test-run-id.pdf",
            headers=auth_headers
        )

        # Should return 404 if run doesn't exist, or PDF if it does
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Check it's a PDF
            assert response.headers["content-type"] == "application/pdf"
            assert "Content-Disposition" in response.headers


class TestTraceabilityMatrix:
    """Test traceability matrix generation."""

    def test_generate_traceability_matrix(self, auth_headers):
        """Test generating traceability matrix for a sample."""
        # This will return 404 for non-existent samples
        response = client.get(
            "/api/compliance/traceability/sample/SAMPLE-TEST-001",
            headers=auth_headers
        )

        # Should return 404 if sample doesn't exist
        if response.status_code == 404:
            assert "not found" in response.json()["detail"].lower()
        elif response.status_code == 200:
            data = response.json()
            assert "sample_id" in data
            assert "runs" in data
            assert "total_runs" in data
            assert isinstance(data["runs"], list)


class TestConfigurationVersioning:
    """Test configuration snapshot management."""

    @patch("retrofitkit.api.compliance.get_current_user", return_value=mock_get_current_user())
    def test_create_config_snapshot(self, mock_auth, auth_headers):
        """Test creating a configuration snapshot."""
        response = client.post(
            "/api/compliance/config/snapshot",
            params={"reason": "Pre-deployment snapshot"},
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()

        assert "snapshot_id" in data
        assert "config_hash" in data
        assert "created_by" in data
        assert data["created_by"] == "compliance@example.com"
        assert data["reason"] == "Pre-deployment snapshot"

    def test_list_config_snapshots(self, auth_headers):
        """Test listing configuration snapshots."""
        response = client.get(
            "/api/compliance/config/snapshots",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_config_snapshot(self, auth_headers):
        """Test retrieving a specific config snapshot."""
        # First, list snapshots to get an ID
        list_response = client.get(
            "/api/compliance/config/snapshots",
            headers=auth_headers
        )

        if list_response.status_code == 200 and len(list_response.json()) > 0:
            snapshot_id = list_response.json()[0]["snapshot_id"]

            # Get specific snapshot
            response = client.get(
                f"/api/compliance/config/snapshots/{snapshot_id}",
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["snapshot_id"] == snapshot_id


class TestComplianceIntegration:
    """Integration tests for compliance workflows."""

    @patch("retrofitkit.api.compliance.get_current_user", return_value=mock_get_current_user())
    def test_full_compliance_workflow(self, mock_auth, auth_headers):
        """Test complete compliance workflow: snapshot → audit → verify."""
        # 1. Create config snapshot
        snapshot_resp = client.post(
            "/api/compliance/config/snapshot",
            params={"reason": "Integration test"},
            headers=auth_headers
        )
        assert snapshot_resp.status_code == 201
        snapshot = snapshot_resp.json()

        # 2. Verify audit chain (should include snapshot creation)
        verify_resp = client.get(
            "/api/compliance/audit/verify-chain",
            headers=auth_headers
        )
        assert verify_resp.status_code == 200
        verification = verify_resp.json()

        # Should have at least the snapshot creation event
        assert verification["total_entries"] > 0

        # 3. Export audit trail
        export_resp = client.get(
            "/api/compliance/audit/export",
            headers=auth_headers
        )
        assert export_resp.status_code == 200
        export_data = export_resp.json()

        # Should contain our snapshot event
        events = [e for e in export_data["entries"] if "CONFIG_SNAPSHOT" in e.get("event", "")]
        assert len(events) > 0


class TestComplianceValidation:
    """Test validation and error handling in compliance endpoints."""

    def test_invalid_snapshot_id(self, auth_headers):
        """Test retrieving non-existent snapshot."""
        response = client.get(
            "/api/compliance/config/snapshots/invalid-id-12345",
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_invalid_sample_traceability(self, auth_headers):
        """Test traceability for non-existent sample."""
        response = client.get(
            "/api/compliance/traceability/sample/NONEXISTENT-SAMPLE",
            headers=auth_headers
        )

        assert response.status_code == 404
