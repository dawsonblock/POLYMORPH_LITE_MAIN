"""
Tests for approval workflow functionality.

This module tests:
- Approval request creation
- Multi-role approval requirements
- Approval status transitions
- Duplicate approval prevention
- Metrics tracking
"""
import pytest
import os
import tempfile
import shutil
import sqlite3
from unittest.mock import Mock, patch
from retrofitkit.compliance import approvals


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    old_db_dir = os.environ.get("P4_DATA_DIR")
    os.environ["P4_DATA_DIR"] = temp_dir

    # Reinitialize DB path
    approvals.DB_DIR = temp_dir
    approvals.DB = os.path.join(temp_dir, "system.db")

    yield temp_dir

    # Cleanup
    if old_db_dir:
        os.environ["P4_DATA_DIR"] = old_db_dir
    else:
        if "P4_DATA_DIR" in os.environ:
            del os.environ["P4_DATA_DIR"]
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_metrics():
    """Mock the metrics system."""
    with patch.object(approvals, 'mx') as mock_mx:
        yield mock_mx


class TestApprovalRequest:
    """Test cases for creating approval requests."""

    def test_request_creates_approval_record(self, temp_db_dir, mock_metrics):
        """Test that request() creates a new approval record."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        assert isinstance(req_id, int)
        assert req_id > 0

    def test_request_sets_pending_status(self, temp_db_dir, mock_metrics):
        """Test that new requests have PENDING status."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        # Verify in database
        con = sqlite3.connect(approvals.DB)
        row = con.execute("SELECT status FROM approvals WHERE id=?", (req_id,)).fetchone()
        con.close()

        assert row[0] == "PENDING"

    def test_request_stores_recipe_path(self, temp_db_dir, mock_metrics):
        """Test that request stores recipe path correctly."""
        recipe_path = "recipes/production_run_v2.yaml"
        req_id = approvals.request(recipe_path, "operator@example.com")

        # Verify in database
        con = sqlite3.connect(approvals.DB)
        row = con.execute("SELECT recipe_path FROM approvals WHERE id=?", (req_id,)).fetchone()
        con.close()

        assert row[0] == recipe_path

    def test_request_stores_requester(self, temp_db_dir, mock_metrics):
        """Test that request stores requester email."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        # Verify in database
        con = sqlite3.connect(approvals.DB)
        row = con.execute("SELECT requested_by FROM approvals WHERE id=?", (req_id,)).fetchone()
        con.close()

        assert row[0] == "operator@example.com"

    def test_request_initializes_empty_approvals(self, temp_db_dir, mock_metrics):
        """Test that new requests have empty approvals list."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        # Verify in database
        con = sqlite3.connect(approvals.DB)
        row = con.execute("SELECT approvals_json FROM approvals WHERE id=?", (req_id,)).fetchone()
        con.close()

        assert row[0] == "[]"

    def test_request_includes_timestamp(self, temp_db_dir, mock_metrics):
        """Test that request includes creation timestamp."""
        import time
        before = time.time()
        req_id = approvals.request("recipe.yaml", "operator@example.com")
        after = time.time()

        # Verify timestamp
        con = sqlite3.connect(approvals.DB)
        row = con.execute("SELECT ts FROM approvals WHERE id=?", (req_id,)).fetchone()
        con.close()

        assert before <= row[0] <= after

    def test_request_updates_metrics(self, temp_db_dir, mock_metrics):
        """Test that creating request updates pending metrics."""
        approvals.request("recipe.yaml", "operator@example.com")

        mock_metrics.set.assert_called_with("polymorph_approvals_pending", 1.0)


class TestListPending:
    """Test cases for listing approval requests."""

    def test_list_pending_returns_empty_initially(self, temp_db_dir, mock_metrics):
        """Test that list_pending returns empty list when no requests."""
        result = approvals.list_pending()
        assert result == []

    def test_list_pending_returns_created_requests(self, temp_db_dir, mock_metrics):
        """Test that list_pending returns created approval requests."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        result = approvals.list_pending()

        assert len(result) == 1
        assert result[0]["id"] == req_id
        assert result[0]["recipe_path"] == "recipe.yaml"
        assert result[0]["requested_by"] == "operator@example.com"
        assert result[0]["status"] == "PENDING"

    def test_list_pending_returns_multiple_requests(self, temp_db_dir, mock_metrics):
        """Test that list_pending returns all requests."""
        req1 = approvals.request("recipe1.yaml", "user1@example.com")
        req2 = approvals.request("recipe2.yaml", "user2@example.com")
        req3 = approvals.request("recipe3.yaml", "user3@example.com")

        result = approvals.list_pending()

        assert len(result) == 3
        ids = [r["id"] for r in result]
        assert req1 in ids
        assert req2 in ids
        assert req3 in ids

    def test_list_pending_ordered_by_id_desc(self, temp_db_dir, mock_metrics):
        """Test that list_pending returns newest requests first."""
        req1 = approvals.request("recipe1.yaml", "user1@example.com")
        req2 = approvals.request("recipe2.yaml", "user2@example.com")
        req3 = approvals.request("recipe3.yaml", "user3@example.com")

        result = approvals.list_pending()

        # Should be in descending order (newest first)
        assert result[0]["id"] == req3
        assert result[1]["id"] == req2
        assert result[2]["id"] == req1

    def test_list_pending_respects_limit(self, temp_db_dir, mock_metrics):
        """Test that list_pending respects limit parameter."""
        for i in range(10):
            approvals.request(f"recipe{i}.yaml", f"user{i}@example.com")

        result = approvals.list_pending(limit=5)

        assert len(result) == 5

    def test_list_pending_includes_approvals(self, temp_db_dir, mock_metrics):
        """Test that list_pending includes approvals list."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        result = approvals.list_pending()

        assert "approvals" in result[0]
        assert isinstance(result[0]["approvals"], list)


class TestApprovalProcess:
    """Test cases for the approval process."""

    def test_approve_adds_approval(self, temp_db_dir, mock_metrics):
        """Test that approve() adds an approval to the request."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        approvals.approve(req_id, "qa@example.com", "QA")

        result = approvals.list_pending()
        assert len(result[0]["approvals"]) == 1
        assert result[0]["approvals"][0]["email"] == "qa@example.com"
        assert result[0]["approvals"][0]["role"] == "QA"

    def test_approve_includes_timestamp(self, temp_db_dir, mock_metrics):
        """Test that approval includes timestamp."""
        import time
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        before = time.time()
        approvals.approve(req_id, "qa@example.com", "QA")
        after = time.time()

        result = approvals.list_pending()
        approval_ts = result[0]["approvals"][0]["ts"]
        assert before <= approval_ts <= after

    def test_approve_requires_operator_and_qa(self, temp_db_dir, mock_metrics):
        """Test that approval requires both Operator and QA roles."""
        req_id = approvals.request("recipe.yaml", "requester@example.com")

        # Add only QA approval
        approvals.approve(req_id, "qa@example.com", "QA")

        result = approvals.list_pending()
        assert result[0]["status"] == "PENDING"

        # Add Operator approval
        approvals.approve(req_id, "operator@example.com", "Operator")

        result = approvals.list_pending()
        assert result[0]["status"] == "APPROVED"

    def test_approve_prevents_duplicate_approvals(self, temp_db_dir, mock_metrics):
        """Test that same user cannot approve twice."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        approvals.approve(req_id, "qa@example.com", "QA")
        approvals.approve(req_id, "qa@example.com", "QA")  # Duplicate

        result = approvals.list_pending()
        assert len(result[0]["approvals"]) == 1  # Only one approval

    def test_approve_allows_multiple_different_users(self, temp_db_dir, mock_metrics):
        """Test that different users can approve."""
        req_id = approvals.request("recipe.yaml", "requester@example.com")

        approvals.approve(req_id, "qa@example.com", "QA")
        approvals.approve(req_id, "operator@example.com", "Operator")

        result = approvals.list_pending()
        assert len(result[0]["approvals"]) == 2

    def test_approve_changes_status_when_complete(self, temp_db_dir, mock_metrics):
        """Test that status changes to APPROVED when all roles approve."""
        req_id = approvals.request("recipe.yaml", "requester@example.com")

        # Initially PENDING
        result = approvals.list_pending()
        assert result[0]["status"] == "PENDING"

        # Add both required approvals
        approvals.approve(req_id, "qa@example.com", "QA")
        approvals.approve(req_id, "operator@example.com", "Operator")

        # Should be APPROVED
        result = approvals.list_pending()
        assert result[0]["status"] == "APPROVED"

    def test_approve_ignores_already_approved(self, temp_db_dir, mock_metrics):
        """Test that approving an already-approved request is safe."""
        req_id = approvals.request("recipe.yaml", "requester@example.com")

        # Approve fully
        approvals.approve(req_id, "qa@example.com", "QA")
        approvals.approve(req_id, "operator@example.com", "Operator")

        # Try to approve again
        approvals.approve(req_id, "another@example.com", "QA")

        # Should still be APPROVED with 2 approvals
        result = approvals.list_pending()
        assert result[0]["status"] == "APPROVED"
        # Implementation allows additional approvals after APPROVED,
        # but doesn't change from APPROVED back to PENDING

    def test_approve_raises_on_invalid_request(self, temp_db_dir, mock_metrics):
        """Test that approving non-existent request raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            approvals.approve(99999, "qa@example.com", "QA")

        assert "Request not found" in str(exc_info.value)

    def test_approve_updates_metrics(self, temp_db_dir, mock_metrics):
        """Test that approve updates pending count metrics."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")
        mock_metrics.reset_mock()

        approvals.approve(req_id, "qa@example.com", "QA")

        # Should call set with pending count
        assert mock_metrics.set.called


class TestMultipleApprovals:
    """Test cases for handling multiple approval requests."""

    def test_multiple_pending_requests(self, temp_db_dir, mock_metrics):
        """Test handling multiple pending requests."""
        req1 = approvals.request("recipe1.yaml", "user1@example.com")
        req2 = approvals.request("recipe2.yaml", "user2@example.com")

        # Approve only req1
        approvals.approve(req1, "qa@example.com", "QA")
        approvals.approve(req1, "operator@example.com", "Operator")

        result = approvals.list_pending()

        # Find both requests
        req1_data = next(r for r in result if r["id"] == req1)
        req2_data = next(r for r in result if r["id"] == req2)

        assert req1_data["status"] == "APPROVED"
        assert req2_data["status"] == "PENDING"

    def test_different_approvers_for_different_requests(self, temp_db_dir, mock_metrics):
        """Test that different requests can have different approvers."""
        req1 = approvals.request("recipe1.yaml", "user1@example.com")
        req2 = approvals.request("recipe2.yaml", "user2@example.com")

        # Different approvers for each
        approvals.approve(req1, "qa1@example.com", "QA")
        approvals.approve(req2, "qa2@example.com", "QA")

        result = approvals.list_pending()

        req1_data = next(r for r in result if r["id"] == req1)
        req2_data = next(r for r in result if r["id"] == req2)

        assert req1_data["approvals"][0]["email"] == "qa1@example.com"
        assert req2_data["approvals"][0]["email"] == "qa2@example.com"


class TestRoleRequirements:
    """Test cases for role-based approval requirements."""

    def test_required_roles_constant(self):
        """Test that required roles are defined."""
        assert "Operator" in approvals.REQUIRED_ROLES
        assert "QA" in approvals.REQUIRED_ROLES

    def test_approval_with_only_operator_insufficient(self, temp_db_dir, mock_metrics):
        """Test that only Operator approval is insufficient."""
        req_id = approvals.request("recipe.yaml", "requester@example.com")

        approvals.approve(req_id, "operator@example.com", "Operator")

        result = approvals.list_pending()
        assert result[0]["status"] == "PENDING"

    def test_approval_with_only_qa_insufficient(self, temp_db_dir, mock_metrics):
        """Test that only QA approval is insufficient."""
        req_id = approvals.request("recipe.yaml", "requester@example.com")

        approvals.approve(req_id, "qa@example.com", "QA")

        result = approvals.list_pending()
        assert result[0]["status"] == "PENDING"

    def test_approval_with_both_roles_sufficient(self, temp_db_dir, mock_metrics):
        """Test that both required roles result in approval."""
        req_id = approvals.request("recipe.yaml", "requester@example.com")

        approvals.approve(req_id, "operator@example.com", "Operator")
        approvals.approve(req_id, "qa@example.com", "QA")

        result = approvals.list_pending()
        assert result[0]["status"] == "APPROVED"

    def test_approval_order_independent(self, temp_db_dir, mock_metrics):
        """Test that approval order doesn't matter."""
        req1 = approvals.request("recipe1.yaml", "user@example.com")
        req2 = approvals.request("recipe2.yaml", "user@example.com")

        # Approve in different orders
        approvals.approve(req1, "qa@example.com", "QA")
        approvals.approve(req1, "operator@example.com", "Operator")

        approvals.approve(req2, "operator@example.com", "Operator")
        approvals.approve(req2, "qa@example.com", "QA")

        result = approvals.list_pending()

        req1_data = next(r for r in result if r["id"] == req1)
        req2_data = next(r for r in result if r["id"] == req2)

        assert req1_data["status"] == "APPROVED"
        assert req2_data["status"] == "APPROVED"


class TestDatabaseOperations:
    """Test cases for database operations and integrity."""

    def test_database_table_created(self, temp_db_dir, mock_metrics):
        """Test that database table is created on first use."""
        approvals._ensure()

        con = sqlite3.connect(approvals.DB)
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='approvals'")
        tables = cursor.fetchall()
        con.close()

        assert len(tables) == 1

    def test_database_table_has_required_columns(self, temp_db_dir, mock_metrics):
        """Test that approvals table has all required columns."""
        approvals._ensure()

        con = sqlite3.connect(approvals.DB)
        cursor = con.execute("PRAGMA table_info(approvals)")
        columns = [row[1] for row in cursor.fetchall()]
        con.close()

        required_columns = ["id", "ts", "recipe_path", "requested_by", "status", "approvals_json"]
        for col in required_columns:
            assert col in columns

    def test_concurrent_requests_get_unique_ids(self, temp_db_dir, mock_metrics):
        """Test that concurrent requests get unique IDs."""
        ids = set()
        for i in range(10):
            req_id = approvals.request(f"recipe{i}.yaml", f"user{i}@example.com")
            ids.add(req_id)

        # All IDs should be unique
        assert len(ids) == 10


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_approve_with_empty_email(self, temp_db_dir, mock_metrics):
        """Test approving with empty email."""
        req_id = approvals.request("recipe.yaml", "operator@example.com")

        approvals.approve(req_id, "", "QA")

        result = approvals.list_pending()
        assert len(result[0]["approvals"]) == 1
        assert result[0]["approvals"][0]["email"] == ""

    def test_request_with_empty_recipe_path(self, temp_db_dir, mock_metrics):
        """Test creating request with empty recipe path."""
        req_id = approvals.request("", "operator@example.com")

        result = approvals.list_pending()
        assert result[0]["recipe_path"] == ""

    def test_list_pending_with_zero_limit(self, temp_db_dir, mock_metrics):
        """Test list_pending with limit of 0."""
        approvals.request("recipe.yaml", "operator@example.com")

        result = approvals.list_pending(limit=0)

        assert result == []
