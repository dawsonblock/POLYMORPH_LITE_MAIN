"""
Tests for Multi-Tenant Isolation.

Verifies that org_id filtering is enforced across all data access.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from retrofitkit.api.middleware.org_context import (
    OrgContextMiddleware,
    require_org_context,
    get_org_context,
)


class TestOrgContextMiddleware:
    """Test organization context middleware."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock(spec=Request)
        request.url.path = "/api/data"
        request.headers = {}
        request.state = MagicMock()
        return request

    def test_exempt_paths(self):
        """Test that exempt paths skip org context."""
        exempt_paths = [
            "/health",
            "/healthz",
            "/metrics",
            "/docs",
            "/auth/login",
            "/auth/token",
        ]
        
        middleware = OrgContextMiddleware(
            app=MagicMock(),
            secret_key="test-secret"
        )
        
        for path in exempt_paths:
            assert middleware._is_exempt(path) is True

    def test_non_exempt_paths(self):
        """Test that API paths are not exempt."""
        paths = [
            "/api/workflows",
            "/api/devices",
            "/workflows/123",
        ]
        
        middleware = OrgContextMiddleware(
            app=MagicMock(),
            secret_key="test-secret"
        )
        
        for path in paths:
            assert middleware._is_exempt(path) is False

    def test_require_org_context_raises_without_org(self, mock_request):
        """Test that require_org_context raises when org_id is missing."""
        mock_request.state.org_id = None
        
        with pytest.raises(HTTPException) as exc_info:
            require_org_context(mock_request)
        
        assert exc_info.value.status_code == 403
        assert "Organization context required" in str(exc_info.value.detail)

    def test_require_org_context_returns_org_id(self, mock_request):
        """Test that require_org_context returns org_id when present."""
        mock_request.state.org_id = "org-123"
        
        result = require_org_context(mock_request)
        assert result == "org-123"

    def test_get_org_context_returns_none_when_missing(self, mock_request):
        """Test that get_org_context returns None when missing."""
        mock_request.state.org_id = None
        
        result = get_org_context(mock_request)
        assert result is None

    def test_get_org_context_returns_org_id(self, mock_request):
        """Test that get_org_context returns org_id when present."""
        mock_request.state.org_id = "org-456"
        
        result = get_org_context(mock_request)
        assert result == "org-456"


class TestMultiTenantIsolation:
    """Test that data is properly isolated by org_id."""

    def test_cross_org_access_denied(self):
        """
        Test that accessing data from another org is denied.
        
        This is a placeholder - real implementation requires
        database fixtures and full integration test setup.
        """
        # In a real test, we would:
        # 1. Create data for org-A
        # 2. Authenticate as org-B
        # 3. Try to access org-A's data
        # 4. Assert access is denied
        
        # For now, just verify the middleware behavior
        middleware = OrgContextMiddleware(
            app=MagicMock(),
            secret_key="test-secret"
        )
        
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer invalid.token.here"}
        
        org_id, user_email = middleware._extract_org_context(mock_request)
        
        # Invalid token should return None
        assert org_id is None

    def test_valid_jwt_extracts_org_id(self):
        """Test that valid JWT extracts org_id correctly."""
        import jwt
        
        secret = "test-secret-key"
        token = jwt.encode(
            {"sub": "user@example.com", "org_id": "org-789"},
            secret,
            algorithm="HS256"
        )
        
        middleware = OrgContextMiddleware(
            app=MagicMock(),
            secret_key=secret
        )
        
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"Authorization": f"Bearer {token}"}
        
        org_id, user_email = middleware._extract_org_context(mock_request)
        
        assert org_id == "org-789"
        assert user_email == "user@example.com"

    def test_expired_jwt_returns_none(self):
        """Test that expired JWT returns None for org_id."""
        import jwt
        from datetime import datetime, timedelta
        
        secret = "test-secret-key"
        token = jwt.encode(
            {
                "sub": "user@example.com",
                "org_id": "org-123",
                "exp": datetime.utcnow() - timedelta(hours=1)
            },
            secret,
            algorithm="HS256"
        )
        
        middleware = OrgContextMiddleware(
            app=MagicMock(),
            secret_key=secret
        )
        
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"Authorization": f"Bearer {token}"}
        
        org_id, user_email = middleware._extract_org_context(mock_request)
        
        assert org_id is None
