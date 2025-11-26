"""
Tests for security middleware and authentication.

This module tests:
- JWT token validation and user extraction
- Security headers middleware
- Rate limiting middleware
- Authentication dependency injection
"""
import pytest
import time
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from retrofitkit.api.security import get_current_user
from retrofitkit.security.headers import SecurityHeadersMiddleware, RateLimitMiddleware
from retrofitkit.compliance.tokens import create_access_token


class TestGetCurrentUser:
    """Test cases for JWT authentication dependency."""

    def test_valid_token(self):
        """Test authentication with valid bearer token."""
        token = create_access_token({"sub": "test@example.com", "role": "operator"})

        user = get_current_user(authorization=f"Bearer {token}")

        assert user["email"] == "test@example.com"
        assert user["role"] == "operator"

    def test_missing_authorization_header(self):
        """Test authentication without authorization header."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(authorization=None)

        assert exc_info.value.status_code == 401
        assert "Missing bearer token" in exc_info.value.detail

    def test_missing_bearer_prefix(self):
        """Test authentication with token but without 'Bearer' prefix."""
        from fastapi import HTTPException

        token = create_access_token({"sub": "test@example.com", "role": "operator"})

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(authorization=token)

        assert exc_info.value.status_code == 401
        assert "Missing bearer token" in exc_info.value.detail

    def test_invalid_token_format(self):
        """Test authentication with malformed JWT token."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(authorization="Bearer invalid_token_format")

        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail

    def test_expired_token(self):
        """Test authentication with expired token."""
        from fastapi import HTTPException

        # Create token that expires immediately
        token = create_access_token({"sub": "test@example.com", "role": "operator"}, expires_minutes=-1)

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(authorization=f"Bearer {token}")

        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail

    def test_token_without_subject(self):
        """Test token without 'sub' or 'email' claim."""
        from fastapi import HTTPException

        token = create_access_token({"role": "operator"})  # Missing sub/email

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(authorization=f"Bearer {token}")

        assert exc_info.value.status_code == 401
        assert "Invalid token payload" in exc_info.value.detail

    def test_token_with_email_claim(self):
        """Test token with 'email' claim instead of 'sub'."""
        token = create_access_token({"email": "test@example.com", "role": "admin"})

        user = get_current_user(authorization=f"Bearer {token}")

        assert user["email"] == "test@example.com"
        assert user["role"] == "admin"

    def test_default_role_when_missing(self):
        """Test that default role is assigned when missing from token."""
        token = create_access_token({"sub": "test@example.com"})  # No role

        user = get_current_user(authorization=f"Bearer {token}")

        assert user["email"] == "test@example.com"
        assert user["role"] == "Operator"  # Default role

    def test_bearer_case_insensitive(self):
        """Test that 'Bearer' prefix is case-insensitive."""
        token = create_access_token({"sub": "test@example.com", "role": "operator"})

        # Test lowercase
        user = get_current_user(authorization=f"bearer {token}")
        assert user["email"] == "test@example.com"

        # Test mixed case
        user = get_current_user(authorization=f"BeArEr {token}")
        assert user["email"] == "test@example.com"

    def test_token_with_extra_whitespace(self):
        """Test token parsing handles extra whitespace."""
        token = create_access_token({"sub": "test@example.com", "role": "operator"})

        user = get_current_user(authorization=f"Bearer  {token}  ")

        assert user["email"] == "test@example.com"


class TestSecurityHeadersMiddleware:
    """Test cases for security headers middleware."""

    @pytest.fixture
    def app(self):
        """Create test app with security headers middleware."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_x_content_type_options_header(self, client):
        """Test X-Content-Type-Options header is set."""
        response = client.get("/test")
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_x_frame_options_header(self, client):
        """Test X-Frame-Options header is set."""
        response = client.get("/test")
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_xss_protection_header(self, client):
        """Test X-XSS-Protection header is set."""
        response = client.get("/test")
        assert response.headers["X-XSS-Protection"] == "1; mode=block"

    def test_hsts_header(self, client):
        """Test Strict-Transport-Security header is set."""
        response = client.get("/test")
        assert "Strict-Transport-Security" in response.headers
        hsts = response.headers["Strict-Transport-Security"]
        assert "max-age=31536000" in hsts
        assert "includeSubDomains" in hsts

    def test_csp_header(self, client):
        """Test Content-Security-Policy header is set."""
        response = client.get("/test")
        assert "Content-Security-Policy" in response.headers
        csp = response.headers["Content-Security-Policy"]

        # Verify key directives
        assert "default-src 'self'" in csp
        assert "script-src" in csp
        assert "style-src" in csp
        assert "frame-ancestors 'none'" in csp

    def test_permissions_policy_header(self, client):
        """Test Permissions-Policy header is set."""
        response = client.get("/test")
        assert "Permissions-Policy" in response.headers
        policy = response.headers["Permissions-Policy"]

        # Verify dangerous features are disabled
        assert "geolocation=()" in policy
        assert "microphone=()" in policy
        assert "camera=()" in policy
        assert "payment=()" in policy

    def test_referrer_policy_header(self, client):
        """Test Referrer-Policy header is set."""
        response = client.get("/test")
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_server_header_removed(self, client):
        """Test Server header is removed for security obscurity."""
        response = client.get("/test")
        assert "Server" not in response.headers

    def test_headers_on_all_responses(self, client):
        """Test security headers are added to all responses."""
        app = client.app

        @app.get("/another")
        def another_endpoint():
            return {"message": "another"}

        response = client.get("/another")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_headers_on_error_responses(self, client):
        """Test security headers are added even to error responses."""
        app = client.app

        @app.get("/error")
        def error_endpoint():
            raise Exception("Test error")

        response = client.get("/error")
        # Should have security headers even on 500 error
        assert "X-Content-Type-Options" in response.headers


class TestRateLimitMiddleware:
    """Test cases for rate limiting middleware."""

    @pytest.fixture
    def app(self):
        """Create test app with rate limiting middleware."""
        app = FastAPI()
        # Set low limits for testing: 5 requests per 10 seconds
        app.add_middleware(RateLimitMiddleware, requests=5, window_sec=10)

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        @app.get("/other")
        def other_endpoint():
            return {"message": "other"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_rate_limit_headers_present(self, client):
        """Test that rate limit headers are included in response."""
        response = client.get("/test")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_allows_requests_under_limit(self, client):
        """Test that requests under the limit are allowed."""
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200
            assert response.json()["message"] == "test"

    def test_rate_limit_blocks_excess_requests(self, client):
        """Test that requests exceeding the limit are blocked."""
        # Make 5 successful requests
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200

        # 6th request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_rate_limit_remaining_decrements(self, client):
        """Test that remaining count decrements with each request."""
        response = client.get("/test")
        remaining1 = int(response.headers["X-RateLimit-Remaining"])

        response = client.get("/test")
        remaining2 = int(response.headers["X-RateLimit-Remaining"])

        assert remaining2 == remaining1 - 1

    def test_rate_limit_retry_after_header(self, client):
        """Test that Retry-After header is set when rate limited."""
        # Exhaust rate limit
        for i in range(5):
            client.get("/test")

        # Get rate limited response
        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        retry_after = int(response.headers["Retry-After"])
        assert 0 <= retry_after <= 10

    def test_rate_limit_per_path(self, client):
        """Test that rate limits are tracked per path."""
        # Make 5 requests to /test
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200

        # /test should be rate limited
        response = client.get("/test")
        assert response.status_code == 429

        # But /other should still work (different path)
        response = client.get("/other")
        assert response.status_code == 200

    def test_rate_limit_refills_over_time(self, client):
        """Test that rate limit tokens refill over time."""
        # Make 5 requests to exhaust limit
        for i in range(5):
            client.get("/test")

        # Should be rate limited
        response = client.get("/test")
        assert response.status_code == 429

        # Wait for some refill (2 seconds = 1 token at 5 req/10sec)
        time.sleep(2.1)

        # Should allow 1 more request
        response = client.get("/test")
        assert response.status_code == 200

    def test_rate_limit_response_structure(self, client):
        """Test the structure of rate limit error response."""
        # Exhaust limit
        for i in range(5):
            client.get("/test")

        response = client.get("/test")
        assert response.status_code == 429

        data = response.json()
        assert "detail" in data
        assert "retry_after_sec" in data
        assert isinstance(data["retry_after_sec"], int)

    def test_rate_limit_limit_header_value(self, client):
        """Test that X-RateLimit-Limit reflects configured limit."""
        response = client.get("/test")
        assert response.headers["X-RateLimit-Limit"] == "5"

    def test_rate_limit_zero_remaining_when_blocked(self, client):
        """Test that remaining is 0 when rate limited."""
        # Exhaust limit
        for i in range(5):
            client.get("/test")

        response = client.get("/test")
        assert response.status_code == 429
        assert response.headers["X-RateLimit-Remaining"] == "0"


class TestProtectedEndpoint:
    """Test protected endpoints requiring authentication."""

    @pytest.fixture
    def app(self):
        """Create test app with protected endpoint."""
        app = FastAPI()

        @app.get("/protected")
        def protected_endpoint(user: dict = Depends(get_current_user)):
            return {"message": f"Hello {user['email']}", "role": user["role"]}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_protected_endpoint_with_valid_token(self, client):
        """Test accessing protected endpoint with valid token."""
        token = create_access_token({"sub": "test@example.com", "role": "admin"})

        response = client.get("/protected", headers={
            "Authorization": f"Bearer {token}"
        })

        assert response.status_code == 200
        assert response.json()["message"] == "Hello test@example.com"
        assert response.json()["role"] == "admin"

    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get("/protected")
        assert response.status_code == 401

    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        response = client.get("/protected", headers={
            "Authorization": "Bearer invalid_token"
        })
        assert response.status_code == 401
