"""
Tests for authentication API endpoints.

This module tests the /login endpoint and authentication flow including:
- Successful login with valid credentials
- Failed login with invalid credentials
- Token generation and validation
- User response structure
- MFA authentication
- Audit trail logging
"""
import pytest
import os
import tempfile
import shutil
from fastapi.testclient import TestClient
from fastapi import FastAPI
from jose import jwt
from retrofitkit.api.auth import router
from retrofitkit.compliance.users import Users
from retrofitkit.compliance.tokens import SECRET, ALG
from retrofitkit.compliance.audit import get_audit_logs


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    old_db_dir = os.environ.get("P4_DATA_DIR")
    os.environ["P4_DATA_DIR"] = temp_dir
    yield temp_dir
    # Cleanup
    if old_db_dir:
        os.environ["P4_DATA_DIR"] = old_db_dir
    else:
        del os.environ["P4_DATA_DIR"]
    shutil.rmtree(temp_dir)


@pytest.fixture
def client(db_session):
    """Create a test client with auth router."""
    from retrofitkit.db.session import get_db
    app = FastAPI()
    app.include_router(router)
    
    # Override get_db to use the test session
    app.dependency_overrides[get_db] = lambda: db_session
    
    return TestClient(app)


@pytest.fixture
def test_user(db_session):
    """Create a test user in the database."""
    users = Users(db=db_session)
    users.create(
        email="test@example.com",
        name="Test User",
        role="operator",
        password="Test123!@#"
    )
    return {
        "email": "test@example.com",
        "name": "Test User",
        "role": "operator",
        "password": "Test123!@#"
    }


class TestLoginEndpoint:
    """Test cases for POST /login endpoint."""

    def test_successful_login(self, client, test_user):
        """Test successful login with valid credentials."""
        response = client.post("/login", json={
            "email": test_user["email"],
            "password": test_user["password"]
        })

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "access_token" in data
        assert "token_type" in data
        assert "user" in data

        # Verify token type
        assert data["token_type"] == "bearer"

        # Verify user data
        user = data["user"]
        assert user["email"] == test_user["email"]
        assert user["username"] == test_user["name"]
        assert user["role"] == test_user["role"]
        assert user["isActive"] is True
        assert "id" in user
        assert isinstance(user["id"], str)
        assert len(user["id"]) > 0  # Should be a UUID string

        # Verify token is valid JWT
        token = data["access_token"]
        decoded = jwt.decode(token, SECRET, algorithms=[ALG])
        assert decoded["sub"] == test_user["email"]
        assert decoded["role"] == test_user["role"]
        assert "exp" in decoded

    def test_login_invalid_email(self, client, test_user):
        """Test login with non-existent email."""
        response = client.post("/login", json={
            "email": "nonexistent@example.com",
            "password": "anypassword"
        })

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_invalid_password(self, client, test_user):
        """Test login with incorrect password."""
        response = client.post("/login", json={
            "email": test_user["email"],
            "password": "WrongPassword123"
        })

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_missing_email(self, client):
        """Test login with missing email field."""
        response = client.post("/login", json={
            "password": "Test123!@#"
        })

        assert response.status_code == 422  # Validation error

    def test_login_missing_password(self, client, test_user):
        """Test login with missing password field."""
        response = client.post("/login", json={
            "email": test_user["email"]
        })

        assert response.status_code == 422  # Validation error

    def test_login_empty_credentials(self, client):
        """Test login with empty credentials."""
        response = client.post("/login", json={
            "email": "",
            "password": ""
        })

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_case_sensitive_email(self, client, test_user):
        """Test that email is case-sensitive (or verify actual behavior)."""
        response = client.post("/login", json={
            "email": test_user["email"].upper(),
            "password": test_user["password"]
        })

        # This tests actual behavior - may need to adjust based on requirements
        assert response.status_code == 401

    def test_multiple_users_login(self, client, db_session):
        """Test login with multiple different users."""
        users = Users(db=db_session)

        # Create multiple users
        user1 = {"email": "user1@example.com", "name": "User One", "role": "operator", "password": "Pass1!"}
        user2 = {"email": "user2@example.com", "name": "User Two", "role": "supervisor", "password": "Pass2!"}

        users.create(user1["email"], user1["name"], user1["role"], user1["password"])
        users.create(user2["email"], user2["name"], user2["role"], user2["password"])

        # Test login for user1
        response1 = client.post("/login", json={
            "email": user1["email"],
            "password": user1["password"]
        })
        assert response1.status_code == 200
        assert response1.json()["user"]["role"] == "operator"

        # Test login for user2
        response2 = client.post("/login", json={
            "email": user2["email"],
            "password": user2["password"]
        })
        assert response2.status_code == 200
        assert response2.json()["user"]["role"] == "supervisor"

    def test_token_contains_correct_claims(self, client, test_user):
        """Test that generated token contains all required claims."""
        response = client.post("/login", json={
            "email": test_user["email"],
            "password": test_user["password"]
        })

        assert response.status_code == 200
        token = response.json()["access_token"]

        # Decode without verification to inspect claims
        decoded = jwt.decode(token, SECRET, algorithms=[ALG])

        # Check required claims
        assert "sub" in decoded  # Subject (email)
        assert "role" in decoded  # User role
        assert "exp" in decoded  # Expiration time

        # Verify claim values
        assert decoded["sub"] == test_user["email"]
        assert decoded["role"] == test_user["role"]

    def test_token_expiration_set(self, client, test_user):
        """Test that token has proper expiration set."""
        import time
        from datetime import datetime, timedelta, timezone

        response = client.post("/login", json={
            "email": test_user["email"],
            "password": test_user["password"]
        })

        assert response.status_code == 200
        token = response.json()["access_token"]
        decoded = jwt.decode(token, SECRET, algorithms=[ALG])

        # Check expiration is in the future
        exp_timestamp = decoded["exp"]
        current_timestamp = time.time()
        assert exp_timestamp > current_timestamp

        # Check expiration is within expected range (30 minutes)
        now_utc = datetime.now(timezone.utc)
        expected_exp = now_utc + timedelta(minutes=30)
        actual_exp = datetime.fromtimestamp(exp_timestamp, timezone.utc)

        # Allow 1 minute tolerance for test execution time
        assert abs((actual_exp - expected_exp).total_seconds()) < 60


class TestMFAAuthentication:
    """Test cases for Multi-Factor Authentication."""

    def test_mfa_required_response(self, client, test_user, db_session):
        """Test that MFA-enabled users get mfa_required response without token."""
        users = Users(db=db_session)
        secret = users.enable_mfa(test_user["email"])

        # MFA enablement should be audited
        assert isinstance(secret, str)
        logs = get_audit_logs(
            db_session,
            event_type="MFA_ENABLED",
            actor=test_user["email"],
        )
        assert len(logs) >= 1

        # Note: Current API doesn't support mfa_token parameter
        # This test documents current behavior
        response = client.post("/login", json={
            "email": test_user["email"],
            "password": test_user["password"]
        })

        # Current implementation returns 401 because API doesn't pass mfa_token
        # This should be enhanced to support MFA in the API endpoint
        # The test expects 401.
        assert response.status_code == 401


class TestAccountLockout:
    """Test cases for account lockout behavior and audit logging."""

    def test_account_locked_after_failed_attempts(self, client, test_user, db_session, temp_db_dir):
        """Account is locked after repeated failed login attempts and audited."""

        # Trigger 4 failed attempts (incrementing failed_login_attempts)
        for _ in range(4):
            resp = client.post(
                "/login",
                json={
                    "email": test_user["email"],
                    "password": "WrongPassword",
                },
            )
            assert resp.status_code == 401

        # 5th attempt with wrong password should cause lockout in the backend
        resp = client.post(
            "/login",
            json={
                "email": test_user["email"],
                "password": "WrongPassword",
            },
        )

        assert resp.status_code == 423
        assert "Account locked until" in resp.json()["detail"]

        # Next attempt with correct password should also return 423 LOCKED
        resp = client.post(
            "/login",
            json={
                "email": test_user["email"],
                "password": test_user["password"],
            },
        )

        assert resp.status_code == 423
        assert "Account locked until" in resp.json()["detail"]

        # Verify ACCOUNT_LOCKED audit event exists
        logs = get_audit_logs(
            db_session,
            event_type="ACCOUNT_LOCKED",
            actor="system",
        )
        assert any(
            entry["subject"] == f"user:{test_user['email']}" for entry in logs
        )


class TestAuditTrail:
    """Test cases for authentication audit logging."""

    def test_successful_login_audited(self, client, test_user, db_session, temp_db_dir):
        """Test that successful logins are recorded in audit trail."""
        
        response = client.post("/login", json={
            "email": test_user["email"],
            "password": test_user["password"]
        })

        assert response.status_code == 200
        logs = get_audit_logs(
            db_session,
            event_type="LOGIN_SUCCESS",
            actor=test_user["email"],
        )
        assert len(logs) >= 1

    def test_failed_login_audited(self, client, test_user, db_session, temp_db_dir):
        """Test that failed logins are recorded in audit trail."""
        
        response = client.post("/login", json={
            "email": test_user["email"],
            "password": "WrongPassword"
        })

        assert response.status_code == 401
        logs = get_audit_logs(
            db_session,
            event_type="LOGIN_FAILED",
            actor=test_user["email"],
        )
        assert len(logs) >= 1


class TestSecurityRequirements:
    """Test cases for security requirements and best practices."""

    def test_password_not_returned(self, client, test_user):
        """Test that password is never returned in response."""
        response = client.post("/login", json={
            "email": test_user["email"],
            "password": test_user["password"]
        })

        assert response.status_code == 200
        data = response.json()

        # Ensure password is not in any part of response
        response_str = str(data).lower()
        assert test_user["password"].lower() not in response_str
        assert "password" not in data
        assert "pw" not in data
        if "user" in data:
            assert "password" not in data["user"]
            assert "pw" not in data["user"]

    def test_sql_injection_attempt(self, client):
        """Test that SQL injection attempts are handled safely."""
        response = client.post("/login", json={
            "email": "admin' OR '1'='1",
            "password": "password' OR '1'='1"
        })

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_special_characters_in_credentials(self, client, db_session):
        """Test that special characters in credentials are handled correctly."""
        users = Users(db=db_session)
        special_password = "P@$$w0rd!#%&*()[]{}|<>?/\\~`"
        special_email = "test+special@example.com"

        users.create(
            email=special_email,
            name="Special User",
            role="operator",
            password=special_password
        )

        response = client.post("/login", json={
            "email": special_email,
            "password": special_password
        })

        assert response.status_code == 200
        assert response.json()["user"]["email"] == special_email
