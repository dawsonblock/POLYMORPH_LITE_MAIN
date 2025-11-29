"""
Tests for multi-tenant enforcement.

Tests:
- Organization context injection
- Automatic query scoping
- Cross-org access prevention
- OrgScopedSession
- Permission checking
"""
import pytest
from fastapi import FastAPI, Depends, Request
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from retrofitkit.security.multi_tenant import (
    OrgContext,
    get_org_context,
    multi_tenant_middleware,
    scope_to_org,
    OrgScopedSession,
    enforce_org_access,
    check_org_permission
)
from retrofitkit.db.models.sample import Sample
from retrofitkit.db.session import get_db


@pytest.fixture
def org_context_org1():
    """Create org context for org 1."""
    return OrgContext(
        org_id="org-1",
        org_name="Lab 1",
        user_email="user1@lab1.com"
    )


@pytest.fixture
def org_context_org2():
    """Create org context for org 2."""
    return OrgContext(
        org_id="org-2",
        org_name="Lab 2",
        user_email="user2@lab2.com"
    )


def test_org_context_creation(org_context_org1):
    """Test OrgContext creation."""
    assert org_context_org1.org_id == "org-1"
    assert org_context_org1.org_name == "Lab 1"
    assert org_context_org1.user_email == "user1@lab1.com"


def test_org_context_repr(org_context_org1):
    """Test OrgContext string representation."""
    repr_str = repr(org_context_org1)
    assert "org-1" in repr_str
    assert "Lab 1" in repr_str
    assert "user1@lab1.com" in repr_str


def test_enforce_org_access_same_org(org_context_org1):
    """Test that same org access is allowed."""
    # Should not raise
    enforce_org_access("org-1", org_context_org1, "sample")


def test_enforce_org_access_different_org(org_context_org1):
    """Test that different org access is blocked."""
    from fastapi import HTTPException
    
    with pytest.raises(HTTPException) as exc_info:
        enforce_org_access("org-2", org_context_org1, "sample")
        
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_check_org_permission_same_org():
    """Test permission checking for same org."""
    user = {
        "email": "user@lab.com",
        "org_id": "org-1",
        "roles": ["scientist"]
    }
    
    # Should have read permission
    assert check_org_permission(user, "org-1", "read") is True
    
    # Should have write permission (scientist role)
    assert check_org_permission(user, "org-1", "write") is True
    
    # Should not have admin permission
    assert check_org_permission(user, "org-1", "admin") is False


def test_check_org_permission_different_org():
    """Test permission checking for different org."""
    user = {
        "email": "user@lab.com",
        "org_id": "org-1",
        "roles": ["admin"]
    }
    
    # Should not have any permission in different org
    assert check_org_permission(user, "org-2", "read") is False
    assert check_org_permission(user, "org-2", "write") is False
    assert check_org_permission(user, "org-2", "admin") is False


def test_check_org_permission_admin():
    """Test admin permissions."""
    user = {
        "email": "admin@lab.com",
        "org_id": "org-1",
        "roles": ["admin"]
    }
    
    assert check_org_permission(user, "org-1", "read") is True
    assert check_org_permission(user, "org-1", "write") is True
    assert check_org_permission(user, "org-1", "admin") is True


def test_check_org_permission_viewer():
    """Test viewer permissions."""
    user = {
        "email": "viewer@lab.com",
        "org_id": "org-1",
        "roles": ["viewer"]
    }
    
    assert check_org_permission(user, "org-1", "read") is True
    assert check_org_permission(user, "org-1", "write") is False
    assert check_org_permission(user, "org-1", "admin") is False


def test_org_scoped_session_query(session, org_context_org1):
    """Test OrgScopedSession query scoping."""
    scoped_session = OrgScopedSession(session, org_context_org1)
    
    # Create samples in different orgs
    sample1 = Sample(
        sample_id="SAMPLE-1",
        lot_number="LOT-1",
        org_id="org-1"
    )
    sample2 = Sample(
        sample_id="SAMPLE-2",
        lot_number="LOT-2",
        org_id="org-2"
    )
    
    session.add(sample1)
    session.add(sample2)
    session.commit()
    
    # Query through scoped session should only return org-1 samples
    samples = scoped_session.query(Sample).all()
    assert len(samples) == 1
    assert samples[0].org_id == "org-1"


def test_org_scoped_session_add(session, org_context_org1):
    """Test OrgScopedSession automatically sets org_id."""
    scoped_session = OrgScopedSession(session, org_context_org1)
    
    # Create sample without org_id
    sample = Sample(
        sample_id="SAMPLE-3",
        lot_number="LOT-3"
    )
    
    scoped_session.add(sample)
    session.commit()
    
    # Should have org_id set automatically
    assert sample.org_id == "org-1"


def test_multi_tenant_middleware_with_user():
    """Test multi-tenant middleware with authenticated user."""
    app = FastAPI()
    
    @app.get("/test")
    async def test_endpoint(request: Request):
        org_context = request.state.org_context
        return {
            "org_id": org_context.org_id,
            "org_name": org_context.org_name
        }
        
    # Add middleware
    app.middleware("http")(multi_tenant_middleware)
    
    client = TestClient(app)
    
    # Mock authenticated user in request state
    # (In real app, this would be set by auth middleware)
    # For testing, we'll need to inject it differently
    # This test would need integration with auth system


def test_multi_tenant_middleware_public_endpoint():
    """Test that public endpoints skip org context."""
    app = FastAPI()
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
        
    app.middleware("http")(multi_tenant_middleware)
    
    client = TestClient(app)
    response = client.get("/health")
    
    # Should succeed without org context
    assert response.status_code == 200


def test_scope_to_org(session, org_context_org1):
    """Test scope_to_org helper function."""
    # Create samples in different orgs
    sample1 = Sample(sample_id="S1", lot_number="L1", org_id="org-1")
    sample2 = Sample(sample_id="S2", lot_number="L2", org_id="org-2")
    
    session.add(sample1)
    session.add(sample2)
    session.commit()
    
    # Query all samples
    query = session.query(Sample)
    
    # Scope to org-1
    scoped_query = scope_to_org(query, org_context_org1, Sample)
    samples = scoped_query.all()
    
    # Should only return org-1 samples
    assert len(samples) == 1
    assert samples[0].org_id == "org-1"


def test_org_isolation_integration(session):
    """Integration test for org isolation."""
    # Create two org contexts
    org1_context = OrgContext("org-1", "Lab 1", "user1@lab1.com")
    org2_context = OrgContext("org-2", "Lab 2", "user2@lab2.com")
    
    # Create scoped sessions
    org1_session = OrgScopedSession(session, org1_context)
    org2_session = OrgScopedSession(session, org2_context)
    
    # Org 1 creates a sample
    sample1 = Sample(sample_id="ORG1-SAMPLE", lot_number="LOT1")
    org1_session.add(sample1)
    session.commit()
    
    # Org 2 creates a sample
    sample2 = Sample(sample_id="ORG2-SAMPLE", lot_number="LOT2")
    org2_session.add(sample2)
    session.commit()
    
    # Org 1 should only see their sample
    org1_samples = org1_session.query(Sample).all()
    assert len(org1_samples) == 1
    assert org1_samples[0].sample_id == "ORG1-SAMPLE"
    
    # Org 2 should only see their sample
    org2_samples = org2_session.query(Sample).all()
    assert len(org2_samples) == 1
    assert org2_samples[0].sample_id == "ORG2-SAMPLE"
    
    # Verify org_ids are set correctly
    assert sample1.org_id == "org-1"
    assert sample2.org_id == "org-2"
