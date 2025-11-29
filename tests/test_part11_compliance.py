import pytest
from retrofitkit.compliance.audit import write_audit_event, verify_audit_chain
from retrofitkit.db.models.audit import AuditEvent

def test_audit_chain_integrity(db_session):
    """Verify audit chain creation and verification."""
    # Write events
    e1 = write_audit_event(db_session, "user1", "LOGIN", "user", "u1", {})
    e2 = write_audit_event(db_session, "user1", "ACTION", "sample", "s1", {})
    e3 = write_audit_event(db_session, "user1", "LOGOUT", "user", "u1", {})
    
    # Verify chain
    result = verify_audit_chain(db_session)
    assert result["valid"] is True
    assert result["entries_checked"] >= 3

def test_audit_tampering_detection(db_session):
    """Verify tampering is detected."""
    # Write events
    e1 = write_audit_event(db_session, "user2", "LOGIN", "user", "u2", {})
    e2 = write_audit_event(db_session, "user2", "ACTION", "sample", "s2", {})
    
    # Tamper with e1 (simulate DB hack)
    # We need to bypass ORM or just update field and commit
    e1.details = '{"tampered": true}'
    db_session.commit()
    
    # Verify chain should fail
    result = verify_audit_chain(db_session)
    assert result["valid"] is False
    assert len(result["errors"]) > 0
    assert result["errors"][0]["error"] == "hash_mismatch"
