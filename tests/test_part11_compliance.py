import pytest
from retrofitkit.compliance.audit import write_audit_event, verify_audit_chain
from retrofitkit.db.models.audit import AuditEvent

@pytest.mark.asyncio
async def test_audit_chain_integrity(db_session):
    """Verify audit chain creation and verification."""
    # Write events
    e1 = await write_audit_event(db_session, "user1", "LOGIN", "user", "u1", {})
    e2 = await write_audit_event(db_session, "user1", "ACTION", "sample", "s1", {})
    e3 = await write_audit_event(db_session, "user1", "LOGOUT", "user", "u1", {})
    
    # Verify chain
    result = await verify_audit_chain(db_session)
    assert result["valid"] is True
    assert result["total_entries"] >= 3

@pytest.mark.asyncio
async def test_audit_tampering_detection(db_session):
    """Verify tampering is detected."""
    # Write events
    e1 = await write_audit_event(db_session, "user2", "LOGIN", "user", "u2", {})
    e2 = await write_audit_event(db_session, "user2", "ACTION", "sample", "s2", {})
    
    # Tamper with e1 (simulate DB hack)
    e1.details = '{"tampered": true}'
    await db_session.commit()
    
    # Verify chain should fail
    result = await verify_audit_chain(db_session)
    assert result["valid"] is False
    assert len(result["invalid_entries"]) > 0
