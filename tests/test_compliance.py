import pytest
import os
import sqlite3
import json
import time
from retrofitkit.compliance.audit import write_audit_event, verify_audit_chain
from retrofitkit.db.models.audit import AuditEvent
from retrofitkit.db.session import SessionLocal, engine
from retrofitkit.db.base import Base

@pytest.fixture
def audit_db(db_session):
    return db_session

@pytest.mark.asyncio
async def test_audit_log_hash_chain(audit_db):
    # 1. Write first event
    e1 = await write_audit_event(
        audit_db, 
        "user1", 
        "TEST_EVENT", 
        "test", 
        "1", 
        {"foo": "bar"}
    )
    
    assert e1.prev_hash == "GENESIS"
    assert e1.hash is not None
    
    # 2. Write second event
    e2 = await write_audit_event(
        audit_db, 
        "user1", 
        "TEST_EVENT_2", 
        "test", 
        "2", 
        {"foo": "baz"}
    )
    
    assert e2.prev_hash == e1.hash
    
    # 3. Verify Chain
    result = await verify_audit_chain(audit_db)
    assert result["valid"] is True
    assert result["total_entries"] == 2

@pytest.mark.asyncio
async def test_audit_tamper_detection(audit_db):
    # 1. Write events
    e1 = await write_audit_event(audit_db, "user1", "E1", "t", "1", {})
    e2 = await write_audit_event(audit_db, "user1", "E2", "t", "2", {})
    
    # 2. Tamper with E1
    e1.details = json.dumps({"tampered": True})
    await audit_db.commit()
    
    # 3. Verify Chain - should fail
    result = await verify_audit_chain(audit_db)
    assert result["valid"] is False
    assert len(result["invalid_entries"]) > 0
