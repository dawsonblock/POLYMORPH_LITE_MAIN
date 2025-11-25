import pytest
import os
import sqlite3
import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
import datetime
from retrofitkit.compliance.audit import CompliantAuditTrail

# Helper to generate keys
def generate_test_keys():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key

@pytest.fixture
def audit_db(tmp_path):
    db_path = tmp_path / "audit.db"
    # Patch DB path in audit module
    with pytest.MonkeyPatch.context() as m:
        m.setattr("retrofitkit.compliance.audit.DB", str(db_path))
        yield db_path

def test_signed_audit_log(audit_db):
    user_key = generate_test_keys()
    audit = CompliantAuditTrail()
    
    # Log signed event
    audit.log_event_signed("TEST_EVENT", {"foo": "bar"}, "user1", user_key)
    
    # Verify in DB
    con = sqlite3.connect(audit_db)
    row = con.execute("SELECT event, signature, public_key FROM audit WHERE event='TEST_EVENT'").fetchone()
    con.close()
    
    assert row is not None
    assert row[0] == "TEST_EVENT"
    assert row[1] is not None # Signature present
    assert row[2] is not None # Public key present

def test_chain_integrity(audit_db):
    user_key = generate_test_keys()
    audit = CompliantAuditTrail()
    
    audit.log_event_signed("EVENT_1", {}, "user1", user_key)
    audit.log_event_signed("EVENT_2", {}, "user1", user_key)
    
    con = sqlite3.connect(audit_db)
    rows = con.execute("SELECT hash, prev_hash FROM audit ORDER BY id ASC").fetchall()
    con.close()
    
    assert len(rows) == 2
    hash1, prev1 = rows[0]
    hash2, prev2 = rows[1]
    
    # Check chain: prev_hash of event 2 should match hash of event 1
    # Wait, implementation: prev = self._last_hash()
    # So yes, prev2 should be hash1.
    assert prev2 == hash1
