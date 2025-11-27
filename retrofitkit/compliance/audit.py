"""
Production-ready audit logging with unified database layer.

Hash-chain audit trail with support for electronic signatures.
"""

import time
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from retrofitkit.db.models.audit import AuditEvent


def write_audit_event(
    db: Session,
    actor_id: str,
    event_type: str,
    entity_type: str,
    entity_id: str,
    payload: Dict[str, Any]
) -> AuditEvent:
    """
    Write an audit event with cryptographic hash chain.
    
    Args:
        db: Database session
        actor_id: Who performed the action (user email)
        event_type: Type of event (e.g., "LOGIN_SUCCESS", "SAMPLE_CREATED")
        entity_type: What was affected (e.g., "user", "sample", "workflow")
        entity_id: ID of the affected entity
        payload: Additional event data
        
    Returns:
        Created AuditEvent
    """
    ts = time.time()
    
    # Get previous hash for chain-of-custody
    prev_entry = db.query(AuditEvent).order_by(AuditEvent.id.desc()).first()
    prev_hash = prev_entry.hash if prev_entry else "GENESIS"
    
    # Create payload string
    details = json.dumps(payload, sort_keys=True)
    
    # Compute current hash
    data = f"{ts}{event_type}{actor_id}{entity_type}{entity_id}{details}{prev_hash}"
    current_hash = hashlib.sha256(data.encode()).hexdigest()
    
    # Create new entry
    entry = AuditEvent(
        ts=ts,
        event=event_type,
        actor=actor_id,
        subject=f"{entity_type}:{entity_id}",
        details=details,
        prev_hash=prev_hash,
        hash=current_hash
    )
    
    db.add(entry)
    db.commit()
    db.refresh(entry)
    
    return entry


def get_audit_logs(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    event_type: Optional[str] = None,
    actor: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve audit logs with optional filtering.
    
    Args:
        db: Database session
        limit: Maximum number of entries to return
        offset: Number of entries to skip
        event_type: Filter by event type
        actor: Filter by actor
        
    Returns:
        List of audit log entries as dictionaries
    """
    query = db.query(AuditEvent).order_by(AuditEvent.id.desc())
    
    if event_type:
        query = query.filter(AuditEvent.event == event_type)
    if actor:
        query = query.filter(AuditEvent.actor == actor)
    
    entries = query.limit(limit).offset(offset).all()
    
    return [
        {
            "id": e.id,
            "ts": e.ts,
            "event": e.event,
            "actor": e.actor,
            "subject": e.subject,
            "details": e.details,
            "hash": e.hash,
            "prev_hash": e.prev_hash
        }
        for e in entries
    ]


def verify_audit_chain(db: Session, start_id: Optional[int] = None, end_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Verify the integrity of the audit trail hash chain.
    
    Args:
        db: Database session
        start_id: Optional starting audit event ID
        end_id: Optional ending audit event ID
        
    Returns:
        Dict with verification results
    """
    query = db.query(AuditEvent).order_by(AuditEvent.id.asc())
    
    if start_id:
        query = query.filter(AuditEvent.id >= start_id)
    if end_id:
        query = query.filter(AuditEvent.id <= end_id)
    
    entries = query.all()
    
    if not entries:
        return {"valid": True, "message": "No entries to verify"}
    
    errors = []
    prev_hash = "GENESIS"
    
    for entry in entries:
        # Recompute hash
        data = f"{entry.ts}{entry.event}{entry.actor}{entry.subject}{entry.details}{prev_hash}"
        expected_hash = hashlib.sha256(data.encode()).hexdigest()
        
        # Check prev_hash matches
        if entry.prev_hash != prev_hash:
            errors.append({
                "id": entry.id,
                "error": "prev_hash_mismatch",
                "expected": prev_hash,
                "actual": entry.prev_hash
            })
        
        # Check hash matches
        if entry.hash != expected_hash:
            errors.append({
                "id": entry.id,
                "error": "hash_mismatch",
                "expected": expected_hash,
                "actual": entry.hash
            })
        
        prev_hash = entry.hash
    
    return {
        "valid": len(errors) == 0,
        "entries_checked": len(entries),
        "errors": errors
    }


# Legacy class wrapper for backwards compatibility
class Audit:
    """Legacy wrapper for backwards compatibility with existing code."""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db
    
    def log(self, event: str, actor: str, subject: str, details: str = "") -> int:
        """Log an audit event - legacy interface."""
        if self.db:
            try:
                payload = json.loads(details) if details else {}
            except:
                payload = {"details": details}
            
            entry = write_audit_event(
                db=self.db,
                actor_id=actor,
                event_type=event,
                entity_type="legacy",
                entity_id=subject,
                payload=payload
            )
            return entry.id
        return 0
    
    def record(self, event: str, actor: str, subject: str, payload: Dict[str, Any]):
        """Record an audit event - legacy interface."""
        if self.db:
            write_audit_event(
                db=self.db,
                actor_id=actor,
                event_type=event,
                entity_type="legacy",
                entity_id=subject,
                payload=payload
            )
    
    def get_logs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get audit logs - legacy interface."""
        if self.db:
            return get_audit_logs(self.db, limit, offset)
        return []
