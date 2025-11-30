import time
import hashlib
import json
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from retrofitkit.db.models.audit import AuditEvent as AuditLog

async def write_audit_event(
    db: AsyncSession,
    actor_id: str,
    event_type: str,
    entity_type: str,
    entity_id: str,
    payload: Dict[str, Any]
) -> AuditLog:
    """
    Write an audit event with cryptographic hash chain.
    """
    ts = time.time()

    # Get previous hash for chain-of-custody
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(1)
    result = await db.execute(stmt)
    prev_entry = result.scalar_one_or_none()
    prev_hash = prev_entry.hash if prev_entry else "GENESIS"

    # Create payload string
    details = json.dumps(payload, sort_keys=True)
    subject = f"{entity_type}:{entity_id}"

    # Compute current hash
    data = f"{ts}{event_type}{actor_id}{subject}{details}{prev_hash}"
    current_hash = hashlib.sha256(data.encode()).hexdigest()

    # Create new entry
    entry = AuditLog(
        ts=ts,
        event=event_type,
        actor=actor_id,
        subject=subject,
        details=details,
        prev_hash=prev_hash,
        hash=current_hash
    )

    db.add(entry)
    await db.commit()
    await db.refresh(entry)

    return entry


async def get_audit_logs(
    db: AsyncSession,
    limit: int = 100,
    offset: int = 0,
    event_type: Optional[str] = None,
    actor: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve audit logs with optional filtering.
    """
    stmt = select(AuditLog).order_by(AuditLog.id.desc())

    if event_type:
        stmt = stmt.where(AuditLog.event == event_type)
    if actor:
        stmt = stmt.where(AuditLog.actor == actor)

    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    entries = result.scalars().all()

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


async def verify_audit_chain(db: AsyncSession) -> Dict[str, Any]:
    """
    Verify the integrity of the audit log hash chain.
    Returns a dict with verification results.
    """
    stmt = select(AuditLog).order_by(AuditLog.id.asc())
    result = await db.execute(stmt)
    entries = result.scalars().all()
    
    if not entries:
        return {"valid": True, "total_entries": 0, "message": "No audit entries to verify"}
    
    prev_hash = "GENESIS"
    invalid_entries = []
    
    for entry in entries:
        # Recompute hash
        data = f"{entry.ts}{entry.event}{entry.actor}{entry.subject}{entry.details}{prev_hash}"
        expected_hash = hashlib.sha256(data.encode()).hexdigest()
        
        # Check if hash matches
        if entry.hash != expected_hash:
            invalid_entries.append({
                "id": entry.id,
                "expected_hash": expected_hash,
                "actual_hash": entry.hash
            })
        
        # Check if prev_hash matches
        if entry.prev_hash != prev_hash:
            invalid_entries.append({
                "id": entry.id,
                "expected_prev_hash": prev_hash,
                "actual_prev_hash": entry.prev_hash
            })
        
        prev_hash = entry.hash
    
    return {
        "valid": len(invalid_entries) == 0,
        "total_entries": len(entries),
        "invalid_entries": invalid_entries,
        "message": "Audit chain is valid" if len(invalid_entries) == 0 else f"Found {len(invalid_entries)} invalid entries"
    }


# Legacy class wrapper updated for async
class Audit:
    """Legacy wrapper for backwards compatibility with existing code."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def log(self, event: str, actor: str, subject: str, details: str = "") -> int:
        """Log an audit event."""
        try:
            payload = json.loads(details) if details else {}
        except:
            payload = {"details": details}

        entry = await write_audit_event(
            db=self.db,
            actor_id=actor,
            event_type=event,
            entity_type="legacy",
            entity_id=subject,
            payload=payload
        )
        return entry.id

    async def record(self, event: str, actor: str, subject: str, payload: Dict[str, Any]):
        """Record an audit event."""
        await write_audit_event(
            db=self.db,
            actor_id=actor,
            event_type=event,
            entity_type="legacy",
            entity_id=subject,
            payload=payload
        )

    async def get_logs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get audit logs."""
        return await get_audit_logs(self.db, limit, offset)

