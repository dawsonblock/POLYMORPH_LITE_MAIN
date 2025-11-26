"""
Compliance API endpoints for audit logs and regulatory data.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from pydantic import BaseModel

from retrofitkit.compliance.audit import Audit

router = APIRouter(prefix="/compliance", tags=["compliance"])

# Initialize audit system
_audit = Audit()

class AuditLogEntry(BaseModel):
    id: int
    ts: float
    event: str
    actor: str
    subject: str
    details: str
    hash: str

@router.get("/audit", response_model=List[AuditLogEntry])
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Retrieve audit logs.
    
    Args:
        limit: Max number of entries
        offset: Pagination offset
        
    Returns:
        List of audit log entries
    """
    logs = _audit.get_logs(limit=limit, offset=offset)
    
    # Ensure details is string (it might be None in DB)
    for log in logs:
        if log["details"] is None:
            log["details"] = ""
            
    return logs
