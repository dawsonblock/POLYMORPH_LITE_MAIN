"""
Workflow checkpoint model for resume capability.

Enables workflows to be paused and resumed from the last completed step.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, JSON, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


def utcnow() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class WorkflowCheckpoint(Base):
    """
    Workflow execution checkpoint for resume capability.
    
    Stores the state of a workflow execution at a specific step,
    allowing workflows to be resumed after interruption.
    """
    __tablename__ = "workflow_checkpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id = Column(UUID(as_uuid=True), ForeignKey("workflow_executions.id"), nullable=False)
    
    # Step information
    step_index = Column(Integer, nullable=False)
    step_type = Column(String(50), nullable=False)
    
    # State snapshot
    step_results = Column(JSON, nullable=False)  # All results up to this point
    device_state = Column(JSON, nullable=True)   # Device state if capturable
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    # Resume safety - ensures workflow hasn't changed
    workflow_definition_hash = Column(String(64), nullable=False)
    
    __table_args__ = (
        Index('idx_checkpoints_execution', 'execution_id'),
        Index('idx_checkpoints_step', 'step_index'),
        Index('idx_checkpoints_created', 'created_at'),
    )
