"""
Workflow and run execution models.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


class WorkflowVersion(Base):
    """Versioned workflow definitions."""
    __tablename__ = 'workflow_versions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)  # Using string for flexibility (e.g., "1.0.0")

    # Workflow definition (JSON representation of workflow graph)
    definition = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)

    # Status
    is_active = Column(String(10), default='false', index=True)  # Using string for SQLite compatibility
    is_approved = Column(String(10), default='false')
    locked = Column(String(10), default='false')  # CFR-11 lock flag

    # Audit
    created_by = Column(String(255), ForeignKey('users.email'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    approved_by = Column(String(255), nullable=True)
    approved_at = Column(DateTime, nullable=True)

    # Hash for integrity verification
    definition_hash = Column(String(64), nullable=False)

    # Relationships
    workflow_executions = relationship("WorkflowExecution", back_populates="workflow_version")

    __table_args__ = (
        UniqueConstraint('workflow_name', 'version', name='uq_workflow_name_version'),
        Index('idx_workflow_active', 'workflow_name', 'is_active'),
    )


class WorkflowExecution(Base):
    """Record of workflow execution / run."""
    __tablename__ = 'workflow_executions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(255), unique=True, nullable=False, index=True)

    # Workflow reference
    workflow_version_id = Column(UUID(as_uuid=True), ForeignKey('workflow_versions.id'), nullable=False, index=True)

    # Execution details
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), default='created', index=True)  # created, running, completed, failed, aborted

    # Operator
    operator = Column(String(255), ForeignKey('users.email'), nullable=False)

    # Results
    results = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)

    # Configuration snapshot
    config_snapshot_id = Column(UUID(as_uuid=True), ForeignKey('config_snapshots.id'), nullable=True)

    # Metadata (sample_id, device_id, etc.)
    run_metadata = Column(JSON, default=dict)

    # Relationships
    workflow_version = relationship("WorkflowVersion", back_populates="workflow_executions")
    config_snapshot = relationship("ConfigSnapshot", backref="workflow_executions")
    sample_assignments = relationship("WorkflowSampleAssignment", back_populates="workflow_execution", cascade="all, delete-orphan")


class WorkflowSampleAssignment(Base):
    """Link between workflows and samples."""
    __tablename__ = 'workflow_sample_assignments'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_execution_id = Column(UUID(as_uuid=True), ForeignKey('workflow_executions.id'), nullable=False, index=True)
    sample_id = Column(UUID(as_uuid=True), ForeignKey('samples.id'), nullable=False, index=True)

    assigned_at = Column(DateTime, default=datetime.utcnow)
    assigned_by = Column(String(255), nullable=False)

    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="sample_assignments")
    sample = relationship("Sample", back_populates="workflow_assignments")


class ConfigSnapshot(Base):
    """Immutable configuration snapshots for compliance."""
    __tablename__ = 'config_snapshots'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_id = Column(String(255), unique=True, nullable=False, index=True)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    config_data = Column(JSON, nullable=False)
    config_hash = Column(String(64), nullable=False, index=True)

    created_by = Column(String(255), nullable=False)
    reason = Column(Text, nullable=True)
