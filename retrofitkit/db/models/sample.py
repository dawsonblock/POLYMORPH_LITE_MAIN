"""
Sample management models for LIMS functionality.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    """Project/Study container for samples and batches."""
    __tablename__ = 'projects'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default='active', index=True)  # active, completed, archived
    owner = Column(String(255), ForeignKey('users.email'), nullable=True)

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    created_by = Column(String(255), nullable=False)

    # Relationships
    samples = relationship("Sample", back_populates="project", cascade="all, delete-orphan")
    batches = relationship("Batch", back_populates="project", cascade="all, delete-orphan")


class Container(Base):
    """Physical container for sample storage."""
    __tablename__ = 'containers'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    container_id = Column(String(255), unique=True, nullable=False, index=True)
    container_type = Column(String(100), nullable=True)  # vial, plate, box, etc.
    location = Column(String(255), nullable=True, index=True)
    capacity = Column(String(50), nullable=True)  # Using string for flexibility
    current_count = Column(String(50), default='0')

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    samples = relationship("Sample", back_populates="container")


class Batch(Base):
    """Batch container for grouping samples."""
    __tablename__ = 'batches'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(String(255), unique=True, nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=True, index=True)

    status = Column(String(50), default='active', index=True)  # active, released, rejected

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    created_by = Column(String(255), nullable=False)

    # Relationships
    project = relationship("Project", back_populates="batches")
    samples = relationship("Sample", back_populates="batch")


class Sample(Base):
    """Sample/specimen with lineage tracking."""
    __tablename__ = 'samples'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sample_id = Column(String(255), unique=True, nullable=False, index=True)
    lot_number = Column(String(255), nullable=True, index=True)

    # Foreign keys
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=True, index=True)
    container_id = Column(UUID(as_uuid=True), ForeignKey('containers.id'), nullable=True, index=True)
    parent_sample_id = Column(UUID(as_uuid=True), ForeignKey('samples.id'), nullable=True, index=True)
    batch_id = Column(UUID(as_uuid=True), ForeignKey('batches.id'), nullable=True, index=True)
    org_id = Column(String(255), ForeignKey('organizations.org_id'), nullable=True, index=True)

    # Status tracking
    status = Column(String(50), default='active', index=True)  # active, consumed, disposed, quarantined

    # Metadata
    extra_data = Column(JSON, default=dict)

    # Audit fields
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    updated_by = Column(String(255), nullable=True)

    # Relationships
    project = relationship("Project", back_populates="samples")
    container = relationship("Container", back_populates="samples")
    batch = relationship("Batch", back_populates="samples")
    parent_sample = relationship("Sample", remote_side=[id], backref="child_samples")
    lineage_as_parent = relationship("SampleLineage", foreign_keys="SampleLineage.parent_sample_id", back_populates="parent_sample")
    lineage_as_child = relationship("SampleLineage", foreign_keys="SampleLineage.child_sample_id", back_populates="child_sample")
    workflow_assignments = relationship("WorkflowSampleAssignment", back_populates="sample", cascade="all, delete-orphan")


class SampleLineage(Base):
    """Explicit lineage tracking between samples."""
    __tablename__ = 'sample_lineage'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_sample_id = Column(UUID(as_uuid=True), ForeignKey('samples.id'), nullable=False, index=True)
    child_sample_id = Column(UUID(as_uuid=True), ForeignKey('samples.id'), nullable=False, index=True)
    relationship_type = Column(String(100), nullable=True)  # split, aliquot, derivative, etc.

    extra_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)
    created_by = Column(String(255), nullable=False)

    # Relationships
    parent_sample = relationship("Sample", foreign_keys=[parent_sample_id], back_populates="lineage_as_parent")
    child_sample = relationship("Sample", foreign_keys=[child_sample_id], back_populates="lineage_as_child")

    __table_args__ = (
        Index('idx_sample_lineage_parent', 'parent_sample_id'),
        Index('idx_sample_lineage_child', 'child_sample_id'),
    )
