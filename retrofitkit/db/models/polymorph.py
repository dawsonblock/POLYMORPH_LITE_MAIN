"""
Database models for Polymorph Discovery v1.0.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, Text, ForeignKey, Index, LargeBinary, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid

from retrofitkit.db.base import Base


class PolymorphMode(Base):
    """
    Active polymorph mode from PMM.
    Stores the learned mode prototype and associated statistics.
    """
    __tablename__ = "polymorph_modes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    device_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Mode identification
    mode_index = Column(Integer, nullable=False)  # Index in PMM
    poly_id_hash = Column(String(64), nullable=False, index=True)  # Stable hash
    poly_name = Column(String(100), nullable=True)  # Human-readable name
    
    # Mode vector (stored as JSON for portability)
    mu = Column(JSON, nullable=False)  # Latent vector
    F = Column(JSON, nullable=True)  # Predictive matrix (optional)
    
    # Statistics
    occupancy = Column(Float, default=0.0)
    risk = Column(Float, default=0.0)
    age = Column(Integer, default=0)
    lambda_i = Column(Float, default=1.0)  # Importance weight
    
    # Status
    is_active = Column(Boolean, default=True)
    first_seen = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Metadata
    metadata_ = Column("metadata", JSON, nullable=True)
    
    # Relationships
    snapshots = relationship("PolymorphModeSnapshot", back_populates="mode", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_polymorph_modes_org', 'org_id'),
        Index('idx_polymorph_modes_poly_hash', 'poly_id_hash'),
        Index('idx_polymorph_modes_active', 'is_active'),
    )


class PolymorphModeSnapshot(Base):
    """
    Snapshot of a polymorph mode at a point in time.
    Used for tracking mode evolution and workflow checkpointing.
    """
    __tablename__ = "polymorph_mode_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mode_id = Column(UUID(as_uuid=True), ForeignKey('polymorph_modes.id', ondelete='CASCADE'), nullable=False)
    
    # Snapshot data
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    state_blob = Column(LargeBinary, nullable=True)  # Compressed PMM state
    
    # Snapshot context
    workflow_execution_id = Column(String(36), nullable=True, index=True)
    trigger = Column(String(50), nullable=True)  # 'workflow_start', 'workflow_end', 'manual', etc.
    
    # Stats at snapshot time
    occupancy = Column(Float, nullable=True)
    risk = Column(Float, nullable=True)
    age = Column(Integer, nullable=True)
    
    # Metadata
    snapshot_metadata = Column(JSON, nullable=True)
    
    # Relationships
    mode = relationship("PolymorphMode", back_populates="snapshots")
    
    __table_args__ = (
        Index('idx_mode_snapshots_mode', 'mode_id'),
        Index('idx_mode_snapshots_timestamp', 'timestamp'),
        Index('idx_mode_snapshots_workflow', 'workflow_execution_id'),
    )


class PolymorphEvent(Base):
    """Polymorph detection event."""
    __tablename__ = "polymorph_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False)
    detected_at = Column(Float, nullable=False)
    polymorph_id = Column(Integer, nullable=False)
    polymorph_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(20), nullable=False)
    workflow_execution_id = Column(String(36), nullable=True)
    sample_id = Column(String(36), nullable=True)
    operator_email = Column(String(255), nullable=False)
    event_metadata = Column(JSON, nullable=True)
    
    # Relationships
    signatures = relationship("PolymorphSignature", back_populates="event", cascade="all, delete-orphan")
    reports = relationship("PolymorphReport", back_populates="event", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_polymorph_events_detected_at', 'detected_at'),
        Index('idx_polymorph_events_polymorph_id', 'polymorph_id'),
        Index('idx_polymorph_events_workflow', 'workflow_execution_id'),
    )


class PolymorphSignature(Base):
    """Polymorph signature vector and features."""
    __tablename__ = "polymorph_signatures"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signature_id = Column(String(36), unique=True, nullable=False)
    event_id = Column(String(36), ForeignKey('polymorph_events.event_id', ondelete='CASCADE'), nullable=False)
    polymorph_id = Column(Integer, nullable=False)
    signature_vector = Column(JSON, nullable=False)
    alternative_forms = Column(JSON, nullable=True)
    spectral_features = Column(JSON, nullable=True)
    created_at = Column(Float, nullable=False)
    
    # Relationships
    event = relationship("PolymorphEvent", back_populates="signatures")
    
    __table_args__ = (
        Index('idx_polymorph_signatures_event', 'event_id'),
        Index('idx_polymorph_signatures_polymorph', 'polymorph_id'),
    )


class PolymorphReport(Base):
    """Generated polymorph reports."""
    __tablename__ = "polymorph_reports"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(36), unique=True, nullable=False)
    event_id = Column(String(36), ForeignKey('polymorph_events.event_id', ondelete='CASCADE'), nullable=False)
    report_format = Column(String(10), nullable=False)  # 'json' or 'pdf'
    report_data = Column(Text, nullable=False)
    generated_at = Column(Float, nullable=False)
    generated_by = Column(String(255), nullable=False)
    signed = Column(Boolean, default=False)
    signature_hash = Column(String(64), nullable=True)
    report_metadata = Column(JSON, nullable=True)
    
    # Relationships
    event = relationship("PolymorphEvent", back_populates="reports")
    
    __table_args__ = (
        Index('idx_polymorph_reports_event', 'event_id'),
        Index('idx_polymorph_reports_generated_at', 'generated_at'),
    )

