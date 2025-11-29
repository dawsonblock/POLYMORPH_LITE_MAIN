"""
Database models for Polymorph Discovery v1.0.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from retrofitkit.db.models.base import Base


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
    metadata = Column(JSON, nullable=True)
    
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
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    event = relationship("PolymorphEvent", back_populates="reports")
    
    __table_args__ = (
        Index('idx_polymorph_reports_event', 'event_id'),
        Index('idx_polymorph_reports_generated_at', 'generated_at'),
    )
