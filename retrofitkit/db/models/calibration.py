"""
Calibration tracking models for instruments.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Date, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CalibrationEntry(Base):
    """Calibration record for instruments."""
    __tablename__ = 'calibration_entries'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(255), nullable=False, index=True)

    # Calibration details
    calibration_date = Column(DateTime, nullable=False, index=True)
    performed_by = Column(String(255), ForeignKey('users.email'), nullable=False)
    next_due_date = Column(Date, nullable=True, index=True)

    # Status
    status = Column(String(50), default='valid', index=True)  # valid, expired, failed

    # Results and documentation
    results = Column(JSON, default=dict)  # Calibration results, tolerances, etc.
    certificate_path = Column(String(500), nullable=True)  # Path to PDF certificate
    notes = Column(String(1000), nullable=True)

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=utcnow)
