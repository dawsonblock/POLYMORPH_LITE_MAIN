"""
Device models for hardware tracking and status.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Date, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Device(Base):
    """Device/instrument definition."""
    __tablename__ = 'devices'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    vendor = Column(String(255), nullable=True)
    model = Column(String(255), nullable=True)
    serial_number = Column(String(255), nullable=True)
    device_type = Column(String(100), nullable=True, index=True)  # daq, raman, etc.
    location = Column(String(255), nullable=True)
    is_active = Column(String(10), default='true')  # Using string for SQLite compatibility

    extra_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class DeviceStatus(Base):
    """Current status of each device."""
    __tablename__ = 'device_status'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(255), unique=True, nullable=False, index=True)

    # Status
    status = Column(String(50), default='operational', index=True)  # operational, maintenance, offline, error

    # Calibration tracking
    last_calibration_date = Column(Date, nullable=True, index=True)
    next_calibration_due = Column(Date, nullable=True, index=True)

    # Health metrics
    health_score = Column(Float, nullable=True)  # 0.0 - 1.0

    extra_data = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
