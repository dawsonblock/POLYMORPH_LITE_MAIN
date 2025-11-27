"""
Audit logging model with cryptographic chain of custody.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Text, LargeBinary, Index
from retrofitkit.db.base import Base


class AuditEvent(Base):
    """Audit log entries with cryptographic chain of custody."""
    __tablename__ = 'audit'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(Float, nullable=False, index=True)
    event = Column(String(255), nullable=False, index=True)
    actor = Column(String(255), nullable=False, index=True)
    subject = Column(String(255), nullable=False, index=True)
    details = Column(Text, nullable=True)
    prev_hash = Column(String(64), nullable=True)
    hash = Column(String(64), nullable=False, index=True)
    signature = Column(LargeBinary, nullable=True)
    public_key = Column(LargeBinary, nullable=True)
    ca_cert = Column(LargeBinary, nullable=True)
    meaning = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_audit_ts', 'ts'),
        Index('idx_audit_event', 'event'),
        Index('idx_audit_actor', 'actor'),
    )
