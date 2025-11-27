"""
User model for authentication and user management.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, LargeBinary, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


class User(Base):
    """User model with MFA and password management."""
    __tablename__ = 'users'

    email = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    role = Column(String(100), nullable=False)  # Legacy field, will be replaced by RBAC
    password_hash = Column(LargeBinary, nullable=False)
    mfa_secret = Column(String(255), nullable=True)

    # CFR 11 compliance fields
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    password_history = Column(JSON, default=list)  # List of previous password hashes

    # SSO fields (Phase 4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=True)
    lab_id = Column(UUID(as_uuid=True), ForeignKey('labs.id'), nullable=True)
    sso_provider = Column(String(100), nullable=True)
    sso_subject = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user_roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")
