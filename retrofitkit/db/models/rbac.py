"""
Role-Based Access Control (RBAC) models.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


class Role(Base):
    """Role definitions with permissions."""
    __tablename__ = 'roles'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    permissions = Column(JSON, default=dict)  # {"resource": ["read", "write", "delete"]}

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user_roles = relationship("UserRole", back_populates="role", cascade="all, delete-orphan")


class UserRole(Base):
    """Many-to-many relationship between users and roles."""
    __tablename__ = 'user_roles'

    user_email = Column(String(255), ForeignKey('users.email'), primary_key=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey('roles.id'), primary_key=True)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    assigned_by = Column(String(255), nullable=True)

    # Relationships
    user = relationship("User", back_populates="user_roles")
    role = relationship("Role", back_populates="user_roles")
