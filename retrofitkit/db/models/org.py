"""
Organization and multi-site models for enterprise deployments.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Organization(Base):
    """Organization/tenant for multi-site deployment."""
    __tablename__ = 'organizations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    subscription_tier = Column(String(50), nullable=True)  # free, professional, enterprise

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    labs = relationship("Lab", back_populates="organization", cascade="all, delete-orphan")


class Lab(Base):
    """Laboratory/site within an organization."""
    __tablename__ = 'labs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lab_id = Column(String(255), unique=True, nullable=False, index=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    timezone = Column(String(50), default='UTC')

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="labs")
    nodes = relationship("Node", back_populates="lab", cascade="all, delete-orphan")


class Node(Base):
    """Compute node in a lab (edge device running workflows)."""
    __tablename__ = 'nodes'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(255), unique=True, nullable=False, index=True)
    lab_id = Column(UUID(as_uuid=True), ForeignKey('labs.id'), nullable=False, index=True)

    hostname = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6

    status = Column(String(50), default='offline', index=True)  # online, offline, maintenance
    last_heartbeat = Column(DateTime, nullable=True, index=True)

    capabilities = Column(JSON, default=dict)  # Available devices, resources

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    lab = relationship("Lab", back_populates="nodes")
    device_hubs = relationship("DeviceHub", back_populates="node", cascade="all, delete-orphan")


class DeviceHub(Base):
    """Device registry for a specific node."""
    __tablename__ = 'device_hubs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hub_id = Column(String(255), unique=True, nullable=False, index=True)
    node_id = Column(UUID(as_uuid=True), ForeignKey('nodes.id'), nullable=False, index=True)

    device_registry = Column(JSON, nullable=False)  # List of available devices
    health_status = Column(JSON, default=dict)  # Health metrics for each device

    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    node = relationship("Node", back_populates="device_hubs")
