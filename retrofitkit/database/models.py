"""
SQLAlchemy ORM models for POLYMORPH-LITE database.

Includes:
- Sample tracking and lineage
- Inventory management
- Calibration logs
- Enhanced RBAC
- Workflow versioning
"""
import os
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, Boolean,
    Date, DateTime, ForeignKey, LargeBinary, JSON, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://polymorph:polymorph_pass@localhost:5432/polymorph_db"
)

# Fallback to SQLite for local development
if "postgresql" not in DATABASE_URL and "sqlite" not in DATABASE_URL:
    DB_DIR = os.environ.get("P4_DATA_DIR", "data")
    os.makedirs(DB_DIR, exist_ok=True)
    DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'polymorph.db')}"

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


# ============================================================================
# CORE USER MANAGEMENT
# ============================================================================

class User(Base):
    """User model with MFA and password management."""
    __tablename__ = 'users'

    email = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    role = Column(String(100), nullable=False)  # Will be FK to roles table later
    password_hash = Column(LargeBinary, nullable=False)
    mfa_secret = Column(String(255), nullable=True)

    # CFR 11 compliance fields
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, default=datetime.utcnow)
    password_history = Column(JSON, default=list)  # List of previous password hashes

    # SSO fields (Phase 4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=True)
    lab_id = Column(UUID(as_uuid=True), ForeignKey('labs.id'), nullable=True)
    sso_provider = Column(String(100), nullable=True)
    sso_subject = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user_roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")


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


# ============================================================================
# AUDIT LOGGING (existing, included for completeness)
# ============================================================================

class AuditLog(Base):
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


# ============================================================================
# SAMPLE MANAGEMENT
# ============================================================================

class Project(Base):
    """Project/Study container for samples and batches."""
    __tablename__ = 'projects'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default='active', index=True)  # active, completed, archived
    owner = Column(String(255), ForeignKey('users.email'), nullable=True)

    extra_data = Column(JSON, default=dict)  # Custom metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
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
    location = Column(String(255), nullable=True, index=True)  # Physical location
    capacity = Column(Integer, nullable=True)
    current_count = Column(Integer, default=0)

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    samples = relationship("Sample", back_populates="container")


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

    # Status tracking
    status = Column(String(50), default='active', index=True)  # active, consumed, disposed, quarantined

    # Metadata
    extra_data = Column(JSON, default=dict)  # Flexible storage for custom properties

    # Audit fields
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(255), nullable=True)

    # Relationships
    project = relationship("Project", back_populates="samples")
    container = relationship("Container", back_populates="samples")
    batch = relationship("Batch", back_populates="samples")

    # Self-referential relationship for lineage
    parent_sample = relationship("Sample", remote_side=[id], backref="child_samples")

    # Lineage tracking
    lineage_as_parent = relationship("SampleLineage", foreign_keys="SampleLineage.parent_sample_id", back_populates="parent_sample")
    lineage_as_child = relationship("SampleLineage", foreign_keys="SampleLineage.child_sample_id", back_populates="child_sample")

    # Workflow assignments
    workflow_assignments = relationship("WorkflowSampleAssignment", back_populates="sample", cascade="all, delete-orphan")


class SampleLineage(Base):
    """Explicit lineage tracking between samples."""
    __tablename__ = 'sample_lineage'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_sample_id = Column(UUID(as_uuid=True), ForeignKey('samples.id'), nullable=False, index=True)
    child_sample_id = Column(UUID(as_uuid=True), ForeignKey('samples.id'), nullable=False, index=True)
    relationship_type = Column(String(100), nullable=True)  # split, aliquot, derivative, etc.

    extra_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=False)

    # Relationships
    parent_sample = relationship("Sample", foreign_keys=[parent_sample_id], back_populates="lineage_as_parent")
    child_sample = relationship("Sample", foreign_keys=[child_sample_id], back_populates="lineage_as_child")

    __table_args__ = (
        Index('idx_sample_lineage_parent', 'parent_sample_id'),
        Index('idx_sample_lineage_child', 'child_sample_id'),
    )


class Batch(Base):
    """Batch container for grouping samples."""
    __tablename__ = 'batches'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(String(255), unique=True, nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=True, index=True)

    status = Column(String(50), default='active', index=True)  # active, released, rejected

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=False)

    # Relationships
    project = relationship("Project", back_populates="batches")
    samples = relationship("Sample", back_populates="batch")


# ============================================================================
# INVENTORY MANAGEMENT
# ============================================================================

class Vendor(Base):
    """Vendor/supplier information."""
    __tablename__ = 'vendors'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vendor_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    contact_info = Column(JSON, default=dict)  # email, phone, address, etc.

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    stock_lots = relationship("StockLot", back_populates="vendor")


class InventoryItem(Base):
    """Inventory item master data."""
    __tablename__ = 'inventory_items'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_code = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=True, index=True)  # reagent, consumable, standard
    unit = Column(String(50), nullable=True)  # mL, g, ea, etc.

    # Stock management
    min_stock = Column(Integer, default=0)
    current_stock = Column(Integer, default=0, index=True)
    reorder_point = Column(Integer, default=0)

    # Location
    location = Column(String(255), nullable=True, index=True)

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=False)

    # Relationships
    stock_lots = relationship("StockLot", back_populates="item", cascade="all, delete-orphan")


class StockLot(Base):
    """Individual stock lot with expiration tracking."""
    __tablename__ = 'stock_lots'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lot_number = Column(String(255), unique=True, nullable=False, index=True)

    # Foreign keys
    item_id = Column(UUID(as_uuid=True), ForeignKey('inventory_items.id'), nullable=False, index=True)
    vendor_id = Column(UUID(as_uuid=True), ForeignKey('vendors.id'), nullable=True, index=True)

    # Quantities
    quantity = Column(Integer, nullable=False)
    quantity_remaining = Column(Integer, nullable=False)

    # Dates
    received_date = Column(Date, default=datetime.utcnow, index=True)
    expiration_date = Column(Date, nullable=True, index=True)

    # Status
    status = Column(String(50), default='active', index=True)  # active, depleted, expired, quarantined

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    item = relationship("InventoryItem", back_populates="stock_lots")
    vendor = relationship("Vendor", back_populates="stock_lots")


# ============================================================================
# INSTRUMENT CALIBRATION
# ============================================================================

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

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships - view-only relationship to device status
    # Note: This is view-only as device_id is not a proper FK (DeviceStatus might not exist yet)


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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# WORKFLOW MANAGEMENT
# ============================================================================

class WorkflowVersion(Base):
    """Versioned workflow definitions."""
    __tablename__ = 'workflow_versions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_name = Column(String(255), nullable=False, index=True)
    version = Column(Integer, nullable=False)

    # Workflow definition (JSON representation of workflow graph)
    definition = Column(JSON, nullable=False)

    # Status
    is_active = Column(Boolean, default=False, index=True)
    is_approved = Column(Boolean, default=False)

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
    """Record of workflow execution."""
    __tablename__ = 'workflow_executions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(255), unique=True, nullable=False, index=True)

    # Workflow reference
    workflow_version_id = Column(UUID(as_uuid=True), ForeignKey('workflow_versions.id'), nullable=False, index=True)

    # Execution details
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), default='running', index=True)  # running, completed, failed, aborted

    # Operator
    operator = Column(String(255), ForeignKey('users.email'), nullable=False)

    # Results
    results = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)

    # Configuration snapshot
    config_snapshot_id = Column(UUID(as_uuid=True), ForeignKey('config_snapshots.id'), nullable=True)

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


# ============================================================================
# MULTI-SITE SUPPORT (Phase 4)
# ============================================================================

class Organization(Base):
    """Organization/tenant for multi-site deployment."""
    __tablename__ = 'organizations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    subscription_tier = Column(String(50), nullable=True)  # free, professional, enterprise

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    node = relationship("Node", back_populates="device_hubs")


# ============================================================================
# Create all tables
# ============================================================================

def init_database():
    """Initialize database tables."""
    Base.metadata.create_all(engine)
    print(f"Database initialized at: {DATABASE_URL}")


def get_session():
    """Get database session."""
    return SessionLocal()


if __name__ == "__main__":
    init_database()
