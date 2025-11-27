"""
SQLAlchemy models for POLYMORPH-LITE.

Organized by domain:
- User management (user, rbac)
- Audit logging (audit)
- Device tracking (device)  
- Sample management (sample)
- Inventory tracking (inventory)
- Calibration (calibration)
- Workflow execution (workflow)
- Multi-site support (org)
"""

from retrofitkit.db.base import Base

# Import all models to register them with Base
from retrofitkit.db.models.user import User
from retrofitkit.db.models.rbac import Role, UserRole
from retrofitkit.db.models.audit import AuditEvent
from retrofitkit.db.models.device import Device, DeviceStatus
from retrofitkit.db.models.sample import Project, Container, Batch, Sample, SampleLineage
from retrofitkit.db.models.inventory import Vendor, InventoryItem, StockLot
from retrofitkit.db.models.calibration import CalibrationEntry
from retrofitkit.db.models.workflow import WorkflowVersion, WorkflowExecution, WorkflowSampleAssignment, ConfigSnapshot
from retrofitkit.db.models.org import Organization, Lab, Node, DeviceHub

__all__ = [
    "Base",
    # User management
    "User",
    "Role",
    "UserRole",
    # Audit
    "AuditEvent",
    # Devices
    "Device",
    "DeviceStatus",
    # Samples
    "Project",
    "Container",
    "Batch",
    "Sample",
    "SampleLineage",
    # Inventory
    "Vendor",
    "InventoryItem",
    "StockLot",
    # Calibration
    "CalibrationEntry",
    # Workflows
    "WorkflowVersion",
    "WorkflowExecution",
    "WorkflowSampleAssignment",
    "ConfigSnapshot",
    # Organization
    "Organization",
    "Lab",
    "Node",
    "DeviceHub",
]
