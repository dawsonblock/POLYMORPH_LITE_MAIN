"""
Temporary compatibility shim for old database.models imports.

This allows existing API code to keep working while we migrate.
All imports are redirected to the new retrofitkit/db/ layer.

DELETE THIS FILE once all API files are fully migrated.
"""

# Import everything from new DB layer
from retrofitkit.db.models.user import User
from retrofitkit.db.models.rbac import Role, UserRole
from retrofitkit.db.models.device import Device, DeviceStatus
from retrofitkit.db.models.sample import Project, Batch, Container, Sample, SampleLineage
from retrofitkit.db.models.inventory import Vendor, InventoryItem, StockLot
from retrofitkit.db.models.calibration import CalibrationEntry
from retrofitkit.db.models.workflow import WorkflowVersion, WorkflowExecution, WorkflowSampleAssignment, ConfigSnapshot
from retrofitkit.db.models.audit import AuditEvent
from retrofitkit.db.models.org import Organization, Lab, Node, DeviceHub
from retrofitkit.db.base import Base
from retrofitkit.db.session import SessionLocal, engine, get_db

# Legacy compatibility function
def get_session():
    """
    DEPRECATED: Use Depends(get_db) instead.
    
    This function exists for backward compatibility during migration.
    It returns a session that must be manually closed.
    """
    return SessionLocal()

# Export everything that old code expects
__all__ = [
    'User', 'Role', 'UserRole',
    'Device', 'DeviceStatus',
    'Project', 'Batch', 'Container', 'Sample', 'SampleLineage',
    'Vendor', 'InventoryItem', 'StockLot',
    'CalibrationEntry',
    'WorkflowVersion', 'WorkflowExecution', 'WorkflowSampleAssignment', 'ConfigSnapshot',
    'AuditEvent',
    'Organization', 'Lab', 'Node', 'DeviceHub',
    'Base', 'SessionLocal', 'engine', 'get_db', 'get_session'
]
