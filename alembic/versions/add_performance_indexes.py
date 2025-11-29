"""
Database indexes for performance optimization.

Adds indexes to frequently queried fields to improve query performance.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_performance_indexes'
down_revision = None  # Update with actual previous revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add performance indexes."""
    
    # Samples table indexes
    op.create_index('idx_samples_org_id', 'samples', ['org_id'])
    op.create_index('idx_samples_project_id', 'samples', ['project_id'])
    op.create_index('idx_samples_created_at', 'samples', ['created_at'])
    op.create_index('idx_samples_sample_id', 'samples', ['sample_id'])
    op.create_index('idx_samples_org_created', 'samples', ['org_id', 'created_at'])
    
    # Workflow executions indexes
    op.create_index('idx_workflow_exec_org_id', 'workflow_executions', ['org_id'])
    op.create_index('idx_workflow_exec_status', 'workflow_executions', ['status'])
    op.create_index('idx_workflow_exec_created', 'workflow_executions', ['created_at'])
    op.create_index('idx_workflow_exec_org_status', 'workflow_executions', ['org_id', 'status'])
    
    # Audit log indexes
    op.create_index('idx_audit_org_id', 'audit_log', ['org_id'])
    op.create_index('idx_audit_event', 'audit_log', ['event'])
    op.create_index('idx_audit_timestamp', 'audit_log', ['timestamp'])
    op.create_index('idx_audit_actor', 'audit_log', ['actor'])
    op.create_index('idx_audit_org_timestamp', 'audit_log', ['org_id', 'timestamp'])
    
    # Users table indexes
    op.create_index('idx_users_email', 'users', ['email'], unique=True)
    op.create_index('idx_users_org_id', 'users', ['org_id'])
    
    # Projects table indexes
    op.create_index('idx_projects_org_id', 'projects', ['org_id'])
    op.create_index('idx_projects_status', 'projects', ['status'])
    
    # Device calibrations indexes
    op.create_index('idx_calibrations_device_id', 'device_calibrations', ['device_id'])
    op.create_index('idx_calibrations_date', 'device_calibrations', ['calibration_date'])


def downgrade() -> None:
    """Remove performance indexes."""
    
    # Samples
    op.drop_index('idx_samples_org_id')
    op.drop_index('idx_samples_project_id')
    op.drop_index('idx_samples_created_at')
    op.drop_index('idx_samples_sample_id')
    op.drop_index('idx_samples_org_created')
    
    # Workflow executions
    op.drop_index('idx_workflow_exec_org_id')
    op.drop_index('idx_workflow_exec_status')
    op.drop_index('idx_workflow_exec_created')
    op.drop_index('idx_workflow_exec_org_status')
    
    # Audit log
    op.drop_index('idx_audit_org_id')
    op.drop_index('idx_audit_event')
    op.drop_index('idx_audit_timestamp')
    op.drop_index('idx_audit_actor')
    op.drop_index('idx_audit_org_timestamp')
    
    # Users
    op.drop_index('idx_users_email')
    op.drop_index('idx_users_org_id')
    
    # Projects
    op.drop_index('idx_projects_org_id')
    op.drop_index('idx_projects_status')
    
    # Calibrations
    op.drop_index('idx_calibrations_device_id')
    op.drop_index('idx_calibrations_date')
