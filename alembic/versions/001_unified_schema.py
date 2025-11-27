"""Initial unified schema - all 27 tables

Revision ID: 001_unified_schema
Revises: 
Create Date: 2025-11-27 01:38:00.000000

This migration creates all tables for the unified POLYMORPH-LITE database:
- Authentication & RBAC (users, roles, user_roles)
- Audit trail (audit)
- Device management (devices, device_status)
- LIMS (projects, containers, batches, samples, sample_lineage)
- Inventory (vendors, inventory_items, stock_lots)
- Calibration (calibration_entries)
- Workflows (workflow_versions, workflow_executions, workflow_sample_assignments, config_snapshots)
- Multi-site (organizations, labs, nodes, device_hubs)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_unified_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables for unified POLYMORPH-LITE schema."""
    
    # ========================================================================
    # ORGANIZATIONS & MULTI-SITE
    # ========================================================================
    
    op.create_table('organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('org_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('subscription_tier', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('org_id')
    )
    op.create_index('ix_organizations_org_id', 'organizations', ['org_id'])
    
    op.create_table('labs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('lab_id', sa.String(length=255), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('timezone', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('lab_id')
    )
    op.create_index('ix_labs_lab_id', 'labs', ['lab_id'])
    op.create_index('ix_labs_organization_id', 'labs', ['organization_id'])
    
    # ========================================================================
    # AUTHENTICATION & RBAC
    # ========================================================================
    
    op.create_table('users',
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=100), nullable=False),
        sa.Column('password_hash', sa.LargeBinary(), nullable=False),
        sa.Column('mfa_secret', sa.String(length=255), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=True),
        sa.Column('account_locked_until', sa.DateTime(), nullable=True),
        sa.Column('password_changed_at', sa.DateTime(), nullable=True),
        sa.Column('password_history', sa.JSON(), nullable=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('lab_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('sso_provider', sa.String(length=100), nullable=True),
        sa.Column('sso_subject', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lab_id'], ['labs.id'], ),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.PrimaryKeyConstraint('email')
    )
    
    op.create_table('roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role_name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('permissions', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('role_name')
    )
    
    op.create_table('user_roles',
        sa.Column('user_email', sa.String(length=255), nullable=False),
        sa.Column('role_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('assigned_at', sa.DateTime(), nullable=True),
        sa.Column('assigned_by', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ),
        sa.ForeignKeyConstraint(['user_email'], ['users.email'], ),
        sa.PrimaryKeyConstraint('user_email', 'role_id')
    )
    
    # ========================================================================
    # AUDIT TRAIL
    # ========================================================================
    
    op.create_table('audit',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('ts', sa.Float(), nullable=False),
        sa.Column('event', sa.String(length=255), nullable=False),
        sa.Column('actor', sa.String(length=255), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=False),
        sa.Column('details', sa.Text(), nullable=True),
        sa.Column('prev_hash', sa.String(length=64), nullable=True),
        sa.Column('hash', sa.String(length=64), nullable=False),
        sa.Column('signature', sa.LargeBinary(), nullable=True),
        sa.Column('public_key', sa.LargeBinary(), nullable=True),
        sa.Column('ca_cert', sa.LargeBinary(), nullable=True),
        sa.Column('meaning', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_actor', 'audit', ['actor'])
    op.create_index('ix_audit_event', 'audit', ['event'])
    op.create_index('ix_audit_hash', 'audit', ['hash'])
    op.create_index('ix_audit_ts', 'audit', ['ts'])
    
    # ========================================================================
    # DEVICES
    # ========================================================================
    
    op.create_table('devices',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('device_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('vendor', sa.String(length=255), nullable=True),
        sa.Column('model', sa.String(length=255), nullable=True),
        sa.Column('serial_number', sa.String(length=255), nullable=True),
        sa.Column('device_type', sa.String(length=100), nullable=True),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.String(length=10), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('device_id')
    )
    op.create_index('ix_devices_device_id', 'devices', ['device_id'])
    op.create_index('ix_devices_device_type', 'devices', ['device_type'])
    
    op.create_table('device_status',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('device_id', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('last_calibration_date', sa.Date(), nullable=True),
        sa.Column('next_calibration_due', sa.Date(), nullable=True),
        sa.Column('health_score', sa.Float(), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('device_id')
    )
    op.create_index('ix_device_status_device_id', 'device_status', ['device_id'])
    op.create_index('ix_device_status_status', 'device_status', ['status'])
    op.create_index('ix_device_status_last_calibration_date', 'device_status', ['last_calibration_date'])
    op.create_index('ix_device_status_next_calibration_due', 'device_status', ['next_calibration_due'])
    
    # ========================================================================
    # LIMS - SAMPLES
    # ========================================================================
    
    op.create_table('projects',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('owner', sa.String(length=255), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(['owner'], ['users.email'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id')
    )
    op.create_index('ix_projects_project_id', 'projects', ['project_id'])
    op.create_index('ix_projects_status', 'projects', ['status'])
    
    op.create_table('containers',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('container_id', sa.String(length=255), nullable=False),
        sa.Column('container_type', sa.String(length=100), nullable=True),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('capacity', sa.String(length=50), nullable=True),
        sa.Column('current_count', sa.String(length=50), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('container_id')
    )
    op.create_index('ix_containers_container_id', 'containers', ['container_id'])
    op.create_index('ix_containers_location', 'containers', ['location'])
    
    op.create_table('batches',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('batch_id', sa.String(length=255), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('batch_id')
    )
    op.create_index('ix_batches_batch_id', 'batches', ['batch_id'])
    op.create_index('ix_batches_project_id', 'batches', ['project_id'])
    op.create_index('ix_batches_status', 'batches', ['status'])
    
    op.create_table('samples',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('sample_id', sa.String(length=255), nullable=False),
        sa.Column('lot_number', sa.String(length=255), nullable=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('container_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('parent_sample_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('batch_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('updated_by', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['batch_id'], ['batches.id'], ),
        sa.ForeignKeyConstraint(['container_id'], ['containers.id'], ),
        sa.ForeignKeyConstraint(['parent_sample_id'], ['samples.id'], ),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('sample_id')
    )
    op.create_index('ix_samples_sample_id', 'samples', ['sample_id'])
    op.create_index('ix_samples_lot_number', 'samples', ['lot_number'])
    op.create_index('ix_samples_project_id', 'samples', ['project_id'])
    op.create_index('ix_samples_container_id', 'samples', ['container_id'])
    op.create_index('ix_samples_parent_sample_id', 'samples', ['parent_sample_id'])
    op.create_index('ix_samples_batch_id', 'samples', ['batch_id'])
    op.create_index('ix_samples_status', 'samples', ['status'])
    op.create_index('ix_samples_created_at', 'samples', ['created_at'])
    
    op.create_table('sample_lineage',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('parent_sample_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('child_sample_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('relationship_type', sa.String(length=100), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(['child_sample_id'], ['samples.id'], ),
        sa.ForeignKeyConstraint(['parent_sample_id'], ['samples.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_sample_lineage_parent', 'sample_lineage', ['parent_sample_id'])
    op.create_index('ix_sample_lineage_child', 'sample_lineage', ['child_sample_id'])
    
    # ========================================================================
    # INVENTORY
    # ========================================================================
    
    op.create_table('vendors',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('vendor_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('contact_info', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('vendor_id')
    )
    op.create_index('ix_vendors_vendor_id', 'vendors', ['vendor_id'])
    
    op.create_table('inventory_items',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('item_code', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('unit', sa.String(length=50), nullable=True),
        sa.Column('min_stock', sa.Integer(), nullable=True),
        sa.Column('current_stock', sa.Integer(), nullable=True),
        sa.Column('reorder_point', sa.Integer(), nullable=True),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('item_code')
    )
    op.create_index('ix_inventory_items_item_code', 'inventory_items', ['item_code'])
    op.create_index('ix_inventory_items_name', 'inventory_items', ['name'])
    op.create_index('ix_inventory_items_category', 'inventory_items', ['category'])
    op.create_index('ix_inventory_items_location', 'inventory_items', ['location'])
    op.create_index('ix_inventory_items_current_stock', 'inventory_items', ['current_stock'])
    
    op.create_table('stock_lots',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('lot_number', sa.String(length=255), nullable=False),
        sa.Column('item_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('vendor_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('quantity_remaining', sa.Integer(), nullable=False),
        sa.Column('received_date', sa.Date(), nullable=True),
        sa.Column('expiration_date', sa.Date(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['item_id'], ['inventory_items.id'], ),
        sa.ForeignKeyConstraint(['vendor_id'], ['vendors.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('lot_number')
    )
    op.create_index('ix_stock_lots_lot_number', 'stock_lots', ['lot_number'])
    op.create_index('ix_stock_lots_item_id', 'stock_lots', ['item_id'])
    op.create_index('ix_stock_lots_vendor_id', 'stock_lots', ['vendor_id'])
    op.create_index('ix_stock_lots_received_date', 'stock_lots', ['received_date'])
    op.create_index('ix_stock_lots_expiration_date', 'stock_lots', ['expiration_date'])
    op.create_index('ix_stock_lots_status', 'stock_lots', ['status'])
    
    # ========================================================================
    # CALIBRATION
    # ========================================================================
    
    op.create_table('calibration_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('device_id', sa.String(length=255), nullable=False),
        sa.Column('calibration_date', sa.DateTime(), nullable=False),
        sa.Column('performed_by', sa.String(length=255), nullable=False),
        sa.Column('next_due_date', sa.Date(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('results', sa.JSON(), nullable=True),
        sa.Column('certificate_path', sa.String(length=500), nullable=True),
        sa.Column('notes', sa.String(length=1000), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['performed_by'], ['users.email'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_calibration_entries_device_id', 'calibration_entries', ['device_id'])
    op.create_index('ix_calibration_entries_calibration_date', 'calibration_entries', ['calibration_date'])
    op.create_index('ix_calibration_entries_next_due_date', 'calibration_entries', ['next_due_date'])
    op.create_index('ix_calibration_entries_status', 'calibration_entries', ['status'])
    
    # ========================================================================
    # WORKFLOWS
    # ========================================================================
    
    op.create_table('workflow_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workflow_name', sa.String(length=255), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('definition', sa.JSON(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.String(length=10), nullable=True),
        sa.Column('is_approved', sa.String(length=10), nullable=True),
        sa.Column('locked', sa.String(length=10), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('approved_by', sa.String(length=255), nullable=True),
        sa.Column('approved_at', sa.DateTime(), nullable=True),
        sa.Column('definition_hash', sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(['created_by'], ['users.email'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('workflow_name', 'version', name='uq_workflow_name_version')
    )
    op.create_index('ix_workflow_versions_workflow_name', 'workflow_versions', ['workflow_name'])
    op.create_index('ix_workflow_versions_is_active', 'workflow_versions', ['workflow_name', 'is_active'])
    op.create_index('ix_workflow_versions_created_at', 'workflow_versions', ['created_at'])
    
    op.create_table('config_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('snapshot_id', sa.String(length=255), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('config_data', sa.JSON(), nullable=False),
        sa.Column('config_hash', sa.String(length=64), nullable=False),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('snapshot_id')
    )
    op.create_index('ix_config_snapshots_snapshot_id', 'config_snapshots', ['snapshot_id'])
    op.create_index('ix_config_snapshots_timestamp', 'config_snapshots', ['timestamp'])
    op.create_index('ix_config_snapshots_config_hash', 'config_snapshots', ['config_hash'])
    
    op.create_table('workflow_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', sa.String(length=255), nullable=False),
        sa.Column('workflow_version_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('operator', sa.String(length=255), nullable=False),
        sa.Column('results', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('config_snapshot_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('run_metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['config_snapshot_id'], ['config_snapshots.id'], ),
        sa.ForeignKeyConstraint(['operator'], ['users.email'], ),
        sa.ForeignKeyConstraint(['workflow_version_id'], ['workflow_versions.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id')
    )
    op.create_index('ix_workflow_executions_run_id', 'workflow_executions', ['run_id'])
    op.create_index('ix_workflow_executions_workflow_version_id', 'workflow_executions', ['workflow_version_id'])
    op.create_index('ix_workflow_executions_started_at', 'workflow_executions', ['started_at'])
    op.create_index('ix_workflow_executions_status', 'workflow_executions', ['status'])
    
    op.create_table('workflow_sample_assignments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workflow_execution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('sample_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('assigned_at', sa.DateTime(), nullable=True),
        sa.Column('assigned_by', sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(['sample_id'], ['samples.id'], ),
        sa.ForeignKeyConstraint(['workflow_execution_id'], ['workflow_executions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_workflow_sample_assignments_workflow_execution_id', 'workflow_sample_assignments', ['workflow_execution_id'])
    op.create_index('ix_workflow_sample_assignments_sample_id', 'workflow_sample_assignments', ['sample_id'])
    
    # ========================================================================
    # MULTI-SITE (NODES & HUBS)
    # ========================================================================
    
    op.create_table('nodes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('node_id', sa.String(length=255), nullable=False),
        sa.Column('lab_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('hostname', sa.String(length=255), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('last_heartbeat', sa.DateTime(), nullable=True),
        sa.Column('capabilities', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lab_id'], ['labs.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('node_id')
    )
    op.create_index('ix_nodes_node_id', 'nodes', ['node_id'])
    op.create_index('ix_nodes_lab_id', 'nodes', ['lab_id'])
    op.create_index('ix_nodes_status', 'nodes', ['status'])
    op.create_index('ix_nodes_last_heartbeat', 'nodes', ['last_heartbeat'])
    
    op.create_table('device_hubs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('hub_id', sa.String(length=255), nullable=False),
        sa.Column('node_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('device_registry', sa.JSON(), nullable=False),
        sa.Column('health_status', sa.JSON(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['node_id'], ['nodes.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('hub_id')
    )
    op.create_index('ix_device_hubs_hub_id', 'device_hubs', ['hub_id'])
    op.create_index('ix_device_hubs_node_id', 'device_hubs', ['node_id'])


def downgrade() -> None:
    """Drop all tables."""
    # Drop in reverse order to respect foreign keys
    op.drop_table('device_hubs')
    op.drop_table('nodes')
    op.drop_table('workflow_sample_assignments')
    op.drop_table('workflow_executions')
    op.drop_table('config_snapshots')
    op.drop_table('workflow_versions')
    op.drop_table('calibration_entries')
    op.drop_table('stock_lots')
    op.drop_table('inventory_items')
    op.drop_table('vendors')
    op.drop_table('sample_lineage')
    op.drop_table('samples')
    op.drop_table('batches')
    op.drop_table('containers')
    op.drop_table('projects')
    op.drop_table('device_status')
    op.drop_table('devices')
    op.drop_table('audit')
    op.drop_table('user_roles')
    op.drop_table('roles')
    op.drop_table('users')
    op.drop_table('labs')
    op.drop_table('organizations')
