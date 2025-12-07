"""Add polymorph_modes and polymorph_mode_snapshots tables

Revision ID: 20241206_add_polymorph_modes
Revises: 
Create Date: 2024-12-06
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20241206_add_polymorph_modes'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create polymorph_modes table
    op.create_table(
        'polymorph_modes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('device_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('mode_index', sa.Integer(), nullable=False),
        sa.Column('poly_id_hash', sa.String(64), nullable=False),
        sa.Column('poly_name', sa.String(100), nullable=True),
        sa.Column('mu', postgresql.JSON(), nullable=False),
        sa.Column('F', postgresql.JSON(), nullable=True),
        sa.Column('occupancy', sa.Float(), server_default='0.0'),
        sa.Column('risk', sa.Float(), server_default='0.0'),
        sa.Column('age', sa.Integer(), server_default='0'),
        sa.Column('lambda_i', sa.Float(), server_default='1.0'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('first_seen', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', postgresql.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for polymorph_modes
    op.create_index('idx_polymorph_modes_org', 'polymorph_modes', ['org_id'])
    op.create_index('idx_polymorph_modes_device', 'polymorph_modes', ['device_id'])
    op.create_index('idx_polymorph_modes_poly_hash', 'polymorph_modes', ['poly_id_hash'])
    op.create_index('idx_polymorph_modes_active', 'polymorph_modes', ['is_active'])
    
    # Create polymorph_mode_snapshots table
    op.create_table(
        'polymorph_mode_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('mode_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('state_blob', sa.LargeBinary(), nullable=True),
        sa.Column('workflow_execution_id', sa.String(36), nullable=True),
        sa.Column('trigger', sa.String(50), nullable=True),
        sa.Column('occupancy', sa.Float(), nullable=True),
        sa.Column('risk', sa.Float(), nullable=True),
        sa.Column('age', sa.Integer(), nullable=True),
        sa.Column('snapshot_metadata', postgresql.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['mode_id'], ['polymorph_modes.id'], ondelete='CASCADE')
    )
    
    # Create indexes for polymorph_mode_snapshots
    op.create_index('idx_mode_snapshots_mode', 'polymorph_mode_snapshots', ['mode_id'])
    op.create_index('idx_mode_snapshots_timestamp', 'polymorph_mode_snapshots', ['timestamp'])
    op.create_index('idx_mode_snapshots_workflow', 'polymorph_mode_snapshots', ['workflow_execution_id'])


def downgrade() -> None:
    op.drop_table('polymorph_mode_snapshots')
    op.drop_table('polymorph_modes')
