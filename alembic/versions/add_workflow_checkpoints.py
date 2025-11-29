"""add workflow checkpoints

Revision ID: add_workflow_checkpoints
Revises: 
Create Date: 2024-11-29

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_workflow_checkpoints'
down_revision = None  # Will be updated by alembic
branch_labels = None
depends_on = None


def upgrade():
    """Add workflow_checkpoints table."""
    op.create_table(
        'workflow_checkpoints',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('step_index', sa.Integer(), nullable=False),
        sa.Column('step_type', sa.String(length=50), nullable=False),
        sa.Column('step_results', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('device_state', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('workflow_definition_hash', sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(['execution_id'], ['workflow_executions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_checkpoints_execution', 'workflow_checkpoints', ['execution_id'], unique=False)
    op.create_index('idx_checkpoints_step', 'workflow_checkpoints', ['step_index'], unique=False)
    op.create_index('idx_checkpoints_created', 'workflow_checkpoints', ['created_at'], unique=False)


def downgrade():
    """Remove workflow_checkpoints table."""
    op.drop_index('idx_checkpoints_created', table_name='workflow_checkpoints')
    op.drop_index('idx_checkpoints_step', table_name='workflow_checkpoints')
    op.drop_index('idx_checkpoints_execution', table_name='workflow_checkpoints')
    op.drop_table('workflow_checkpoints')
