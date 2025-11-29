"""add_polymorph_tables

Revision ID: add_polymorph_tables
Revises: 
Create Date: 2024-11-29

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'add_polymorph_tables'
down_revision = None  # Set to your current head revision
branch_labels = None
depends_on = None


def upgrade():
    # Create polymorph_events table
    op.create_table(
        'polymorph_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('event_id', sa.String(36), nullable=False, unique=True),
        sa.Column('detected_at', sa.Float(), nullable=False),
        sa.Column('polymorph_id', sa.Integer(), nullable=False),
        sa.Column('polymorph_name', sa.String(100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(20), nullable=False),
        sa.Column('workflow_execution_id', sa.String(36), nullable=True),
        sa.Column('sample_id', sa.String(36), nullable=True),
        sa.Column('operator_email', sa.String(255), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_polymorph_events_detected_at', 'detected_at'),
        sa.Index('idx_polymorph_events_polymorph_id', 'polymorph_id'),
        sa.Index('idx_polymorph_events_workflow', 'workflow_execution_id')
    )
    
    # Create polymorph_signatures table
    op.create_table(
        'polymorph_signatures',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('signature_id', sa.String(36), nullable=False, unique=True),
        sa.Column('event_id', sa.String(36), nullable=False),
        sa.Column('polymorph_id', sa.Integer(), nullable=False),
        sa.Column('signature_vector', sa.JSON(), nullable=False),  # Store as JSON array
        sa.Column('alternative_forms', sa.JSON(), nullable=True),
        sa.Column('spectral_features', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['event_id'], ['polymorph_events.event_id'], ondelete='CASCADE'),
        sa.Index('idx_polymorph_signatures_event', 'event_id'),
        sa.Index('idx_polymorph_signatures_polymorph', 'polymorph_id')
    )
    
    # Create polymorph_reports table
    op.create_table(
        'polymorph_reports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_id', sa.String(36), nullable=False, unique=True),
        sa.Column('event_id', sa.String(36), nullable=False),
        sa.Column('report_format', sa.String(10), nullable=False),  # 'json' or 'pdf'
        sa.Column('report_data', sa.Text(), nullable=False),  # JSON string or base64 PDF
        sa.Column('generated_at', sa.Float(), nullable=False),
        sa.Column('generated_by', sa.String(255), nullable=False),
        sa.Column('signed', sa.Boolean(), default=False),
        sa.Column('signature_hash', sa.String(64), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['event_id'], ['polymorph_events.event_id'], ondelete='CASCADE'),
        sa.Index('idx_polymorph_reports_event', 'event_id'),
        sa.Index('idx_polymorph_reports_generated_at', 'generated_at')
    )


def downgrade():
    op.drop_table('polymorph_reports')
    op.drop_table('polymorph_signatures')
    op.drop_table('polymorph_events')
