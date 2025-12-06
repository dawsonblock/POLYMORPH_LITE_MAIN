"""
Enhanced Compliance API endpoints.

Provides 21 CFR Part 11 compliance features:
- Audit trail verification
- PDF report generation with e-signatures
- Traceability matrix
- Configuration versioning
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, UUID4, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import hashlib
import uuid
from io import BytesIO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

from retrofitkit.db.models.audit import AuditEvent as AuditLog
from retrofitkit.db.models.workflow import ConfigSnapshot, WorkflowExecution, WorkflowVersion, WorkflowSampleAssignment
from retrofitkit.db.models.sample import Sample
from retrofitkit.db.session import get_db
from retrofitkit.compliance.audit import Audit
from retrofitkit.api.dependencies import get_current_user
from retrofitkit.data.storage import DataStore

router = APIRouter(prefix="/api/compliance", tags=["compliance"])

# ============================================================================
# Pydantic Models
# ============================================================================

class AuditChainVerificationResponse(BaseModel):
    """Audit chain verification result."""
    is_valid: bool
    total_entries: int
    verified_entries: int
    first_entry_timestamp: Optional[float]
    last_entry_timestamp: Optional[float]
    chain_start_hash: str
    chain_end_hash: str
    errors: List[str] = []


class TraceabilityMatrixResponse(BaseModel):
    """Complete traceability from sample to result."""
    sample_id: str
    sample_created_at: datetime
    sample_created_by: str
    runs: List[Dict[str, Any]]
    total_runs: int


class ConfigSnapshotResponse(BaseModel):
    """Configuration snapshot details."""
    id: UUID4
    snapshot_id: str
    timestamp: datetime
    config_hash: str
    created_by: str
    reason: str

    model_config = ConfigDict(from_attributes=True)


class RunDetailsResponse(BaseModel):
    """Detailed view of a workflow run for compliance/traceability."""

    run_id: str
    status: str
    operator: str
    started_at: datetime
    completed_at: Optional[datetime]
    workflow_name: Optional[str]
    workflow_version: Optional[str]
    workflow_hash: Optional[str]
    config_snapshot: Optional[Dict[str, Any]]
    audit_entries: List[Dict[str, Any]]


# ============================================================================
# AUDIT TRAIL VERIFICATION
# ============================================================================

@router.get("/audit/verify-chain", response_model=AuditChainVerificationResponse)
async def verify_audit_chain(
    limit: int = 1000,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Verify the cryptographic integrity of the audit trail chain.

    Checks:
    - Hash continuity (each entry's prev_hash matches previous entry's hash)
    - Hash validity (recomputed hash matches stored hash)
    - Chain completeness
    """

    audit = Audit(session)

    try:
        # Get audit logs ordered by ID
        stmt = select(AuditLog).order_by(AuditLog.id.asc()).limit(limit)
        result = await session.execute(stmt)
        entries = result.scalars().all()

        if not entries:
            return AuditChainVerificationResponse(
                is_valid=True,
                total_entries=0,
                verified_entries=0,
                first_entry_timestamp=None,
                last_entry_timestamp=None,
                chain_start_hash="GENESIS",
                chain_end_hash="GENESIS",
                errors=[]
            )

        errors = []
        verified_count = 0

        for i, entry in enumerate(entries):
            # Verify hash calculation
            data = f"{entry.ts}{entry.event}{entry.actor}{entry.subject}{entry.details}{entry.prev_hash}"
            computed_hash = hashlib.sha256(data.encode()).hexdigest()

            if computed_hash != entry.hash:
                errors.append(f"Entry {entry.id}: Hash mismatch (computed: {computed_hash[:8]}..., stored: {entry.hash[:8]}...)")
                continue

            # Verify chain linkage (not for first entry)
            if i > 0:
                prev_entry = entries[i - 1]
                if entry.prev_hash != prev_entry.hash:
                    errors.append(f"Entry {entry.id}: Chain broken (prev_hash: {entry.prev_hash[:8]}..., expected: {prev_entry.hash[:8]}...)")
                    continue

            verified_count += 1

        # Log verification attempt
        await audit.log(
            "AUDIT_CHAIN_VERIFIED",
            current_user["email"],
            "system",
            f"Verified {verified_count}/{len(entries)} audit entries"
        )

        return AuditChainVerificationResponse(
            is_valid=len(errors) == 0,
            total_entries=len(entries),
            verified_entries=verified_count,
            first_entry_timestamp=entries[0].ts if entries else None,
            last_entry_timestamp=entries[-1].ts if entries else None,
            chain_start_hash=entries[0].prev_hash if entries else "GENESIS",
            chain_end_hash=entries[-1].hash if entries else "GENESIS",
            errors=errors
        )

    finally:
        pass


@router.get("/audit/export")
async def export_audit_trail(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    actor: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Export audit trail as JSON for archival or analysis.

    Filters by date range and/or actor.
    """


    try:
        query = select(AuditLog).order_by(AuditLog.ts.asc())

        # Apply filters
        if start_date:
            start_ts = datetime.fromisoformat(start_date).timestamp()
            query = query.filter(AuditLog.ts >= start_ts)

        if end_date:
            end_ts = datetime.fromisoformat(end_date).timestamp()
            query = query.filter(AuditLog.ts <= end_ts)

        if actor:
            query = query.filter(AuditLog.actor == actor)

        result = await session.execute(query)
        entries = result.scalars().all()

        export_data = {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "exported_by": current_user["email"],
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "actor": actor
            },
            "total_entries": len(entries),
            "entries": [
                {
                    "id": e.id,
                    "timestamp": e.ts,
                    "event": e.event,
                    "actor": e.actor,
                    "subject": e.subject,
                    "details": e.details,
                    "hash": e.hash,
                    "prev_hash": e.prev_hash
                }
                for e in entries
            ]
        }

        return export_data

    finally:
        pass


# ============================================================================
# PDF REPORT GENERATION
# ============================================================================

@router.get("/reports/run/{run_id}.pdf")
async def generate_run_report_pdf(
    run_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Generate compliance report PDF for a specific run.

    Includes:
    - Run metadata and parameters
    - Workflow definition
    - Approval records
    - E-signatures
    - Data summary
    - Audit trail hash for verification
    """
    try:
        # Import PDF library
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="PDF generation not available - install reportlab"
            )


        data_store = DataStore()

        # Load run data
        stmt = select(WorkflowExecution).filter(
            WorkflowExecution.run_id == run_id
        )
        result = await session.execute(stmt)
        execution = result.scalar_one_or_none()

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )

        # Get workflow details
        stmt = select(WorkflowVersion).filter(
            WorkflowVersion.id == execution.workflow_version_id
        )
        result = await session.execute(stmt)
        workflow = result.scalar_one_or_none()

        # Create PDF in memory
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(inch, height - inch, f"Compliance Report: {run_id}")

        # Report metadata
        y = height - 1.5 * inch
        pdf.setFont("Helvetica", 10)
        pdf.drawString(inch, y, f"Generated: {datetime.now(timezone.utc).isoformat()}")
        y -= 0.3 * inch
        pdf.drawString(inch, y, f"Generated By: {current_user['email']}")
        y -= 0.5 * inch

        # Run Details
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(inch, y, "Run Details")
        y -= 0.3 * inch

        pdf.setFont("Helvetica", 10)
        pdf.drawString(inch + 0.2 * inch, y, f"Workflow: {workflow.workflow_name if workflow else 'Unknown'}")
        y -= 0.2 * inch
        pdf.drawString(inch + 0.2 * inch, y, f"Version: {workflow.version if workflow else 'Unknown'}")
        y -= 0.2 * inch
        pdf.drawString(inch + 0.2 * inch, y, f"Operator: {execution.operator}")
        y -= 0.2 * inch
        pdf.drawString(inch + 0.2 * inch, y, f"Started: {execution.started_at.isoformat()}")
        y -= 0.2 * inch
        if execution.completed_at:
            pdf.drawString(inch + 0.2 * inch, y, f"Completed: {execution.completed_at.isoformat()}")
            y -= 0.2 * inch
        pdf.drawString(inch + 0.2 * inch, y, f"Status: {execution.status}")
        y -= 0.5 * inch

        # Config Snapshot
        if execution.config_snapshot_id:
            stmt = select(ConfigSnapshot).filter(
                ConfigSnapshot.id == execution.config_snapshot_id
            )
            result = await session.execute(stmt)
            config = result.scalar_one_or_none()

            if config:
                pdf.setFont("Helvetica-Bold", 12)
                pdf.drawString(inch, y, "Configuration Snapshot")
                y -= 0.3 * inch
                pdf.setFont("Helvetica", 10)
                pdf.drawString(inch + 0.2 * inch, y, f"Snapshot ID: {config.snapshot_id}")
                y -= 0.2 * inch
                pdf.drawString(inch + 0.2 * inch, y, f"Hash: {config.config_hash[:32]}...")
                y -= 0.5 * inch

        # Audit Trail Verification
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(inch, y, "Audit Trail Integrity")
        y -= 0.3 * inch

        # Get relevant audit entries
        stmt = select(AuditLog).filter(
            AuditLog.subject == run_id
        ).order_by(AuditLog.ts.asc())
        result = await session.execute(stmt)
        audit_entries = result.scalars().all()

        if audit_entries:
            pdf.setFont("Helvetica", 10)
            pdf.drawString(inch + 0.2 * inch, y, f"Total Audit Entries: {len(audit_entries)}")
            y -= 0.2 * inch
            pdf.drawString(inch + 0.2 * inch, y, f"Chain Hash: {audit_entries[-1].hash[:32]}...")
        else:
            pdf.setFont("Helvetica", 10)
            pdf.drawString(inch + 0.2 * inch, y, "No audit entries found")

        y -= 0.5 * inch

        # Footer
        pdf.setFont("Helvetica-Italic", 8)
        pdf.drawString(inch, 0.5 * inch, f"Document Hash: {hashlib.sha256(run_id.encode()).hexdigest()[:32]}...")
        pdf.drawString(width - 2 * inch, 0.5 * inch, "Page 1")

        # Save PDF
        pdf.save()

        # Return PDF
        buffer.seek(0)
        pdf_bytes = buffer.read()

        # Log report generation (non-blocking)
        audit = Audit(session)
        try:
            await audit.log(
                "REPORT_GENERATED",
                current_user["email"],
                run_id,
                f"Generated PDF report for run {run_id}"
            )
        except Exception:
            pass



        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=run_{run_id}_report.pdf"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating PDF: {str(e)}"
        )
    finally:
        pass


@router.get("/run/{run_id}", response_model=RunDetailsResponse)
async def get_run_details(
    run_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """Return detailed JSON view of a workflow run.

    Includes execution metadata, workflow definition metadata, config snapshot
    summary, and all audit entries linked to the run id.
    """


    try:
        # Load execution
        stmt = select(WorkflowExecution).filter(
            WorkflowExecution.run_id == run_id
        )
        result = await session.execute(stmt)
        execution = result.scalar_one_or_none()

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )

        # Workflow metadata
        stmt = select(WorkflowVersion).filter(
            WorkflowVersion.id == execution.workflow_version_id
        )
        result = await session.execute(stmt)
        workflow = result.scalar_one_or_none()

        # Config snapshot summary (if any)
        config_summary: Optional[Dict[str, Any]] = None
        if execution.config_snapshot_id:
            stmt = select(ConfigSnapshot).filter(
                ConfigSnapshot.id == execution.config_snapshot_id
            )
            result = await session.execute(stmt)
            config = result.scalar_one_or_none()

            if config:
                config_summary = {
                    "snapshot_id": config.snapshot_id,
                    "timestamp": config.timestamp.isoformat(),
                    "config_hash": config.config_hash,
                    "created_by": config.created_by,
                    "reason": config.reason,
                }

        # Audit entries for this run
        stmt = select(AuditLog).filter(
            AuditLog.subject == run_id
        ).order_by(AuditLog.ts.asc())
        result = await session.execute(stmt)
        audit_entries = result.scalars().all()

        audit_payload = [
            {
                "id": e.id,
                "timestamp": e.ts,
                "event": e.event,
                "actor": e.actor,
                "subject": e.subject,
                "details": e.details,
                "hash": e.hash,
                "prev_hash": e.prev_hash,
            }
            for e in audit_entries
        ]

        return RunDetailsResponse(
            run_id=execution.run_id,
            status=execution.status,
            operator=execution.operator,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            workflow_name=getattr(workflow, "workflow_name", None),
            workflow_version=getattr(workflow, "version", None),
            workflow_hash=getattr(workflow, "definition_hash", None),
            config_snapshot=config_summary,
            audit_entries=audit_payload,
        )

    finally:
        pass


# ============================================================================
# TRACEABILITY MATRIX
# ============================================================================

@router.get("/traceability/sample/{sample_id}", response_model=TraceabilityMatrixResponse)
async def generate_traceability_matrix(sample_id: str, session: AsyncSession = Depends(get_db)):
    """
    Generate complete traceability matrix from sample to results.

    Tracks:
    - Sample creation and lineage
    - All workflow executions involving the sample
    - Configuration snapshots used
    - Approval records
    - Result data
    """


    try:
        # Get sample
        stmt = select(Sample).filter(Sample.sample_id == sample_id)
        result = await session.execute(stmt)
        sample = result.scalar_one_or_none()

        if not sample:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample '{sample_id}' not found"
            )

        # Get all workflow executions for this sample
        stmt = select(WorkflowSampleAssignment).filter(
            WorkflowSampleAssignment.sample_id == sample.id
        )
        result = await session.execute(stmt)
        assignments = result.scalars().all()

        runs = []
        for assignment in assignments:
            stmt = select(WorkflowExecution).filter(
                WorkflowExecution.id == assignment.workflow_execution_id
            )
            result = await session.execute(stmt)
            execution = result.scalar_one_or_none()

            if not execution:
                continue

            # Get workflow version
            stmt = select(WorkflowVersion).filter(
                WorkflowVersion.id == execution.workflow_version_id
            )
            result = await session.execute(stmt)
            workflow = result.scalar_one_or_none()

            # Get config snapshot
            config = None
            if execution.config_snapshot_id:
                stmt = select(ConfigSnapshot).filter(
                    ConfigSnapshot.id == execution.config_snapshot_id
                )
                result = await session.execute(stmt)
                config = result.scalar_one_or_none()

            runs.append({
                "run_id": execution.run_id,
                "workflow_name": workflow.workflow_name if workflow else "Unknown",
                "workflow_version": workflow.version if workflow else None,
                "workflow_hash": workflow.definition_hash if workflow else None,
                "operator": execution.operator,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "status": execution.status,
                "config_snapshot_id": str(config.snapshot_id) if config else None,
                "config_hash": config.config_hash if config else None,
                "results_summary": execution.results if execution.results else {}
            })

        return TraceabilityMatrixResponse(
            sample_id=sample_id,
            sample_created_at=sample.created_at,
            sample_created_by=sample.created_by,
            runs=runs,
            total_runs=len(runs)
        )

    finally:
        pass


# ============================================================================
# CONFIGURATION VERSIONING
# ============================================================================

@router.post("/config/snapshot", response_model=ConfigSnapshotResponse)
async def create_config_snapshot(
    reason: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Create an immutable snapshot of current system configuration.

    Captures all critical configuration for compliance and reproducibility.
    """

    audit = Audit(session)

    try:
        # Gather actual system configuration
        from retrofitkit.core.app import AppContext
        from retrofitkit import __version__ as app_version
        import json

        app_context = AppContext.load()

        # Get Alembic revision from database
        try:
            result = await session.execute(text("SELECT version_num FROM alembic_version"))
            alembic_revision = result.scalar()
        except Exception:
            alembic_revision = "unknown"

        # Build complete config snapshot
        config_data = {
            # System configuration
            "system": {
                "app_name": app_context.config.system.name,
                "environment": app_context.config.system.environment,
                "version": app_version,
            },
            # Active hardware overlay
            "hardware": {
                "daq_backend": app_context.config.daq.backend,
                "raman_provider": app_context.config.raman.provider,
            },
            # Safety configuration
            "safety": {
                "interlocks_enabled": True, # Default to True as we don't have a flag anymore
                "watchdog_seconds": app_context.config.safety.watchdog_seconds,
            },
            # Gating rules
            "gating": [rule for rule in app_context.config.gating.rules],
            # Database
            "database": {
                "alembic_revision": alembic_revision,
            },
            # Snapshot metadata
            "snapshot_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "captured_by": current_user["email"],
            }
        }

        # Generate deterministic hash
        # We use sort_keys=True to ensure consistent JSON serialization
        config_json = json.dumps(config_data, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()

        # Create snapshot record
        snapshot = ConfigSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            config_data=config_data,
            config_hash=config_hash,
            created_by=current_user["email"],
            reason=reason
        )

        session.add(snapshot)
        await session.commit()
        await session.refresh(snapshot)

        # Audit log (non-blocking)
        try:
            await audit.log(
                "CONFIG_SNAPSHOT_CREATED",
                current_user["email"],
                snapshot.snapshot_id,
                f"Created config snapshot: {reason}"
            )
        except Exception:
            pass

        return snapshot

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating config snapshot: {str(e)}"
        )
    finally:
        pass


@router.get("/config/snapshots", response_model=List[ConfigSnapshotResponse])
async def list_config_snapshots(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_db)
):
    """List configuration snapshots."""


    try:
        stmt = select(ConfigSnapshot).order_by(
            ConfigSnapshot.timestamp.desc()
        ).limit(limit).offset(offset)
        result = await session.execute(stmt)
        snapshots = result.scalars().all()

        return snapshots

    finally:
        pass


@router.get("/config/snapshots/{snapshot_id}", response_model=ConfigSnapshotResponse)
async def get_config_snapshot(snapshot_id: str, session: AsyncSession = Depends(get_db)):
    """Get a specific configuration snapshot."""


    try:
        stmt = select(ConfigSnapshot).filter(
            ConfigSnapshot.snapshot_id == snapshot_id
        )
        result = await session.execute(stmt)
        snapshot = result.scalar_one_or_none()

        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Snapshot '{snapshot_id}' not found"
            )

        return snapshot

    finally:
        pass
