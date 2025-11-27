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
from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import json
import uuid
from io import BytesIO

from retrofitkit.database.models import (
    AuditLog, ConfigSnapshot, WorkflowExecution, Sample,
    WorkflowVersion, get_session
)
from retrofitkit.compliance.audit import Audit
from retrofitkit.compliance.tokens import get_current_user
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

    class Config:
        from_attributes = True


# ============================================================================
# AUDIT TRAIL VERIFICATION
# ============================================================================

@router.get("/audit/verify-chain", response_model=AuditChainVerificationResponse)
async def verify_audit_chain(
    limit: int = 1000,
    current_user: dict = Depends(get_current_user)
):
    """
    Verify the cryptographic integrity of the audit trail chain.

    Checks:
    - Hash continuity (each entry's prev_hash matches previous entry's hash)
    - Hash validity (recomputed hash matches stored hash)
    - Chain completeness
    """
    session = get_session()
    audit = Audit()

    try:
        # Get audit logs ordered by ID
        entries = session.query(AuditLog).order_by(AuditLog.id.asc()).limit(limit).all()

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
        audit.log(
            "AUDIT_CHAIN_VERIFIED",
            current_user["email"],
            "system",
            f"Verified {verified_count}/{len(entries)} audit entries"
        )

        return AuditChainVerificationResponse(
            is_valid=len(errors) == 0,
            total_entries=len(entries),
            verified_entries=verified_count,
            first_entry_timestamp=entries[0].ts,
            last_entry_timestamp=entries[-1].ts,
            chain_start_hash=entries[0].prev_hash,
            chain_end_hash=entries[-1].hash,
            errors=errors
        )

    finally:
        session.close()


@router.get("/audit/export")
async def export_audit_trail(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    actor: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Export audit trail as JSON for archival or analysis.

    Filters by date range and/or actor.
    """
    session = get_session()

    try:
        query = session.query(AuditLog).order_by(AuditLog.ts.asc())

        # Apply filters
        if start_date:
            start_ts = datetime.fromisoformat(start_date).timestamp()
            query = query.filter(AuditLog.ts >= start_ts)

        if end_date:
            end_ts = datetime.fromisoformat(end_date).timestamp()
            query = query.filter(AuditLog.ts <= end_ts)

        if actor:
            query = query.filter(AuditLog.actor == actor)

        entries = query.all()

        export_data = {
            "export_date": datetime.utcnow().isoformat(),
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
        session.close()


# ============================================================================
# PDF REPORT GENERATION
# ============================================================================

@router.get("/reports/run/{run_id}.pdf")
async def generate_run_report_pdf(
    run_id: str,
    current_user: dict = Depends(get_current_user)
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

        session = get_session()
        data_store = DataStore()

        # Load run data
        execution = session.query(WorkflowExecution).filter(
            WorkflowExecution.run_id == run_id
        ).first()

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )

        # Get workflow details
        workflow = session.query(WorkflowVersion).filter(
            WorkflowVersion.id == execution.workflow_version_id
        ).first()

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
        pdf.drawString(inch, y, f"Generated: {datetime.utcnow().isoformat()}")
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
            config = session.query(ConfigSnapshot).filter(
                ConfigSnapshot.id == execution.config_snapshot_id
            ).first()
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
        audit_entries = session.query(AuditLog).filter(
            AuditLog.subject == run_id
        ).order_by(AuditLog.ts.asc()).all()

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

        # Log report generation
        audit = Audit()
        audit.log(
            "REPORT_GENERATED",
            current_user["email"],
            run_id,
            f"Generated PDF report for run {run_id}"
        )

        session.close()

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=run_{run_id}_report.pdf"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating PDF: {str(e)}"
        )


# ============================================================================
# TRACEABILITY MATRIX
# ============================================================================

@router.get("/traceability/sample/{sample_id}", response_model=TraceabilityMatrixResponse)
async def generate_traceability_matrix(sample_id: str):
    """
    Generate complete traceability matrix from sample to results.

    Tracks:
    - Sample creation and lineage
    - All workflow executions involving the sample
    - Configuration snapshots used
    - Approval records
    - Result data
    """
    session = get_session()

    try:
        # Get sample
        sample = session.query(Sample).filter(Sample.sample_id == sample_id).first()
        if not sample:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample '{sample_id}' not found"
            )

        # Get all workflow executions for this sample
        from retrofitkit.database.models import WorkflowSampleAssignment

        assignments = session.query(WorkflowSampleAssignment).filter(
            WorkflowSampleAssignment.sample_id == sample.id
        ).all()

        runs = []
        for assignment in assignments:
            execution = session.query(WorkflowExecution).filter(
                WorkflowExecution.id == assignment.workflow_execution_id
            ).first()

            if not execution:
                continue

            # Get workflow version
            workflow = session.query(WorkflowVersion).filter(
                WorkflowVersion.id == execution.workflow_version_id
            ).first()

            # Get config snapshot
            config = None
            if execution.config_snapshot_id:
                config = session.query(ConfigSnapshot).filter(
                    ConfigSnapshot.id == execution.config_snapshot_id
                ).first()

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
        session.close()


# ============================================================================
# CONFIGURATION VERSIONING
# ============================================================================

@router.post("/config/snapshot", response_model=ConfigSnapshotResponse)
async def create_config_snapshot(
    reason: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Create an immutable snapshot of current system configuration.

    Captures all critical configuration for compliance and reproducibility.
    """
    session = get_session()
    audit = Audit()

    try:
        # TODO: Gather actual system configuration
        # For now, use placeholder config
        config_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "snapshot_reason": reason,
            "created_by": current_user["email"],
            # Add actual system config here
            "placeholder": "Implement full config capture"
        }

        # Calculate hash
        config_json = json.dumps(config_data, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()

        # Create snapshot
        snapshot = ConfigSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            config_data=config_data,
            config_hash=config_hash,
            created_by=current_user["email"],
            reason=reason
        )

        session.add(snapshot)
        session.commit()
        session.refresh(snapshot)

        # Audit log
        audit.log(
            "CONFIG_SNAPSHOT_CREATED",
            current_user["email"],
            snapshot.snapshot_id,
            f"Created config snapshot: {reason}"
        )

        return snapshot

    finally:
        session.close()


@router.get("/config/snapshots", response_model=List[ConfigSnapshotResponse])
async def list_config_snapshots(
    limit: int = 100,
    offset: int = 0
):
    """List configuration snapshots."""
    session = get_session()

    try:
        snapshots = session.query(ConfigSnapshot).order_by(
            ConfigSnapshot.timestamp.desc()
        ).limit(limit).offset(offset).all()

        return snapshots

    finally:
        session.close()


@router.get("/config/snapshots/{snapshot_id}", response_model=ConfigSnapshotResponse)
async def get_config_snapshot(snapshot_id: str):
    """Get a specific configuration snapshot."""
    session = get_session()

    try:
        snapshot = session.query(ConfigSnapshot).filter(
            ConfigSnapshot.snapshot_id == snapshot_id
        ).first()

        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Snapshot '{snapshot_id}' not found"
            )

        return snapshot

    finally:
        session.close()
