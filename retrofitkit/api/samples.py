"""
Sample Tracking API endpoints.

Provides CRUD operations for samples, containers, projects, and batches.
Includes lineage tracking and workflow assignment functionality.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, UUID4, field_validator, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from retrofitkit.db.models.sample import Project, Batch, Container, Sample, SampleLineage
from retrofitkit.db.models.workflow import WorkflowSampleAssignment
from retrofitkit.core.database import get_db_session
from retrofitkit.compliance.audit import Audit
from retrofitkit.api.dependencies import get_current_user, require_role

router = APIRouter(prefix="/api/samples", tags=["samples"])

# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class SampleCreate(BaseModel):
    sample_id: str = Field(..., min_length=1, max_length=255, description="Unique sample identifier")
    lot_number: Optional[str] = Field(None, max_length=255)
    project_id: Optional[UUID4] = None
    container_id: Optional[UUID4] = None
    parent_sample_id: Optional[UUID4] = None
    batch_id: Optional[UUID4] = None
    status: str = Field(default='active', pattern='^(active|consumed|disposed|quarantined)$')
    extra_data: Dict[str, Any] = {}

    @field_validator('sample_id')
    @classmethod
    def validate_sample_id(cls, v: str) -> str:
        """Validate sample ID format - alphanumeric, hyphens, underscores only."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('sample_id must contain only letters, numbers, hyphens, and underscores')
        return v


class SampleUpdate(BaseModel):
    status: Optional[str] = None
    container_id: Optional[UUID4] = None
    lot_number: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None


class SampleResponse(BaseModel):
    id: UUID4
    sample_id: str
    lot_number: Optional[str]
    project_id: Optional[UUID4]
    container_id: Optional[UUID4]
    parent_sample_id: Optional[UUID4]
    batch_id: Optional[UUID4]
    status: str
    extra_data: Dict[str, Any]
    created_by: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SampleWithLineage(SampleResponse):
    parent_samples: List[SampleResponse] = []
    child_samples: List[SampleResponse] = []


class ContainerCreate(BaseModel):
    container_id: str
    container_type: Optional[str] = None
    location: Optional[str] = None
    capacity: Optional[int] = None


class ContainerResponse(BaseModel):
    id: UUID4
    container_id: str
    container_type: Optional[str]
    location: Optional[str]
    capacity: Optional[int]
    current_count: int

    model_config = ConfigDict(from_attributes=True)


class ProjectCreate(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    status: str = 'active'


class ProjectResponse(BaseModel):
    id: UUID4
    project_id: str
    name: str
    description: Optional[str]
    status: str
    owner: Optional[str]
    created_by: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BatchCreate(BaseModel):
    batch_id: str
    project_id: Optional[UUID4] = None
    status: str = 'active'


class BatchResponse(BaseModel):
    id: UUID4
    batch_id: str
    project_id: Optional[UUID4]
    status: str
    created_by: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# SAMPLE ENDPOINTS
# ============================================================================

@router.post(
    "/",
    response_model=SampleResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("admin", "scientist"))]
)
async def create_sample(
    sample: SampleCreate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """Create a new sample."""
    audit = Audit(session)

    try:
        # Check if sample_id already exists
        result = await session.execute(select(Sample).where(Sample.sample_id == sample.sample_id))
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Sample with ID '{sample.sample_id}' already exists"
            )

        # Create sample
        new_sample = Sample(
            sample_id=sample.sample_id,
            lot_number=sample.lot_number,
            project_id=sample.project_id,
            container_id=sample.container_id,
            parent_sample_id=sample.parent_sample_id,
            batch_id=sample.batch_id,
            status=sample.status,
            extra_data=sample.extra_data,
            created_by=current_user["email"]
        )

        session.add(new_sample)
        await session.flush()  # Get the ID without committing

        # Create lineage entry if parent exists (same transaction)
        if sample.parent_sample_id:
            lineage = SampleLineage(
                parent_sample_id=sample.parent_sample_id,
                child_sample_id=new_sample.id,
                relationship_type='derived',
                created_by=current_user["email"]
            )
            session.add(lineage)

        # Commit everything in one transaction
        await session.commit()
        await session.refresh(new_sample)

        # Audit log (outside transaction - non-critical)
        try:
            await audit.log(
                "SAMPLE_CREATED",
                current_user["email"],
                sample.sample_id,
                f"Created sample {sample.sample_id}"
            )
        except Exception as e:
            # Log audit failure but don't fail the request
            print(f"Audit log failed: {e}")

        return new_sample

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Rollback on any other error
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating sample: {str(e)}"
        )


@router.post(
    "/bulk",
    response_model=List[SampleResponse],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("admin", "scientist"))]
)
async def create_samples_bulk(
    samples: List[SampleCreate],
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Create multiple samples in a single transaction.

    Maximum 100 samples per request for performance.
    """
    audit = Audit(session)

    # Validate batch size
    if len(samples) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 samples per bulk creation"
        )

    if len(samples) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No samples provided"
        )

    try:
        created_samples = []

        # Check for duplicates in request
        sample_ids = [s.sample_id for s in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Duplicate sample IDs in request"
            )

        # Check for existing samples
        result = await session.execute(select(Sample).where(Sample.sample_id.in_(sample_ids)))
        existing = result.scalars().all()

        if existing:
            existing_ids = [s.sample_id for s in existing]
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Samples already exist: {', '.join(existing_ids)}"
            )

        # Create all samples
        for sample in samples:
            new_sample = Sample(
                sample_id=sample.sample_id,
                lot_number=sample.lot_number,
                project_id=sample.project_id,
                container_id=sample.container_id,
                parent_sample_id=sample.parent_sample_id,
                batch_id=sample.batch_id,
                status=sample.status,
                extra_data=sample.extra_data,
                created_by=current_user["email"]
            )
            session.add(new_sample)
            created_samples.append(new_sample)

        # Commit all at once
        await session.commit()

        # Refresh all
        for sample in created_samples:
            await session.refresh(sample)

        # Audit log (non-blocking)
        try:
            await audit.log(
                "SAMPLES_BULK_CREATED",
                current_user["email"],
                "bulk_operation",
                f"Created {len(created_samples)} samples in bulk"
            )
        except Exception as e:
            print(f"Audit log failed: {e}")

        return created_samples

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating samples in bulk: {str(e)}"
        )


@router.post("/containers", response_model=ContainerResponse, status_code=status.HTTP_201_CREATED)
async def create_container(
    container: ContainerCreate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """Create a new container."""
    audit = Audit(session)

    try:
        result = await session.execute(select(Container).where(Container.container_id == container.container_id))
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Container '{container.container_id}' already exists"
            )

        new_container = Container(
            container_id=container.container_id,
            container_type=container.container_type,
            location=container.location,
            capacity=container.capacity
        )

        session.add(new_container)
        await session.commit()
        await session.refresh(new_container)

        await audit.log(
            "CONTAINER_CREATED",
            current_user["email"],
            container.container_id,
            f"Created container {container.container_id}"
        )

        return new_container

    finally:
        pass


@router.get("/containers", response_model=List[ContainerResponse])
async def list_containers(limit: int = 100, offset: int = 0, session: AsyncSession = Depends(get_db_session)):
    """List all containers."""
    result = await session.execute(select(Container).limit(limit).offset(offset))
    containers = result.scalars().all()
    return containers


@router.get("/containers/{container_id}", response_model=ContainerResponse)
async def get_container(container_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get container details."""
    result = await session.execute(select(Container).where(Container.container_id == container_id))
    container = result.scalar_one_or_none()
    if not container:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Container '{container_id}' not found"
        )
    return container


# ============================================================================
# PROJECT ENDPOINTS
# ============================================================================

@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project: ProjectCreate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """Create a new project."""
    audit = Audit(session)

    try:
        result = await session.execute(select(Project).where(Project.project_id == project.project_id))
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Project '{project.project_id}' already exists"
            )

        new_project = Project(
            project_id=project.project_id,
            name=project.name,
            description=project.description,
            status=project.status,
            owner=current_user["email"],
            created_by=current_user["email"]
        )

        session.add(new_project)
        await session.commit()
        await session.refresh(new_project)

        await audit.log(
            "PROJECT_CREATED",
            current_user["email"],
            project.project_id,
            f"Created project {project.project_id}"
        )

        return new_project

    finally:
        pass


@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_db_session)
):
    """List projects."""
    stmt = select(Project)
    if status:
        stmt = stmt.where(Project.status == status)

    stmt = stmt.limit(limit).offset(offset)
    result = await session.execute(stmt)
    projects = result.scalars().all()
    return projects


# ============================================================================
# BATCH ENDPOINTS
# ============================================================================

@router.post("/batches", response_model=BatchResponse, status_code=status.HTTP_201_CREATED)
async def create_batch(
    batch: BatchCreate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """Create a new batch."""
    audit = Audit(session)

    try:
        result = await session.execute(select(Batch).where(Batch.batch_id == batch.batch_id))
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Batch '{batch.batch_id}' already exists"
            )

        new_batch = Batch(
            batch_id=batch.batch_id,
            project_id=batch.project_id,
            status=batch.status,
            created_by=current_user["email"]
        )

        session.add(new_batch)
        await session.commit()
        await session.refresh(new_batch)

        await audit.log(
            "BATCH_CREATED",
            current_user["email"],
            batch.batch_id,
            f"Created batch {batch.batch_id}"
        )

        return new_batch

    finally:
        pass


@router.get("/batches", response_model=List[BatchResponse])
async def list_batches(limit: int = 100, offset: int = 0, session: AsyncSession = Depends(get_db_session)):
    """List batches."""
    result = await session.execute(select(Batch).limit(limit).offset(offset))
    batches = result.scalars().all()
    return batches


@router.get("/{sample_id}", response_model=SampleWithLineage)
async def get_sample(sample_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get sample details with lineage."""
    result = await session.execute(select(Sample).where(Sample.sample_id == sample_id))
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample '{sample_id}' not found"
        )

    # Get lineage
    parent_result = await session.execute(select(SampleLineage).where(SampleLineage.child_sample_id == sample.id))
    parent_lineages = parent_result.scalars().all()

    child_result = await session.execute(select(SampleLineage).where(SampleLineage.parent_sample_id == sample.id))
    child_lineages = child_result.scalars().all()

    parent_samples = []
    for lineage in parent_lineages:
        res = await session.execute(select(Sample).where(Sample.id == lineage.parent_sample_id))
        s = res.scalar_one_or_none()
        if s:
            parent_samples.append(s)

    child_samples = []
    for lineage in child_lineages:
        res = await session.execute(select(Sample).where(Sample.id == lineage.child_sample_id))
        s = res.scalar_one_or_none()
        if s:
            child_samples.append(s)

    return {
        **SampleResponse.model_validate(sample).model_dump(),
        "parent_samples": [SampleResponse.model_validate(p) for p in parent_samples],
        "child_samples": [SampleResponse.model_validate(c) for c in child_samples]
    }


@router.get("/", response_model=List[SampleResponse])
async def list_samples(
    status: Optional[str] = None,
    project_id: Optional[str] = None,
    container_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_db_session)
):
    """List samples with optional filtering."""
    stmt = select(Sample)

    if status:
        stmt = stmt.where(Sample.status == status)
    if project_id:
        # Look up project UUID from project_id string
        proj_res = await session.execute(select(Project).where(Project.project_id == project_id))
        project = proj_res.scalar_one_or_none()
        if project:
            stmt = stmt.where(Sample.project_id == project.id)
    if container_id:
        cont_res = await session.execute(select(Container).where(Container.container_id == container_id))
        container = cont_res.scalar_one_or_none()
        if container:
            stmt = stmt.where(Sample.container_id == container.id)

    stmt = stmt.order_by(Sample.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    samples = result.scalars().all()
    return samples


@router.put("/{sample_id}", response_model=SampleResponse)
async def update_sample(
    sample_id: str,
    update: SampleUpdate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """Update sample properties."""
    audit = Audit(session)

    try:
        result = await session.execute(select(Sample).where(Sample.sample_id == sample_id))
        sample = result.scalar_one_or_none()
        if not sample:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample '{sample_id}' not found"
            )

        # Update fields
        if update.status is not None:
            sample.status = update.status
        if update.container_id is not None:
            sample.container_id = update.container_id
        if update.lot_number is not None:
            sample.lot_number = update.lot_number
        if update.extra_data is not None:
            sample.extra_data = {**sample.extra_data, **update.extra_data}

        sample.updated_by = current_user["email"]
        sample.updated_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(sample)

        # Audit log
        await audit.log(
            "SAMPLE_UPDATED",
            current_user["email"],
            sample_id,
            f"Updated sample {sample_id}: {update.model_dump(exclude_none=True)}"
        )

        return sample

    finally:
        pass


@router.delete(
    "/{sample_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_role("admin"))]
)
async def delete_sample(
    sample_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """Delete (soft delete) a sample."""
    audit = Audit(session)

    try:
        result = await session.execute(select(Sample).where(Sample.sample_id == sample_id))
        sample = result.scalar_one_or_none()
        if not sample:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample '{sample_id}' not found"
            )

        # Soft delete - mark as disposed
        sample.status = 'disposed'
        sample.updated_by = current_user["email"]
        sample.updated_at = datetime.now(timezone.utc)

        await session.commit()

        # Audit log
        await audit.log(
            "SAMPLE_DELETED",
            current_user["email"],
            sample_id,
            f"Soft deleted sample {sample_id}"
        )

    finally:
        pass


@router.post("/{sample_id}/split")
async def split_sample(
    sample_id: str,
    child_sample_ids: List[str],
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """Create child samples from a parent sample (aliquoting)."""
    audit = Audit(session)

    try:
        result = await session.execute(select(Sample).where(Sample.sample_id == sample_id))
        parent = result.scalar_one_or_none()
        if not parent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent sample '{sample_id}' not found"
            )

        created_samples = []

        for child_id in child_sample_ids:
            # Create child sample
            child = Sample(
                sample_id=child_id,
                lot_number=parent.lot_number,
                project_id=parent.project_id,
                parent_sample_id=parent.id,
                batch_id=parent.batch_id,
                status='active',
                extra_data={'derived_from': sample_id},
                created_by=current_user["email"]
            )
            session.add(child)
            await session.flush()

            # Create lineage
            lineage = SampleLineage(
                parent_sample_id=parent.id,
                child_sample_id=child.id,
                relationship_type='split',
                created_by=current_user["email"]
            )
            session.add(lineage)

            created_samples.append(child)

        await session.commit()

        # Audit log
        await audit.log(
            "SAMPLE_SPLIT",
            current_user["email"],
            sample_id,
            f"Split sample {sample_id} into {len(child_sample_ids)} aliquots"
        )

        return {
            "message": f"Created {len(created_samples)} child samples",
            "child_samples": [SampleResponse.model_validate(s) for s in created_samples]
        }

    finally:
        pass


@router.post("/{sample_id}/assign-workflow")
async def assign_sample_to_workflow(
    sample_id: str,
    workflow_execution_id: UUID4,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """Assign sample to a workflow execution."""
    audit = Audit(session)

    try:
        result = await session.execute(select(Sample).where(Sample.sample_id == sample_id))
        sample = result.scalar_one_or_none()
        if not sample:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample '{sample_id}' not found"
            )

        assignment = WorkflowSampleAssignment(
            workflow_execution_id=workflow_execution_id,
            sample_id=sample.id,
            assigned_by=current_user["email"]
        )

        session.add(assignment)
        await session.commit()

        # Audit log
        await audit.log(
            "SAMPLE_WORKFLOW_ASSIGNED",
            current_user["email"],
            sample_id,
            f"Assigned sample {sample_id} to workflow {workflow_execution_id}"
        )

        return {"message": "Sample assigned to workflow successfully"}

    finally:
        pass


@router.get("/{sample_id}/history")
async def get_sample_history(sample_id: str, session: AsyncSession = Depends(get_db_session)):
    """Get audit history for a sample."""
    audit = Audit(session)

    # Get all audit logs for this sample
    logs = await audit.get_logs(limit=1000)

    # Filter logs related to this sample
    sample_logs = [
        log for log in logs
        if log.get('subject') == sample_id or sample_id in log.get('details', '')
    ]

    return sample_logs


# ============================================================================
# CONTAINER ENDPOINTS
# ============================================================================

