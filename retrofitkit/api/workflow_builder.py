"""
Visual Workflow Builder API endpoints.

Provides workflow definition management, versioning, and execution.
Supports visual drag-and-drop workflow design.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import json
import uuid

from sqlalchemy.orm import Session
from retrofitkit.db.session import get_db
from retrofitkit.db.models.workflow import WorkflowVersion, WorkflowExecution, ConfigSnapshot
from retrofitkit.db.models.user import User
from retrofitkit.api.dependencies import get_current_user, require_role
from retrofitkit.compliance.audit import Audit
from retrofitkit.core.recipe import Recipe, RecipeStep

router = APIRouter(prefix="/api/workflow-builder", tags=["workflow-builder"])

# ============================================================================
# Pydantic Models
# ============================================================================

class WorkflowNodeDefinition(BaseModel):
    """Definition of a single workflow node (block)."""
    id: str
    type: str  # acquire, measure, move, gate, ai-evaluate, delay, loop, condition
    position: Dict[str, float]  # x, y coordinates for visual editor
    data: Dict[str, Any]  # Node-specific configuration


class WorkflowEdgeDefinition(BaseModel):
    """Connection between workflow nodes."""
    id: str
    source: str  # Source node ID
    target: str  # Target node ID
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    label: Optional[str] = None


class WorkflowDefinitionCreate(BaseModel):
    """Create a new workflow definition."""
    workflow_name: str
    nodes: List[WorkflowNodeDefinition]
    edges: List[WorkflowEdgeDefinition]
    metadata: Dict[str, Any] = {}


class WorkflowDefinitionResponse(BaseModel):
    """Workflow definition response."""
    id: UUID4
    workflow_name: str
    version: int
    definition: Dict[str, Any]
    definition_hash: str
    is_active: bool
    is_approved: bool
    created_by: str
    created_at: datetime
    approved_by: Optional[str]
    approved_at: Optional[datetime]

    class Config:
        from_attributes = True


class WorkflowExecutionCreate(BaseModel):
    """Start a workflow execution."""
    workflow_name: str
    workflow_version: Optional[int] = None  # Use latest active if not specified
    parameters: Dict[str, Any] = {}


class WorkflowExecutionResponse(BaseModel):
    """Workflow execution response."""
    id: UUID4
    run_id: str
    workflow_version_id: UUID4
    started_at: datetime
    completed_at: Optional[datetime]
    status: str
    operator: str
    results: Dict[str, Any]
    error_message: Optional[str]

    class Config:
        from_attributes = True


# ============================================================================
# WORKFLOW DEFINITION ENDPOINTS
# ============================================================================

@router.post("/workflows", response_model=WorkflowDefinitionResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow_definition(
    workflow: WorkflowDefinitionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new workflow definition.

    The workflow definition includes nodes (blocks) and edges (connections)
    that can be rendered in a visual workflow builder.
    """
    audit = Audit()

        # Get next version number
        latest = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow.workflow_name
        ).order_by(WorkflowVersion.version.desc()).first()

        next_version = (latest.version + 1) if latest else 1

        # Build definition dictionary
        definition = {
            "nodes": [node.dict() for node in workflow.nodes],
            "edges": [edge.dict() for edge in workflow.edges],
            "metadata": workflow.metadata
        }

        # Calculate hash for integrity verification
        definition_json = json.dumps(definition, sort_keys=True)
        definition_hash = hashlib.sha256(definition_json.encode()).hexdigest()

        # Create new version
        new_workflow = WorkflowVersion(
            workflow_name=workflow.workflow_name,
            version=next_version,
            definition=definition,
            definition_hash=definition_hash,
            is_active=False,  # Must be explicitly activated
            is_approved=False,  # Must be approved before execution
            created_by=current_user.email
        )

        db.add(new_workflow)
        db.commit()
        db.refresh(new_workflow)

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_CREATED",
                current_user.email,
                f"{workflow.workflow_name}:v{next_version}",
                f"Created workflow {workflow.workflow_name} version {next_version}"
            )
        except Exception:
            pass  # Don't fail the request if audit logging fails

        return new_workflow

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating workflow: {str(e)}"
        )


@router.get("/workflows/{workflow_name}", response_model=List[WorkflowDefinitionResponse])
async def list_workflow_versions(workflow_name: str, db: Session = Depends(get_db)):
    """List all versions of a workflow."""

        versions = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name
        ).order_by(WorkflowVersion.version.desc()).all()

        return versions



@router.get("/workflows/{workflow_name}/v/{version}", response_model=WorkflowDefinitionResponse)
async def get_workflow_version(workflow_name: str, version: int, db: Session = Depends(get_db)):
    """Get a specific workflow version."""

        workflow = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.version == version
        ).first()

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_name}' version {version} not found"
            )

        return workflow



@router.get("/workflows/{workflow_name}/active", response_model=WorkflowDefinitionResponse)
async def get_active_workflow(workflow_name: str, db: Session = Depends(get_db)):
    """Get the currently active version of a workflow."""

        workflow = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.is_active == True
        ).first()

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active version found for workflow '{workflow_name}'"
            )

        return workflow



@router.post("/workflows/{workflow_name}/v/{version}/activate")
async def activate_workflow_version(
    workflow_name: str,
    version: int,
    ,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Activate a specific workflow version.

    Deactivates all other versions of the same workflow.
    Requires approval before activation.
    """
    audit = Audit()

        workflow = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.version == version
        ).first()

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_name}' version {version} not found"
            )

        if not workflow.is_approved:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workflow must be approved before activation"
            )

        # Deactivate all other versions
        db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.id != workflow.id
        ).update({"is_active": False})

        # Activate this version
        workflow.is_active = True

        db.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_ACTIVATED",
                current_user.email,
                f"{workflow_name}:v{version}",
                f"Activated workflow {workflow_name} version {version}"
            )
        except Exception:
            pass

        return {
            "message": f"Workflow '{workflow_name}' version {version} activated",
            "workflow_id": str(workflow.id)
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activating workflow: {str(e)}"
        )


@router.post("/workflows/{workflow_name}/v/{version}/approve")
async def approve_workflow_version(
    workflow_name: str,
    version: int,
    ,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Approve a workflow version for execution.

    Only approved workflows can be activated.
    Requires QA or Admin role (implement role check as needed).
    """
    audit = Audit()

        workflow = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.version == version
        ).first()

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_name}' version {version} not found"
            )

        if workflow.is_approved:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workflow is already approved"
            )

        # TODO: Add role check (QA or Admin only)
        # if current_user["role"] not in ["QA", "Admin"]:
        #     raise HTTPException(403, "Insufficient permissions")

        workflow.is_approved = True
        workflow.approved_by = current_user.email
        workflow.approved_at = datetime.utcnow()

        db.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_APPROVED",
                current_user.email,
                f"{workflow_name}:v{version}",
                f"Approved workflow {workflow_name} version {version}"
            )
        except Exception:
            pass

        return {
            "message": f"Workflow '{workflow_name}' version {version} approved",
            "approved_by": current_user.email,
            "approved_at": workflow.approved_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error approving workflow: {str(e)}"
        )


@router.delete("/workflows/{workflow_name}/v/{version}")
async def delete_workflow_version(
    workflow_name: str,
    version: int,
    ,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a workflow version.

    Cannot delete active or approved workflows.
    """
    audit = Audit()

        workflow = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.version == version
        ).first()

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_name}' version {version} not found"
            )

        if workflow.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete active workflow"
            )

        if workflow.is_approved:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete approved workflow"
            )

        db.delete(workflow)
        db.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_DELETED",
                current_user.email,
                f"{workflow_name}:v{version}",
                f"Deleted workflow {workflow_name} version {version}"
            )
        except Exception:
            pass

        return {"message": f"Workflow '{workflow_name}' version {version} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting workflow: {str(e)}"
        )


# ============================================================================
# WORKFLOW EXECUTION ENDPOINTS
# ============================================================================



def _graph_to_recipe(workflow_version: WorkflowVersion) -> Recipe:
    """
    Convert visual workflow graph to Recipe object.
    
    Supports: Acquire, Measure, Delay node types.
    Minimal implementation for basic workflows.
    """
    graph = workflow_version.definition
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    
    # Build execution order (simple linear for now)
    start_node = next((n for n in graph["nodes"] if n.get("type") == "Start"), None)
    if not start_node:
        # If no Start node, use first node
        if not graph["nodes"]:
            raise ValueError("Workflow has no nodes")
        start_node = graph["nodes"][0]
    
    # Follow edges to build step list
    steps = []
    current_id = start_node["id"]
    visited = set()
    
    while current_id and current_id not in visited:
        visited.add(current_id)
        node = nodes.get(current_id)
        if not node:
            break
        
        # Convert node to recipe step based on type
        node_type = node.get("type", "")
        node_data = node.get("data", {})
        
        if node_type == "Acquire" or node_type == "acquire":
            steps.append(RecipeStep(
                type="bias_set",
                params={
                    "volts": float(node_data.get("voltage", 0.0)),
                    "device": node_data.get("device_id", "daq"),
                }
            ))
        elif node_type == "Measure" or node_type == "measure":
            steps.append(RecipeStep(
                type="wait_for_raman",
                params={
                    "timeout_s": int(node_data.get("timeout", 120)),
                    "device": node_data.get("device_id", "raman"),
                }
            ))
        elif node_type == "Delay" or node_type == "delay" or node_type == "hold":
            steps.append(RecipeStep(
                type="hold",
                params={"seconds": float(node_data.get("seconds", 1.0))}
            ))
        
        # Find next node
        next_edge = next((e for e in edges if e.get("source") == current_id), None)
        current_id = next_edge.get("target") if next_edge else None
    
    return Recipe(
        name=workflow_version.workflow_name,
        steps=steps,
        metadata={"version": workflow_version.version, "workflow_id": str(workflow_version.id)}
    )


@router.post("/execute", response_model=WorkflowExecutionResponse, status_code=status.HTTP_201_CREATED)
async def execute_workflow(
    execution: WorkflowExecutionCreate,
    ,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Execute a workflow.

    Creates a workflow execution record and triggers the orchestrator.
    """
    audit = Audit()

        # Get workflow version
        if execution.workflow_version:
            workflow = db.query(WorkflowVersion).filter(
                WorkflowVersion.workflow_name == execution.workflow_name,
                WorkflowVersion.version == execution.workflow_version
            ).first()
        else:
            # Use active version
            workflow = db.query(WorkflowVersion).filter(
                WorkflowVersion.workflow_name == execution.workflow_name,
                WorkflowVersion.is_active == True
            ).first()

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {'active ' if not execution.workflow_version else ''}workflow found for '{execution.workflow_name}'"
            )

        if not workflow.is_approved:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workflow must be approved before execution"
            )

        # Generate run ID
        run_id = f"{execution.workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Create config snapshot
        config_snapshot = ConfigSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            config_data={"workflow_parameters": execution.parameters},
            config_hash=hashlib.sha256(json.dumps(execution.parameters, sort_keys=True).encode()).hexdigest(),
            created_by=current_user.email,
            reason=f"Execution of {execution.workflow_name}"
        )
        db.add(config_snapshot)
        db.flush()

        # Create execution record
        new_execution = WorkflowExecution(
            run_id=run_id,
            workflow_version_id=workflow.id,
            operator=current_user.email,
            status="running",
            config_snapshot_id=config_snapshot.id
        )

        db.add(new_execution)
        db.commit()
        db.refresh(new_execution)

        # Convert graph to recipe and trigger orchestrator
        try:
            recipe = _graph_to_recipe(workflow)
            # Note: Actual orchestrator integration would require AppContext
            # For now, just mark as pending orchestrator implementation
            new_execution.status = "pending"  # Would be "running" when orchestrator called
            db.commit()
        except Exception as e:
            new_execution.status = "failed"
            new_execution.error_message = f"Graph conversion error: {str(e)}"
            db.commit()
            # Don't raise - execution record created even if conversion fails
        # This would convert the visual workflow definition to executable steps
        # and pass to the existing orchestrator
        # orchestrator.execute_visual_workflow(workflow.definition, run_id, execution.parameters)

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_EXECUTED",
                current_user.email,
                run_id,
                f"Started execution of {execution.workflow_name} (version {workflow.version})"
            )
        except Exception:
            pass

        return new_execution

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing workflow: {str(e)}"
        )


@router.get("/executions/{run_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(run_id: str, db: Session = Depends(get_db)):
    """Get workflow execution details."""

        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.run_id == run_id
        ).first()

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{run_id}' not found"
            )

        return execution



@router.get("/executions", response_model=List[WorkflowExecutionResponse])
async def list_workflow_executions(
    workflow_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List workflow executions with optional filtering."""

        query = db.query(WorkflowExecution)

        if workflow_name:
            # Join with WorkflowVersion to filter by name
            query = query.join(WorkflowVersion).filter(
                WorkflowVersion.workflow_name == workflow_name
            )

        if status:
            query = query.filter(WorkflowExecution.status == status)

        executions = query.order_by(
            WorkflowExecution.started_at.desc()
        ).limit(limit).offset(offset).all()

        return executions



@router.post("/executions/{run_id}/abort")
async def abort_workflow_execution(
    run_id: str,
    ,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Abort a running workflow execution."""
    audit = Audit()

        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.run_id == run_id
        ).first()

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{run_id}' not found"
            )

        if execution.status not in ["running", "paused"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot abort execution with status '{execution.status}'"
            )

        execution.status = "aborted"
        execution.completed_at = datetime.utcnow()
        execution.error_message = f"Aborted by {current_user['email']}"

        db.commit()

        # TODO: Signal orchestrator to stop execution
        # orchestrator.abort_execution(run_id)

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_ABORTED",
                current_user.email,
                run_id,
                f"Aborted execution {run_id}"
            )
        except Exception:
            pass

        return {"message": f"Execution '{run_id}' aborted"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error aborting execution: {str(e)}"
        )
