"""
Visual Workflow Builder API endpoints.

Provides workflow definition management, versioning, and execution.
Supports visual drag-and-drop workflow design with orchestrator integration.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, UUID4, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import hashlib
import json
import uuid

from retrofitkit.database.models import (
    WorkflowVersion, WorkflowExecution, ConfigSnapshot,
    get_session
)
from retrofitkit.compliance.audit import Audit
from retrofitkit.api.dependencies import get_current_user
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

    model_config = ConfigDict(from_attributes=True)


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

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# WORKFLOW DEFINITION ENDPOINTS
# ============================================================================

@router.post("/workflows", response_model=WorkflowDefinitionResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow_definition(
    workflow: WorkflowDefinitionCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new workflow definition.

    The workflow definition includes nodes (blocks) and edges (connections)
    that can be rendered in a visual workflow builder.
    """
    session = get_session()
    audit = Audit()

    try:
        # Get next version number
        latest = session.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow.workflow_name
        ).order_by(WorkflowVersion.version.desc()).first()

        next_version = (int(latest.version) + 1) if latest else 1

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
            created_by=current_user["email"]
        )

        session.add(new_workflow)
        session.commit()
        session.refresh(new_workflow)

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_CREATED",
                current_user["email"],
                f"{workflow.workflow_name}:v{next_version}",
                f"Created workflow {workflow.workflow_name} version {next_version}"
            )
        except Exception:
            pass  # Don't fail the request if audit logging fails

        return new_workflow

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating workflow: {str(e)}"
        )
    finally:
        session.close()


@router.get("/workflows/{workflow_name}", response_model=List[WorkflowDefinitionResponse])
async def list_workflow_versions(workflow_name: str):
    """List all versions of a workflow."""
    session = get_session()

    try:
        versions = session.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name
        ).order_by(WorkflowVersion.version.desc()).all()

        return versions

    finally:
        session.close()


@router.get("/workflows/{workflow_name}/v/{version}", response_model=WorkflowDefinitionResponse)
async def get_workflow_version(workflow_name: str, version: int):
    """Get a specific workflow version."""
    session = get_session()

    try:
        workflow = session.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.version == version
        ).first()

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_name}' version {version} not found"
            )

        return workflow

    finally:
        session.close()


@router.get("/workflows/{workflow_name}/active", response_model=WorkflowDefinitionResponse)
async def get_active_workflow(workflow_name: str):
    """Get the currently active version of a workflow."""
    session = get_session()

    try:
        workflow = session.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.is_active == True
        ).first()

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active version found for workflow '{workflow_name}'"
            )

        return workflow

    finally:
        session.close()


@router.post("/workflows/{workflow_name}/v/{version}/activate")
async def activate_workflow_version(
    workflow_name: str,
    version: int,
    current_user: dict = Depends(get_current_user)
):
    """
    Activate a specific workflow version.

    Deactivates all other versions of the same workflow.
    Requires approval before activation.
    Requires QA or Admin role.
    """
    session = get_session()
    audit = Audit()

    try:
        # Enforce role check: QA or Admin only
        from retrofitkit.db.models.user import User as UserModel
        user_obj = session.query(UserModel).filter(UserModel.email == current_user["email"]).first()
        if not user_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Check user roles
        user_roles = [ur.role.name for ur in user_obj.roles] if hasattr(user_obj, 'roles') else []
        if not any(role in ["admin", "compliance", "qa"] for role in user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions. Only admin, compliance, or QA roles can activate workflows."
            )

        workflow = session.query(WorkflowVersion).filter(
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
        session.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_name == workflow_name,
            WorkflowVersion.id != workflow.id
        ).update({"is_active": False})

        # Activate this version
        workflow.is_active = True

        session.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_ACTIVATED",
                current_user["email"],
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
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activating workflow: {str(e)}"
        )
    finally:
        session.close()


@router.post("/workflows/{workflow_name}/v/{version}/approve")
async def approve_workflow_version(
    workflow_name: str,
    version: int,
    current_user: dict = Depends(get_current_user)
):
    """
    Approve a workflow version for execution.

    Only approved workflows can be activated.
    Requires QA or Admin role (implement role check as needed).
    """
    session = get_session()
    audit = Audit()

    try:
        workflow = session.query(WorkflowVersion).filter(
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

        # Enforce role check: QA or Admin only
        from retrofitkit.db.models.user import User as UserModel
        user_obj = session.query(UserModel).filter(UserModel.email == current_user["email"]).first()
        if not user_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Check user roles
        user_roles = [ur.role.name for ur in user_obj.roles] if hasattr(user_obj, 'roles') else []
        if not any(role in ["admin", "compliance"] for role in user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions. Only admin and compliance roles can approve workflows."
            )

        workflow.is_approved = True
        workflow.approved_by = current_user["email"]
        workflow.approved_at = datetime.now(timezone.utc)

        session.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_APPROVED",
                current_user["email"],
                f"{workflow_name}:v{version}",
                f"Approved workflow {workflow_name} version {version}"
            )
        except Exception:
            pass

        return {
            "message": f"Workflow '{workflow_name}' version {version} approved",
            "approved_by": current_user["email"],
            "approved_at": workflow.approved_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error approving workflow: {str(e)}"
        )
    finally:
        session.close()


@router.delete("/workflows/{workflow_name}/v/{version}")
async def delete_workflow_version(
    workflow_name: str,
    version: int,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a workflow version.

    Cannot delete active or approved workflows.
    """
    session = get_session()
    audit = Audit()

    try:
        workflow = session.query(WorkflowVersion).filter(
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

        session.delete(workflow)
        session.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_DELETED",
                current_user["email"],
                f"{workflow_name}:v{version}",
                f"Deleted workflow {workflow_name} version {version}"
            )
        except Exception:
            pass

        return {"message": f"Workflow '{workflow_name}' version {version} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting workflow: {str(e)}"
        )
    finally:
        session.close()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _graph_to_recipe(workflow_version: WorkflowVersion, parameters: Dict[str, Any]) -> Recipe:
    """
    Convert visual workflow graph to executable Recipe.
    
    Supports basic node types:
    - Acquire: Sets voltage/bias (→ bias_set step)
    - Measure: Waits for measurement (→ wait_for_raman step)
    - Delay/Hold: Time delay (→ hold step)
    
    Args:
        workflow_version: The workflow definition
        parameters: Runtime parameters to merge with node configs
    
    Returns:
        Recipe object ready for orchestrator execution
    """
    graph = workflow_version.definition
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    # Find start node
    start_node = next((n for n in graph["nodes"] if n.get("type", "").lower() == "start"), None)
    if not start_node:
        # Use first node if no explicit Start
        if not graph["nodes"]:
            raise ValueError("Workflow has no nodes")
        start_node = graph["nodes"][0]

    # Build execution order by following edges (simple linear for now)
    steps = []
    current_id = start_node["id"]
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = nodes.get(current_id)
        if not node:
            break

        node_type = node.get("type", "").lower()
        node_data = node.get("data", {})

        # Merge runtime parameters
        merged_params = {**node_data, **parameters.get(node["id"], {})}

        # Convert to recipe step based on type
        if node_type == "acquire":
            steps.append(RecipeStep(
                type="bias_set",
                params={
                    "volts": float(merged_params.get("voltage", 0.0)),
                    "device": merged_params.get("device_id", "daq"),
                }
            ))
        elif node_type == "measure":
            steps.append(RecipeStep(
                type="wait_for_raman",
                params={
                    "timeout_s": int(merged_params.get("timeout", 120)),
                    "device": merged_params.get("device_id", "raman"),
                }
            ))
        elif node_type in ["delay", "hold", "wait"]:
            steps.append(RecipeStep(
                type="hold",
                params={"seconds": float(merged_params.get("seconds", 1.0))}
            ))
        # Skip Start/End nodes - they're just UI markers

        # Find next node
        next_edge = next((e for e in edges if e.get("source") == current_id), None)
        current_id = next_edge.get("target") if next_edge else None

    return Recipe(
        name=workflow_version.workflow_name,
        steps=steps,
        metadata={
            "version": workflow_version.version,
            "workflow_id": str(workflow_version.id),
            "visual_graph": True
        }
    )


# ============================================================================
# WORKFLOW EXECUTION ENDPOINTS
# ============================================================================

@router.post("/execute", response_model=WorkflowExecutionResponse, status_code=status.HTTP_201_CREATED)
async def execute_workflow(
    execution: WorkflowExecutionCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Execute a workflow.

    Creates a workflow execution record and triggers the orchestrator.
    """
    session = get_session()
    audit = Audit()

    try:
        # Get workflow version
        if execution.workflow_version:
            workflow = session.query(WorkflowVersion).filter(
                WorkflowVersion.workflow_name == execution.workflow_name,
                WorkflowVersion.version == execution.workflow_version
            ).first()
        else:
            # Use active version
            workflow = session.query(WorkflowVersion).filter(
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
            timestamp=datetime.now(timezone.utc),
            config_data={"workflow_parameters": execution.parameters},
            config_hash=hashlib.sha256(json.dumps(execution.parameters, sort_keys=True).encode()).hexdigest(),
            created_by=current_user["email"],
            reason=f"Execution of {execution.workflow_name}"
        )
        session.add(config_snapshot)
        session.flush()

        # Create execution record
        new_execution = WorkflowExecution(
            run_id=run_id,
            workflow_version_id=workflow.id,
            operator=current_user["email"],
            status="running",
            config_snapshot_id=config_snapshot.id
        )

        session.add(new_execution)
        session.commit()
        session.refresh(new_execution)

        # Convert visual graph to executable recipe
        try:
            recipe = _graph_to_recipe(workflow, execution.parameters)

            # NOTE: Full orchestrator execution requires AppContext singleton from server.py
            # For now, we mark as 'ready' and document the recipe was built
            new_execution.status = "ready"
            new_execution.results = {
                "recipe_generated": True,
                "steps_count": len(recipe.steps),
                "step_types": [step.type for step in recipe.steps],
                "note": "Recipe generated successfully. Orchestrator execution requires AppContext integration."
            }
            session.commit()

            # TODO: When AppContext available, execute via:
            # from retrofitkit.api.server import app_context
            # result = app_context.orchestrator.execute_recipe(recipe, run_id)
            # Update new_execution.status based on result

        except Exception as e:
            # Failed to convert graph to recipe
            new_execution.status = "failed"
            new_execution.error_message = f"Graph conversion error: {str(e)}"
            session.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_EXECUTED",
                current_user["email"],
                run_id,
                f"Started execution of {execution.workflow_name} (version {workflow.version})"
            )
        except Exception:
            pass

        return new_execution

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing workflow: {str(e)}"
        )
    finally:
        session.close()


@router.get("/executions/{run_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(run_id: str):
    """Get workflow execution details."""
    session = get_session()

    try:
        execution = session.query(WorkflowExecution).filter(
            WorkflowExecution.run_id == run_id
        ).first()

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{run_id}' not found"
            )

        return execution

    finally:
        session.close()


@router.get("/executions", response_model=List[WorkflowExecutionResponse])
async def list_workflow_executions(
    workflow_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List workflow executions with optional filtering."""
    session = get_session()

    try:
        query = session.query(WorkflowExecution)

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

    finally:
        session.close()


@router.post("/executions/{run_id}/abort")
async def abort_workflow_execution(
    run_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Abort a running workflow execution."""
    session = get_session()
    audit = Audit()

    try:
        execution = session.query(WorkflowExecution).filter(
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
        execution.completed_at = datetime.now(timezone.utc)
        execution.error_message = f"Aborted by {current_user['email']}"

        session.commit()

        # TODO: Signal orchestrator to stop execution
        # orchestrator.abort_execution(run_id)

        # Audit log (non-blocking)
        try:
            audit.log(
                "WORKFLOW_ABORTED",
                current_user["email"],
                run_id,
                f"Aborted execution {run_id}"
            )
        except Exception:
            pass

        return {"message": f"Execution '{run_id}' aborted"}

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error aborting execution: {str(e)}"
        )
    finally:
        session.close()
