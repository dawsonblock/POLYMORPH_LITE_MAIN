"""
Workflow API endpoints for workflow management and execution.

Provides REST interface to:
- Upload and manage workflows
- Execute workflows with safety enforcement
- Query execution results
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path

from retrofitkit.core.workflows.models import WorkflowDefinition
from retrofitkit.core.workflows.engine import WorkflowEngine
from retrofitkit.core.workflows.safety import SafetyManager, LoggingPolicy


router = APIRouter(prefix="/workflows", tags=["workflows"])

# Global workflow storage (in-memory for now)
_workflows: Dict[str, WorkflowDefinition] = {}

# Global safety manager and engine
_safety_manager = SafetyManager()
_safety_manager.add_policy(LoggingPolicy())  # Default audit logging

_workflow_engine = WorkflowEngine(_safety_manager)


class WorkflowUploadRequest(BaseModel):
    """Request model for uploading workflow YAML."""
    yaml_content: str


class WorkflowListResponse(BaseModel):
    """Response model for workflow listing."""
    id: str
    name: str
    entry_step: str
    num_steps: int


class WorkflowExecuteResponse(BaseModel):
    """Response model for workflow execution."""
    workflow_id: str
    success: bool
    steps_executed: List[str]
    step_results: Dict[str, Any]
    error: Optional[str] = None
    duration_seconds: float


@router.get("/", response_model=List[WorkflowListResponse])
async def list_workflows():
    """
    List all available workflows.
    
    Returns:
        List of workflows with summary information
    """
    return [
        WorkflowListResponse(
            id=wf.id,
            name=wf.name,
            entry_step=wf.entry_step,
            num_steps=len(wf.steps),
        )
        for wf in _workflows.values()
    ]


@router.post("/", status_code=201)
async def upload_workflow(request: WorkflowUploadRequest):
    """
    Upload a new workflow from YAML.
    
    Args:
        request: Request containing YAML workflow definition
        
    Returns:
        Workflow summary
        
    Raises:
        400: If YAML is invalid
    """
    try:
        workflow = WorkflowDefinition.from_yaml(request.yaml_content)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workflow YAML: {str(e)}"
        )

    # Store workflow
    _workflows[workflow.id] = workflow

    return {
        "id": workflow.id,
        "name": workflow.name,
        "message": f"Workflow '{workflow.name}' uploaded successfully"
    }


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """
    Get workflow details.
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        Full workflow definition
        
    Raises:
        404: If workflow not found
    """
    workflow = _workflows.get(workflow_id)

    if workflow is None:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found"
        )

    return {
        "id": workflow.id,
        "name": workflow.name,
        "entry_step": workflow.entry_step,
        "steps": {
            step_id: {
                "kind": step.kind,
                "params": step.params,
                "children": step.children,
            }
            for step_id, step in workflow.steps.items()
        },
        "metadata": workflow.metadata,
    }


@router.post("/{workflow_id}/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(
    workflow_id: str,
    context: Dict[str, Any] = Body(default={})
):
    """
    Execute a workflow.
    
    Args:
        workflow_id: Workflow identifier
        context: Optional initial execution context
        
    Returns:
        Execution result with step results
        
    Raises:
        404: If workflow not found
    """
    workflow = _workflows.get(workflow_id)

    if workflow is None:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found"
        )

    # Execute workflow
    result = await _workflow_engine.run(workflow, context)

    return WorkflowExecuteResponse(
        workflow_id=result.workflow_id,
        success=result.success,
        steps_executed=result.steps_executed,
        step_results=result.step_results,
        error=result.error,
        duration_seconds=result.duration_seconds,
    )


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow.
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        Success message
        
    Raises:
        404: If workflow not found
    """
    if workflow_id not in _workflows:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found"
        )

    del _workflows[workflow_id]

    return {"message": f"Workflow '{workflow_id}' deleted"}


@router.get("/safety/policies")
async def list_safety_policies():
    """
    List active safety policies.
    
    Returns:
        List of policy names
    """
    return {"policies": _safety_manager.list_policies()}


@router.post("/safety/policies/{policy_name}/disable")
async def disable_safety_policy(policy_name: str):
    """
    Disable a safety policy.
    
    WARNING: Use with caution. Only for testing/emergency.
    
    Args:
        policy_name: Policy to disable
        
    Returns:
        Success message
    """
    _safety_manager.remove_policy(policy_name)

    return {"message": f"Policy '{policy_name}' disabled"}


# Load example workflows on startup
def _load_example_workflows():
    """Load example workflows from workflows/ directory."""
    workflows_dir = Path("workflows")

    if not workflows_dir.exists():
        return

    for yaml_file in workflows_dir.glob("*.yaml"):
        try:
            workflow = WorkflowDefinition.from_file(yaml_file)
            _workflows[workflow.id] = workflow
            print(f"Loaded workflow: {workflow.name} ({workflow.id})")
        except Exception as e:
            print(f"Warning: Failed to load {yaml_file}: {e}")


# Load examples on import
_load_example_workflows()
