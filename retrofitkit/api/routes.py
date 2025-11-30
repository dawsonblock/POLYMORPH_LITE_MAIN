from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from retrofitkit.api.dependencies import get_current_user, get_db
from retrofitkit.core.workflow.runner import workflow_runner, WorkflowStatus
from retrofitkit.core.models import AuditLog
from retrofitkit.config import settings
import uuid

router = APIRouter()

class RunRequest(BaseModel):
    workflow_version_id: str
    context: Dict[str, Any] = {}

class RunResponse(BaseModel):
    run_id: str
    status: str

@router.get("/status")
async def status_endpoint(user=Depends(get_current_user)):
    """Get current system status."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
        "user": user
    }

@router.post("/workflows/run", response_model=RunResponse)
async def run_workflow(
    payload: RunRequest, 
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Start a new workflow execution."""
    try:
        # In a real app, 'user' would be a User object. Here it's a dict or mock.
        operator = user.get("email") if isinstance(user, dict) else user.email
        
        run_id = await workflow_runner.start_workflow(
            workflow_version_id=payload.workflow_version_id,
            context=payload.context,
            operator=operator
        )
        return {"run_id": run_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{run_id}/pause")
async def pause_workflow(run_id: str, user=Depends(get_current_user)):
    await workflow_runner.pause_workflow(run_id)
    return {"status": "paused"}

@router.post("/workflows/{run_id}/resume")
async def resume_workflow(run_id: str, user=Depends(get_current_user)):
    await workflow_runner.resume_workflow(run_id)
    return {"status": "resumed"}

@router.post("/workflows/{run_id}/cancel")
async def cancel_workflow(run_id: str, user=Depends(get_current_user)):
    await workflow_runner.cancel_workflow(run_id, reason=f"Cancelled by {user.get('email', 'unknown')}")
    return {"status": "cancelled"}

@router.get("/workflows/{run_id}")
async def get_workflow_state(run_id: str, user=Depends(get_current_user)):
    state = await workflow_runner.get_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    return state

