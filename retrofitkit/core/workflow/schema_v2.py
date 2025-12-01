from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid

class WorkflowStepInput(BaseModel):
    name: str
    type: Literal["string", "number", "boolean", "select", "file"]
    label: str
    required: bool = True
    options: Optional[List[str]] = None # For select type
    default: Optional[Any] = None

class WorkflowStep(BaseModel):
    id: str
    name: str
    type: Literal["action", "approval", "input", "checkpoint"]
    action: Optional[str] = None # e.g., "drivers.raman.acquire"
    params: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[WorkflowStepInput] = Field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    on_failure: Literal["abort", "retry", "skip"] = "abort"
    
    # v2 Features
    pre_hook: Optional[str] = None # e.g. "hooks.check_calibration"
    post_hook: Optional[str] = None
    approval_role: Optional[str] = None # For approval steps

class WorkflowDefinition(BaseModel):
    id: str
    version: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    steps: List[WorkflowStep]
    schema_version: str = "2.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deprecated: bool = False

class WorkflowContext(BaseModel):
    run_id: str
    workflow_id: str
    user_id: str
    start_time: datetime
    variables: Dict[str, Any] = Field(default_factory=dict)
    step_results: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["running", "paused", "completed", "failed", "cancelled"] = "running"
    current_step_index: int = 0
