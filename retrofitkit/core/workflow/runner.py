import time
import logging
import asyncio
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Awaitable
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from retrofitkit.core.models import WorkflowExecution, WorkflowVersion, AuditLog
from retrofitkit.core.database import get_db_session
from retrofitkit.config import settings

logger = logging.getLogger(__name__)

# --- Models (Pydantic for API/Runtime) ---

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepType(str, Enum):
    ACTION = "action"
    CONDITION = "condition"
    WAIT = "wait"
    HUMAN_INPUT = "human_input"
    SUB_WORKFLOW = "sub_workflow"

class RetryPolicy(BaseModel):
    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0

class WorkflowStep(BaseModel):
    id: str
    type: StepType
    name: str
    action: str # Function name or command
    params: Dict[str, Any] = {}
    retry_policy: Optional[RetryPolicy] = None
    next_step_id: Optional[str] = None
    next_step_map: Optional[Dict[str, str]] = None
    timeout_seconds: Optional[float] = None

class WorkflowDefinition(BaseModel):
    id: str
    name: str
    version: str
    steps: List[WorkflowStep]
    start_step_id: str

class WorkflowState(BaseModel):
    run_id: str
    workflow_id: str
    status: WorkflowStatus
    current_step_id: Optional[str]
    context: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    error: Optional[str] = None
    start_time: float
    end_time: Optional[float] = None

# --- Engine ---

class WorkflowRunner:
    def __init__(self):
        self._action_handlers: Dict[str, Callable] = {}
        self._event_hooks: List[Callable[[WorkflowState], Awaitable[None]]] = []

    def register_action(self, name: str, handler: Callable):
        """Register a function to handle a specific action type."""
        self._action_handlers[name] = handler

    def register_hook(self, hook: Callable[[WorkflowState], Awaitable[None]]):
        """Register a hook to be called on state changes."""
        self._event_hooks.append(hook)

    async def start_workflow(self, workflow_version_id: str, context: Dict[str, Any] = {}, operator: str = "system") -> str:
        """Start a new workflow instance."""
        run_id = str(uuid.uuid4())
        
        async with get_db_session() as session:
            # Verify workflow exists
            stmt = select(WorkflowVersion).where(WorkflowVersion.id == workflow_version_id)
            result = await session.execute(stmt)
            wf_version = result.scalar_one_or_none()
            
            if not wf_version:
                raise ValueError(f"Workflow Version {workflow_version_id} not found")

            defn = WorkflowDefinition(**wf_version.definition)

            # Create Execution Record
            execution = WorkflowExecution(
                id=str(uuid.uuid4()),
                run_id=run_id,
                workflow_version_id=workflow_version_id,
                status=WorkflowStatus.RUNNING,
                operator=operator,
                results={"context": context, "history": [], "current_step_id": defn.start_step_id}
            )
            session.add(execution)
            
            # Audit Log
            audit = AuditLog(
                event="WORKFLOW_STARTED",
                actor=operator,
                subject=run_id,
                details=f"Started workflow {wf_version.workflow_name} v{wf_version.version}",
                hash=str(uuid.uuid4()) # Placeholder for real hash
            )
            session.add(audit)
            await session.commit()

            # Initialize State Object
            state = WorkflowState(
                run_id=run_id,
                workflow_id=workflow_version_id,
                status=WorkflowStatus.RUNNING,
                current_step_id=defn.start_step_id,
                context=context,
                start_time=time.time()
            )

        # Start execution loop in background
        asyncio.create_task(self._execute_loop(run_id))
        
        await self._notify_hooks(state)
        return run_id

    async def pause_workflow(self, run_id: str):
        async with get_db_session() as session:
            stmt = select(WorkflowExecution).where(WorkflowExecution.run_id == run_id)
            result = await session.execute(stmt)
            execution = result.scalar_one_or_none()
            
            if execution and execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.PAUSED
                await session.commit()
                # Hook notification would need state reconstruction here

    async def resume_workflow(self, run_id: str):
        async with get_db_session() as session:
            stmt = select(WorkflowExecution).where(WorkflowExecution.run_id == run_id)
            result = await session.execute(stmt)
            execution = result.scalar_one_or_none()
            
            if execution and execution.status == WorkflowStatus.PAUSED:
                execution.status = WorkflowStatus.RUNNING
                await session.commit()
                asyncio.create_task(self._execute_loop(run_id))

    async def cancel_workflow(self, run_id: str, reason: str = "Cancelled by user"):
        async with get_db_session() as session:
            stmt = select(WorkflowExecution).where(WorkflowExecution.run_id == run_id)
            result = await session.execute(stmt)
            execution = result.scalar_one_or_none()
            
            if execution and execution.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED, WorkflowStatus.WAITING_FOR_INPUT]:
                execution.status = WorkflowStatus.CANCELLED
                execution.error_message = reason
                execution.completed_at = datetime.utcnow()
                await session.commit()

    async def submit_input(self, run_id: str, step_id: str, data: Dict[str, Any]):
        """Submit input for a HUMAN_INPUT step."""
        async with get_db_session() as session:
            stmt = select(WorkflowExecution).where(WorkflowExecution.run_id == run_id)
            result = await session.execute(stmt)
            execution = result.scalar_one_or_none()
            
            if not execution:
                raise ValueError("Execution not found")

            if execution.status != WorkflowStatus.WAITING_FOR_INPUT:
                raise RuntimeError("Workflow is not waiting for input")
            
            current_results = execution.results or {}
            if current_results.get("current_step_id") != step_id:
                raise RuntimeError(f"Input submitted for wrong step.")

            # Update context
            current_results["context"].update(data)
            execution.results = current_results
            execution.status = WorkflowStatus.RUNNING
            await session.commit()
        
        # Resume execution
        asyncio.create_task(self._execute_loop(run_id))

    async def get_state(self, run_id: str) -> Optional[WorkflowState]:
        async with get_db_session() as session:
            stmt = select(WorkflowExecution).where(WorkflowExecution.run_id == run_id)
            result = await session.execute(stmt)
            execution = result.scalar_one_or_none()
            
            if not execution:
                return None
            
            res = execution.results or {}
            return WorkflowState(
                run_id=execution.run_id,
                workflow_id=execution.workflow_version_id,
                status=WorkflowStatus(execution.status),
                current_step_id=res.get("current_step_id"),
                context=res.get("context", {}),
                history=res.get("history", []),
                error=execution.error_message,
                start_time=execution.started_at.timestamp() if execution.started_at else 0
            )

    # --- Internal Execution Loop ---

    async def _execute_loop(self, run_id: str):
        # Load initial state
        state = await self.get_state(run_id)
        if not state:
            return

        async with get_db_session() as session:
            # Get Definition
            stmt = select(WorkflowVersion).where(WorkflowVersion.id == state.workflow_id)
            result = await session.execute(stmt)
            wf_version = result.scalar_one_or_none()
            if not wf_version:
                logger.error(f"Workflow version {state.workflow_id} not found for run {run_id}")
                return
            
            defn = WorkflowDefinition(**wf_version.definition)
            steps_map = {s.id: s for s in defn.steps}

            # Re-fetch execution to lock/update
            stmt_exec = select(WorkflowExecution).where(WorkflowExecution.run_id == run_id)
            res_exec = await session.execute(stmt_exec)
            execution = res_exec.scalar_one_or_none()

            while state.status == WorkflowStatus.RUNNING:
                if not state.current_step_id:
                    # End of workflow
                    state.status = WorkflowStatus.COMPLETED
                    state.end_time = time.time()
                    
                    execution.status = WorkflowStatus.COMPLETED
                    execution.completed_at = datetime.utcnow()
                    execution.results = state.dict() # Persist final state
                    await session.commit()
                    
                    await self._notify_hooks(state)
                    break

                step = steps_map.get(state.current_step_id)
                if not step:
                    state.status = WorkflowStatus.FAILED
                    state.error = f"Step {state.current_step_id} not found"
                    
                    execution.status = WorkflowStatus.FAILED
                    execution.error_message = state.error
                    await session.commit()
                    
                    await self._notify_hooks(state)
                    break

                try:
                    # Execute Step
                    logger.info(f"Executing step: {step.name} ({step.type})")
                    
                    if step.type == StepType.HUMAN_INPUT:
                        state.status = WorkflowStatus.WAITING_FOR_INPUT
                        
                        execution.status = WorkflowStatus.WAITING_FOR_INPUT
                        execution.results = state.dict() # Save state before waiting
                        await session.commit()
                        
                        await self._notify_hooks(state)
                        return # Exit loop, wait for submit_input

                    elif step.type == StepType.WAIT:
                        seconds = step.params.get("seconds", 1.0)
                        await asyncio.sleep(seconds)
                        result = None

                    elif step.type == StepType.ACTION:
                        handler = self._action_handlers.get(step.action)
                        if not handler:
                            raise RuntimeError(f"Action handler '{step.action}' not registered")
                        
                        # Execute with retry policy
                        result = await self._execute_with_retry(handler, step, state.context)
                        if isinstance(result, dict):
                            state.context.update(result)

                    elif step.type == StepType.CONDITION:
                        var = step.params.get("variable")
                        val = state.context.get(var)
                        result = str(val).lower()

                    # Determine Next Step
                    next_id = step.next_step_id
                    if step.type == StepType.CONDITION and step.next_step_map:
                        next_id = step.next_step_map.get(result, step.next_step_id)

                    # Record History
                    state.history.append({
                        "step_id": step.id,
                        "status": "completed",
                        "timestamp": time.time(),
                        "result": str(result)
                    })

                    # Move to next
                    state.current_step_id = next_id
                    
                    # Persist Progress
                    execution.results = state.dict()
                    await session.commit()
                    
                    # Check for pause signal (re-fetch status)
                    await session.refresh(execution)
                    if execution.status == WorkflowStatus.PAUSED:
                        state.status = WorkflowStatus.PAUSED
                        return

                except Exception as e:
                    logger.error(f"Workflow Error: {e}")
                    state.status = WorkflowStatus.FAILED
                    state.error = str(e)
                    state.end_time = time.time()
                    
                    execution.status = WorkflowStatus.FAILED
                    execution.error_message = str(e)
                    execution.completed_at = datetime.utcnow()
                    await session.commit()
                    
                    await self._notify_hooks(state)
                    break

    async def _execute_with_retry(self, handler: Callable, step: WorkflowStep, context: Dict) -> Any:
        attempts = 0
        policy = step.retry_policy or RetryPolicy(max_attempts=1)
        
        while attempts < policy.max_attempts:
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(context, step.params)
                else:
                    return handler(context, step.params)
            except Exception as e:
                attempts += 1
                if attempts >= policy.max_attempts:
                    raise e
                
                delay = policy.delay_seconds * (policy.backoff_factor ** (attempts - 1))
                logger.warning(f"Step failed, retrying in {delay}s... ({attempts}/{policy.max_attempts})")
                await asyncio.sleep(delay)

    async def _notify_hooks(self, state: WorkflowState):
        for hook in self._event_hooks:
            try:
                await hook(state)
            except Exception as e:
                logger.error(f"Error in workflow hook: {e}")

# Global Runner Instance
workflow_runner = WorkflowRunner()
