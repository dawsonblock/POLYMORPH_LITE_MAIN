"""
Workflow Engine Runner for POLYMORPH v8.0.

This module implements the core workflow execution engine with support for:
- State management (Running, Paused, Completed, Failed, Cancelled)
- Step-level hooks (on_start, on_complete, on_error)
- Retry policies
- Conditional branching
- Human-in-the-loop confirmation
- Persistence and recovery
"""

import time
import logging
import asyncio
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Models ---

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
    next_step_id: Optional[str] = None # Default next
    next_step_map: Optional[Dict[str, str]] = None # For conditions: {"true": "step_a", "false": "step_b"}
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
        self._runs: Dict[str, WorkflowState] = {}
        self._definitions: Dict[str, WorkflowDefinition] = {}
        self._action_handlers: Dict[str, Callable] = {}
        self._event_hooks: List[Callable[[WorkflowState], Awaitable[None]]] = []

    def register_action(self, name: str, handler: Callable):
        """Register a function to handle a specific action type."""
        self._action_handlers[name] = handler

    def register_hook(self, hook: Callable[[WorkflowState], Awaitable[None]]):
        """Register a hook to be called on state changes."""
        self._event_hooks.append(hook)

    def load_definition(self, definition: WorkflowDefinition):
        self._definitions[definition.id] = definition

    async def start_workflow(self, workflow_id: str, context: Dict[str, Any] = {}) -> str:
        """Start a new workflow instance."""
        if workflow_id not in self._definitions:
            raise ValueError(f"Workflow {workflow_id} not found")

        defn = self._definitions[workflow_id]
        run_id = str(uuid.uuid4())
        
        state = WorkflowState(
            run_id=run_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            current_step_id=defn.start_step_id,
            context=context,
            start_time=time.time()
        )
        self._runs[run_id] = state
        
        # Start execution loop in background
        asyncio.create_task(self._execute_loop(run_id))
        
        await self._notify_hooks(state)
        return run_id

    async def pause_workflow(self, run_id: str):
        state = self._get_run(run_id)
        if state.status == WorkflowStatus.RUNNING:
            state.status = WorkflowStatus.PAUSED
            await self._notify_hooks(state)

    async def resume_workflow(self, run_id: str):
        state = self._get_run(run_id)
        if state.status == WorkflowStatus.PAUSED:
            state.status = WorkflowStatus.RUNNING
            asyncio.create_task(self._execute_loop(run_id))
            await self._notify_hooks(state)

    async def cancel_workflow(self, run_id: str, reason: str = "Cancelled by user"):
        state = self._get_run(run_id)
        if state.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED, WorkflowStatus.WAITING_FOR_INPUT]:
            state.status = WorkflowStatus.CANCELLED
            state.error = reason
            state.end_time = time.time()
            await self._notify_hooks(state)

    async def submit_input(self, run_id: str, step_id: str, data: Dict[str, Any]):
        """Submit input for a HUMAN_INPUT step."""
        state = self._get_run(run_id)
        if state.status != WorkflowStatus.WAITING_FOR_INPUT:
            raise RuntimeError("Workflow is not waiting for input")
        
        if state.current_step_id != step_id:
            raise RuntimeError(f"Input submitted for wrong step. Current: {state.current_step_id}")

        # Update context with input
        state.context.update(data)
        state.status = WorkflowStatus.RUNNING
        
        # Resume execution
        asyncio.create_task(self._execute_loop(run_id))
        await self._notify_hooks(state)

    def get_state(self, run_id: str) -> WorkflowState:
        return self._get_run(run_id)

    # --- Internal Execution Loop ---

    async def _execute_loop(self, run_id: str):
        state = self._get_run(run_id)
        defn = self._definitions[state.workflow_id]
        steps_map = {s.id: s for s in defn.steps}

        while state.status == WorkflowStatus.RUNNING:
            if not state.current_step_id:
                # End of workflow
                state.status = WorkflowStatus.COMPLETED
                state.end_time = time.time()
                await self._notify_hooks(state)
                break

            step = steps_map.get(state.current_step_id)
            if not step:
                state.status = WorkflowStatus.FAILED
                state.error = f"Step {state.current_step_id} not found"
                await self._notify_hooks(state)
                break

            try:
                # Execute Step
                logger.info(f"Executing step: {step.name} ({step.type})")
                
                if step.type == StepType.HUMAN_INPUT:
                    state.status = WorkflowStatus.WAITING_FOR_INPUT
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
                    # Evaluate condition
                    # Simple eval for now: context variable check
                    var = step.params.get("variable")
                    val = state.context.get(var)
                    result = str(val).lower() # "true", "false", or value

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
                
                # Check for pause signal (between steps)
                if state.status == WorkflowStatus.PAUSED:
                    return

            except Exception as e:
                logger.error(f"Workflow Error: {e}")
                state.status = WorkflowStatus.FAILED
                state.error = str(e)
                state.end_time = time.time()
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

    def _get_run(self, run_id: str) -> WorkflowState:
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        return self._runs[run_id]

    async def _notify_hooks(self, state: WorkflowState):
        for hook in self._event_hooks:
            try:
                await hook(state)
            except Exception as e:
                logger.error(f"Error in workflow hook: {e}")

# Global Runner Instance
workflow_runner = WorkflowRunner()
