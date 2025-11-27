"""
Workflow execution engine for POLYMORPH-4 Lite.

Executes multi-step workflows with device actions, timing control,
and safety policy enforcement.
"""
import asyncio
import time
from typing import Dict, Any, Optional

from .models import WorkflowDefinition, WorkflowStep, WorkflowExecutionResult
from .safety import SafetyManager
from retrofitkit.core.registry import registry


class WorkflowEngine:
    """
   Executes workflows with device orchestration and safety enforcement.
    
    Example:
        engine = WorkflowEngine(safety_manager)
        result = await engine.run(workflow, context={})
    """
    
    def __init__(self, safety: SafetyManager):
        """
        Initialize workflow engine.
        
        Args:
            safety: SafetyManager instance for policy enforcement
        """
        self._safety = safety
        self._device_instances: Dict[str, Any] = {}
        self.current_step_index = 0
        self.total_steps = 0
    
    async def run(
        self,
        workflow: WorkflowDefinition,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecutionResult:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow definition to execute
            context: Optional initial execution context
            
        Returns:
            WorkflowExecutionResult with execution details
        """
        start_time = time.time()
        ctx = {} if context is None else dict(context)
        steps_executed = []
        step_results = {}
        
        try:
            current_id = workflow.entry_step
            total_steps = len(workflow.steps)
            step_index = 0
            
            while current_id:
                step_index += 1
                # Update progress
                self.current_step_index = step_index
                self.total_steps = total_steps
                
                step = workflow.steps[current_id]
                steps_executed.append(current_id)
                
                step_result: Any = None # Initialize step_result for assignment below
                
                # Execute step based on kind
                if step.kind == "action":
                    step_result = await self._run_action(step, ctx)
                elif step.kind == "wait":
                    wait_seconds = float(step.params.get("seconds", 1.0))
                    await asyncio.sleep(wait_seconds)
                    step_result = {"waited_seconds": wait_seconds}
                else:
                    # Unsupported step kind
                    raise ValueError(
                        f"Unsupported workflow step kind: '{step.kind}'. "
                        f"Supported kinds: 'action', 'wait'. "
                        f"Loop and condition steps are planned for a future release."
                    )
                
                # Store result
                step_results[current_id] = step_result
                
                # Move to next step
                current_id = step.children[0] if step.children else ""
            
            duration = time.time() - start_time
            
            return WorkflowExecutionResult(
                workflow_id=workflow.id,
                success=True,
                steps_executed=steps_executed,
                step_results=step_results,
                duration_seconds=duration,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            return WorkflowExecutionResult(
                workflow_id=workflow.id,
                success=False,
                steps_executed=steps_executed,
                step_results=step_results,
                error=str(e),
                duration_seconds=duration,
            )
        
        finally:
            # Cleanup devices
            await self._cleanup_devices()
    
    async def _run_action(self, step: WorkflowStep, ctx: Dict[str, Any]) -> Any:
        """
        Execute an action step.
        
        Args:
            step: Action step to execute
            ctx: Execution context for storing results
            
        Returns:
            Action result
        """
        params = step.params
        device_name = params.get("device")
        action = params.get("action")
        args = params.get("args", {})
        
        if not device_name or not action:
            raise ValueError(f"Action step {step.id} missing device or action")
        
        # Get or create device instance
        device = self._device_instances.get(device_name)
        if device is None:
            device = registry.create(device_name)
            await device.connect()
            self._device_instances[device_name] = device
        
        # Safety check
        await self._safety.check_before_action(device, action, args)
        
        # Execute action
        method = getattr(device, action)
        result = await method(**args)
        
        #  Store result in context
        ctx[step.id] = result
        
        # Convert result to serializable format if it has to_dict()
        if hasattr(result, "to_dict"):
            return result.to_dict()
        
        return result
    
    async def _run_wait(self, step: WorkflowStep) -> None:
        """
        Execute a wait step.
        
        Args:
            step: Wait step with seconds parameter
        """
        seconds = float(step.params.get("seconds", 1.0))
        await asyncio.sleep(seconds)
    
    async def _cleanup_devices(self) -> None:
        """Disconnect all device instances."""
        for device in self._device_instances.values():
            try:
                await device.disconnect()
            except Exception as e:
                print(f"Warning: Failed to disconnect device: {e}")
        
        self._device_instances.clear()
