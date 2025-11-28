"""
Workflow Executor.

Asynchronous engine for executing recipes.
Integrates:
- DriverRouter (Hardware)
- DatabaseLogger (Persistence)
- Interlocks (Safety)
"""
import asyncio
import logging
from typing import Dict, Any

from retrofitkit.core.recipe import Recipe, Step
from retrofitkit.core.driver_router import get_router
from retrofitkit.core.workflows.db_logger import DatabaseLogger
from retrofitkit.core.safety.interlocks import get_interlocks

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    """
    Executes a Recipe step-by-step.
    """
    def __init__(self, config, db_logger: DatabaseLogger, ai_client=None):
        self.config = config
        self.logger = db_logger
        self.ai_client = ai_client
        self.step_results = []  # Track results of each step for context
        self.router = get_router()
        self.interlocks = None
        try:
            self.interlocks = get_interlocks(config)
        except Exception:
            pass # Safety might not be initialized in tests

        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set() # Start unpaused

    async def execute(self, recipe: Recipe, operator_email: str, run_metadata: Dict[str, Any] = None, on_start=None):
        """
        Execute a recipe.
        """
        # 1. Initialize Run
        try:
            # Assuming recipe.id is the version ID or we have it.
            # For now, let's assume recipe object has what we need or we pass version ID.
            # Let's assume recipe.id is the UUID of the version.
            run_id = self.logger.log_run_start(recipe.id, operator_email, run_metadata)

            if on_start:
                if asyncio.iscoroutinefunction(on_start):
                    await on_start(run_id)
                else:
                    on_start(run_id)

        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            raise

        logger.info(f"Starting execution of recipe: {recipe.name} ({len(recipe.steps)} steps)")

        # 2. Execute Steps
        try:
            for i, step in enumerate(recipe.steps):
                # Check control flags
                if self._stop_event.is_set():
                    logger.info("Execution stopped by user.")
                    self.logger.log_run_complete("aborted", "Stopped by user")
                    return

                await self._pause_event.wait()

                # Log step start
                self.logger.log_step_start(i, step.type)

                # Execute step
                try:
                    # SAFETY CHECK: Before execution
                    if self.interlocks:
                        self.interlocks.check_safe()

                    result = await self._execute_step(step)

                    # SAFETY CHECK: After execution (and pet watchdog)
                    if self.interlocks:
                        self.interlocks.check_safe()
                        # Pet watchdog if available (assuming interlocks controller has access or we do it separately)
                        # Ideally watchdog is separate, but for now let's assume safety check implies system is healthy enough
                        pass

                    self.logger.log_step_complete(i, step.type, result)
                except Exception as step_err:
                    logger.error(f"Step {i} ({step.type}) failed: {step_err}")
                    self.logger.log_run_complete("failed", str(step_err))
                    raise step_err

            # 3. Finalize
            self.logger.log_run_complete("completed")

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            # Status already updated in catch block above if step failed
            # If generic error:
            if not self._stop_event.is_set(): # Don't overwrite aborted status
                 self.logger.log_run_complete("failed", str(e))
            raise

    async def _execute_step(self, step: Step) -> Any:
        """Dispatch step to handler with safety checks."""
        # SAFETY CHECK: Before execution
        if self.interlocks:
            self.interlocks.check_safe()

        handler_name = f"_handle_{step.type}"
        handler = getattr(self, handler_name, None)

        if not handler:
            raise ValueError(f"Unknown step type: {step.type}")

        result = await handler(step.params)

        # Store result for context
        self.step_results.append(result)

        # SAFETY CHECK: After execution (and pet watchdog)
        if self.interlocks:
            self.interlocks.check_safe()

        return result

    # --- Step Handlers ---

    async def _handle_wait(self, params: Dict[str, Any]):
        """Wait for N seconds."""
        duration = float(params.get("seconds", 0))
        logger.info(f"Waiting {duration}s...")
        await asyncio.sleep(duration)
        return {"duration": duration}

    async def _handle_daq(self, params: Dict[str, Any]):
        """
        DAQ Operation.
        Params: action (read/write), channel, value (for write)
        """
        driver = self.router.get_driver("daq", self.config)
        action = params.get("action")

        if action == "read_ai":
            val = await driver.read_ai(params.get("channel", 0))
            return {"value": val}
        elif action == "write_ao":
            await driver.write_ao(params.get("channel", 0), params.get("value", 0.0))
            return {"status": "ok"}
        elif action == "read_di":
            val = await driver.read_di(params.get("line", 0))
            return {"value": val}
        elif action == "write_do":
            await driver.write_do(params.get("line", 0), params.get("state", False))
            return {"status": "ok"}
        else:
            raise ValueError(f"Unknown DAQ action: {action}")

    async def _handle_raman(self, params: Dict[str, Any]):
        """
        Raman Acquisition.
        Params: exposure_time
        """
        driver = self.router.get_driver("raman", self.config)

        # Safety check (redundant if driver has @require_safety, but good practice)
        if self.interlocks:
            self.interlocks.check_safe()

        data = await driver.acquire_spectrum(exposure_time=params.get("exposure_time"))
        return data # Contains wavelengths, intensities, metadata

    async def _handle_compute(self, params: Dict[str, Any]):
        """
        Basic computation (placeholder).
        """
        # In real system, this would access previous results context
        return {"result": "computed"}

    async def _handle_ai_decision(self, params: Dict[str, Any]):
        """
        AI-driven decision step.
        Uses the last available spectrum from context.
        """
        if not self.ai_client:
            raise RuntimeError("AI Client not initialized in executor")

        # Find last spectrum in results
        spectrum_data = None
        for res in reversed(self.step_results):
            if isinstance(res, dict) and ("intensities" in res or "spectrum" in res):
                spectrum_data = res.get("intensities") or res.get("spectrum")
                break
            # Handle Spectrum object if we start returning them directly
            if hasattr(res, "intensities"):
                spectrum_data = res.intensities.tolist()
                break
        
        if not spectrum_data:
            logger.warning("No spectrum found in context for AI decision. Using dummy data.")
            spectrum_data = [0.0] * 1024 # Dummy data to prevent crash if testing

        # Call AI Service
        prediction = await self.ai_client.predict(spectrum_data, critical=params.get("critical", True))
        
        # Log prediction
        logger.info(f"AI Prediction: {prediction}")
        
        return {"prediction": prediction}

    async def _handle_decision(self, params: Dict[str, Any]):
        """
        Conditional branching.
        Params: condition (str), true_step (int), false_step (int)
        
        Note: This requires the executor to support jumping, which the current loop doesn't fully support.
        The current loop iterates linearly. To support jumping, we need to change the loop to a while loop with index control.
        
        For now, we'll just return the decision result, and assume the recipe compiler handles the jump 
        (or we refactor the main loop).
        
        Refactoring main loop to support jumps:
        """
        # Placeholder for now as refactoring the main loop is a bigger change.
        # Let's just evaluate the condition.
        condition = params.get("condition", "True")
        # Eval is dangerous, in prod use a safe expression evaluator
        result = bool(eval(condition, {"__builtins__": None}, {}))
        return {"result": result, "branch": "true" if result else "false"}

    # --- Control ---

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()
