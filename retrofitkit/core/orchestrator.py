import asyncio
import time
import os
import httpx
from typing import Dict, Any, Union
from retrofitkit.core.app import AppContext
from retrofitkit.core.recipe import Recipe
from retrofitkit.compliance.audit import Audit
from retrofitkit.data.storage import DataStore
from retrofitkit.metrics.exporter import Metrics
from retrofitkit.core.data_models import Spectrum
from retrofitkit.core.registry import registry
import retrofitkit.drivers  # noqa: F401

# Import drivers to trigger auto-registration

RUN_STATE = {"IDLE":0, "ACTIVE":1, "ERROR":2}

class AIFailsafeError(Exception):
    """Raised when AI service is critical but unreachable."""
    pass

class Orchestrator:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.audit = Audit()
        self.store = DataStore(ctx.config.system.data_dir)

        # Create devices via DeviceRegistry (Option C path)
        self.daq = self._create_daq_device(ctx.config)
        self.raman = self._create_raman_device(ctx.config)
        self.mx = Metrics.get()

        # State tracking
        self._active_run_id = None
        self._run_state = RUN_STATE["IDLE"]

        # Async components - initialized on start()
        self._watchdog_task = None
        self._redis = None

        # AI Service URL (with fallback)
        try:
            self.ai_service_url = ctx.config.ai.service_url
        except AttributeError:
            # Fallback if ai config missing
            self.ai_service_url = os.getenv("P4_AI_URL", "http://localhost:3000")

        # AI Circuit Breaker state
        self._ai_failures = 0
        self._ai_circuit_open = False
        self._ai_circuit_threshold = 3
        self._ai_failure_threshold = 3
        self._ai_recovery_timeout = 60.0

    @property
    def status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "ai_circuit_open": self._ai_circuit_open,
            "ai_failures": self._ai_failures,
            "run_state": self._run_state,
            "active_run_id": self._active_run_id,
            "progress": getattr(self, "_progress", {"current": 0, "total": 0}),
        }

    def _spectrum_to_dict(self, data: Union[Spectrum, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert Spectrum object to dict, or pass through if already dict.
        
        Handles transition period where drivers may return either Spectrum objects
        or legacy dicts.
        
        Args:
            data: Either a Spectrum object or dict
            
        Returns:
            Dict representation suitable for legacy code
        """
        if isinstance(data, Spectrum):
            return data.to_dict()
        return data

    def _extract_spectrum_for_ai(self, data: Union[Spectrum, Dict[str, Any]]) -> list:
        """
        Extract spectrum intensities for AI service.
        
        Args:
            data: Either Spectrum object or dict
            
        Returns:
            List of intensity values for AI processing
        """
        if isinstance(data, Spectrum):
            return data.intensities.tolist()

        # Legacy dict format
        return data.get("intensities", [])

    def _create_daq_device(self, config):
        """
        Create DAQ device via DeviceRegistry with factory fallback.
        
        Args:
            config: Application configuration
            
        Returns:
            DAQ device instance
        """
        backend = config.daq.backend

        # Map configuration backend names to registry keys
        registry_name = {
            "simulator": "daq_simulator",
            "redpitaya": "redpitaya_daq",
        }.get(backend, backend)

        try:
            # Try DeviceRegistry first (Option C path)
            return registry.create(registry_name, cfg=config)
        except KeyError:
            # Fallback to legacy factory during transition
            from retrofitkit.drivers.daq.factory import make_daq
            return make_daq(config)

    def _create_raman_device(self, config):
        """
        Create Raman device via DeviceRegistry with factory fallback.
        
        Args:
            config: Application configuration
            
        Returns:
            Raman device instance
        """
        provider = config.raman.provider

        try:
            # Try DeviceRegistry first (Option C path)
            return registry.create(provider, cfg=config)
        except KeyError:
            # Fallback to legacy factory during transition
            from retrofitkit.drivers.raman.factory import make_raman
            return make_raman(config)

    async def _call_inference_service(self, spectrum: list, critical: bool = True) -> dict:
        """
        Call AI service with Circuit Breaker. 
        If critical=True and service fails/timeout, raises AIFailsafeError.
        """
        if self._ai_circuit_open:
            if time.time() - self._ai_last_failure_time > self._ai_recovery_timeout:
                print("AI Circuit Breaker: Attempting recovery...")
            elif critical:
                raise AIFailsafeError("AI Circuit Breaker OPEN - Failsafe Triggered")
            else:
                return {}

        try:
            async with httpx.AsyncClient() as client:
                payload = {"spectrum": spectrum}
                response = await client.post(self.ai_service_url, json=payload, timeout=2.0)

                if response.status_code == 200:
                    if self._ai_circuit_open:
                        print("AI Circuit Breaker: Recovered.")
                        self._ai_failures = 0
                        self._ai_circuit_open = False
                    return response.json()
                else:
                    self._record_ai_failure()
                    msg = f"AI Service Error: {response.status_code}"
                    print(msg)
                    if critical: raise AIFailsafeError(msg)
                    return {}
        except httpx.TimeoutException as e:
            self._record_ai_failure()
            msg = f"AI Connection Timeout: {str(e)}"
            print(msg)
            if critical: raise AIFailsafeError(msg)
            return {}
        except Exception as e:
            self._record_ai_failure()
            msg = f"AI Connection Failed: {str(e)}"
            print(msg)
            if critical: raise AIFailsafeError(msg)
            return {}

    def _record_ai_failure(self):
        self._ai_failures += 1
        self._ai_last_failure_time = time.time()
        if self._ai_failures >= self._ai_failure_threshold:
            if not self._ai_circuit_open:
                print("AI Circuit Breaker: OPEN (Too many failures)")
            self._ai_circuit_open = True

    async def _watchdog_loop(self):
        """Global watchdog to monitor system health."""
        while True:
            try:
                # Update heartbeat
                self._last_heartbeat = time.time()

                # Check for system freeze (if this loop doesn't run, external watchdog should trigger)
                # Here we can check if hardware is responsive if needed

                # Toggle hardware watchdog if supported
                if hasattr(self.daq, "toggle_watchdog"):
                    await self.daq.toggle_watchdog(True)
                    await asyncio.sleep(0.1)
                    await self.daq.toggle_watchdog(False)

                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Watchdog error: {e}")
                await asyncio.sleep(1.0)

    async def _emergency_shutdown(self):
        """Stop all hardware immediately."""
        try:
            if hasattr(self.daq, "set_voltage"):
                await self.daq.set_voltage(0.0)
            # Add other shutdown logic here
        except Exception as e:
            print(f"Emergency shutdown failed: {e}")

    async def run(self, recipe: Recipe, operator_email: str, simulation: bool=False, resume: bool=True) -> str:
        return await self.execute_recipe(recipe, operator_email, simulation, resume)

    async def execute_recipe(self, recipe: Recipe, operator_email: str, simulation: bool=False, resume: bool=True) -> str:
        """
        Execute recipe using the new WorkflowExecutor.
        """
        from retrofitkit.core.workflows.executor import WorkflowExecutor
        from retrofitkit.core.workflows.db_logger import DatabaseLogger
        from retrofitkit.api.compliance import get_session

        # Initialize new engine components
        # Initialize new engine components
        db_logger = DatabaseLogger(get_session)
        executor = WorkflowExecutor(self.ctx.config, db_logger)

        # Update metrics and state
        self._run_state = RUN_STATE["ACTIVE"]
        self.mx.set("polymorph_run_state", RUN_STATE["ACTIVE"])

        def on_start(run_id):
            self._active_run_id = run_id

        try:
            # Execute
            await executor.execute(recipe, operator_email, {"simulation": simulation}, on_start=on_start)

            return self._active_run_id

        except Exception as e:
            self._run_state = RUN_STATE["ERROR"]
            self.mx.set("polymorph_run_state", RUN_STATE["ERROR"])
            raise e
        finally:
            self._run_state = RUN_STATE["IDLE"]
            self._active_run_id = None
            self.mx.set("polymorph_run_state", RUN_STATE["IDLE"])
