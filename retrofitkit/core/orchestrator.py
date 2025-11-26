import asyncio, time, json
import redis.asyncio as redis
import httpx
from typing import Dict, Any, Union
from retrofitkit.core.app import AppContext
from retrofitkit.core.recipe import Recipe
from retrofitkit.core.gating import GatingEngine
from retrofitkit.compliance.audit import Audit
from retrofitkit.data.storage import DataStore
from retrofitkit.metrics.exporter import Metrics
from retrofitkit.core.data_models import Spectrum
from retrofitkit.core.registry import registry

# Import drivers to trigger auto-registration
from retrofitkit.drivers.raman import simulator as raman_sim, vendor_ocean_optics
from retrofitkit.drivers.daq import simulator as daq_sim, ni

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
        self.mx.set("polymorph_run_state", RUN_STATE["IDLE"])
        
        # Redis connection for state persistence
        self.redis = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Watchdog
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        self._last_heartbeat = time.time()
        
        # AI Service URL
        self.ai_service_url = ctx.config.ai.service_url
        
        # AI Service Circuit Breaker
        self._ai_failures = 0
        self._ai_last_failure_time = 0
        self._ai_circuit_open = False
        self._ai_failure_threshold = 3
        self._ai_recovery_timeout = 60.0
    
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
        
        try:
            # Try DeviceRegistry first (Option C path)
            return registry.create(backend, cfg=config)
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
        recipe_id = f"{recipe.name}_{operator_email}" # Simple ID for checkpointing
        checkpoint_key = f"checkpoint:{recipe_id}"
        
        start_idx = 0
        rid = None
        
        if resume:
            try:
                checkpoint = await self.redis.get(checkpoint_key)
                if checkpoint:
                    data = json.loads(checkpoint)
                    start_idx = data["step"] + 1
                    rid = data["rid"]
                    self.audit.record(event="RESUME_FOUND", actor=operator_email, subject=recipe.name, details={"step": start_idx})
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

        if rid is None:
            rid = self.store.start_run(recipe.name, operator_email, simulation)
            self.audit.record(event="RUN_START", actor=operator_email, subject=recipe.name, details={"run_id": rid, "simulation": simulation})

        self.mx.set("polymorph_run_active", 1, {"run_id": rid})
        self.mx.set("polymorph_run_state", RUN_STATE["ACTIVE"])
        gating = GatingEngine(self.ctx.config.gating.rules)

        try:
            steps = recipe.steps
            for idx, step in enumerate(steps):
                if idx < start_idx:
                    continue

                # Checkpoint before step
                await self.redis.setex(checkpoint_key, 86400, json.dumps({"step": idx, "rid": rid, "ts": time.time()}))

                try:
                    async with asyncio.timeout(float(step.params.get("timeout", 300))):
                        if step.type == "bias_set":
                            v = float(step.params.get("volts", 0.0))
                            await self.daq.set_voltage(v)
                            self.mx.set("polymorph_ao_volts", v)
                            self.mx.set("polymorph_ai_volts", await self.daq.read_ai())
                            self.audit.record(event="BIAS_SET", actor=operator_email, subject=recipe.name, details={"V": v})
                        elif step.type == "bias_ramp":
                            v0 = float(step.params.get("from", 0.0))
                            v1 = float(step.params.get("to", 1.0))
                            dt = float(step.params.get("seconds", 5.0))
                            n = max(2, int(dt * 20))
                            for i in range(n):
                                v = v0 + (v1 - v0) * (i / (n - 1))
                                await self.daq.set_voltage(v)
                                self.mx.set("polymorph_ao_volts", v)
                                self.mx.set("polymorph_ai_volts", await self.daq.read_ai())
                                await asyncio.sleep(dt / n)
                            self.audit.record(event="BIAS_RAMP", actor=operator_email, subject=recipe.name, details={"from": v0, "to": v1, "seconds": dt})
                        elif step.type == "hold":
                            dur = float(step.params.get("seconds", 1.0))
                            t0 = time.time()
                            while time.time()-t0 < dur:
                                self.mx.set("polymorph_ai_volts", await self.daq.read_ai())
                                await asyncio.sleep(0.2)
                            self.audit.record(event="HOLD", actor=operator_email, subject=recipe.name, details={"seconds": dur})
                        elif step.type == "wait_for_raman":
                            timeout = float(step.params.get("timeout_s", 120.0))
                            tstart = time.time()
                            while True:
                                spec = await self.raman.read_frame()
                                self.store.append_spectrum(rid, spec)
                                self.mx.set("polymorph_raman_peak_intensity", spec.get("peak_intensity", 0.0))
                                
                                # AI Inference Call
                                if "intensity" in spec:
                                    ai_result = await self._call_inference_service(spec["intensity"])
                                    if ai_result:
                                        self.mx.set("polymorph_ai_active_modes", ai_result.get("active_modes", 0))
                                        if ai_result.get("new_polymorph"):
                                            self.audit.record(event="AI_DISCOVERY", actor="BentoML", subject=recipe.name, details={"polymorph": ai_result["new_polymorph"]})
                                            print(f"AI DISCOVERY: {ai_result['new_polymorph']}")

                                hit = gating.update(spec)
                                if hit:
                                    self.audit.record(event="GATE_TRIGGERED", actor=operator_email, subject=recipe.name, details={"t": spec["t"], "peak": spec["peak_nm"], "intensity": spec["peak_intensity"]})
                                    break
                                if time.time() - tstart > timeout:
                                    self.audit.record(event="GATE_TIMEOUT", actor=operator_email, subject=recipe.name, details={"timeout_s": timeout})
                                    break
                        elif step.type == "gate_stop":
                            await self.daq.set_voltage(0.0)
                            self.mx.set("polymorph_ao_volts", 0.0)
                            self.mx.set("polymorph_ai_volts", await self.daq.read_ai())
                            self.audit.record(event="STOP", actor=operator_email, subject=recipe.name, details={})
                        else:
                            self.audit.record(event="UNKNOWN_STEP", actor=operator_email, subject=recipe.name, details={"step": step.type})
                except asyncio.TimeoutError:
                    await self._emergency_shutdown()
                    raise

            self.audit.record(event="RUN_END", actor=operator_email, subject=recipe.name, details={"run_id": rid})
            # Clear checkpoint on success
            await self.redis.delete(checkpoint_key)
            
        except Exception as e:
            self.audit.record(event="RUN_ERROR", actor=operator_email, subject=recipe.name, details={"error": str(e)})
            self.mx.set("polymorph_run_state", RUN_STATE["ERROR"])
            raise
        finally:
            # Ensure safe state
            try:
                await self.daq.set_voltage(0.0)
            except:
                pass
            self.mx.set("polymorph_run_active", 0, {"run_id": rid})
            self.mx.set("polymorph_run_state", RUN_STATE["IDLE"])
        return rid
