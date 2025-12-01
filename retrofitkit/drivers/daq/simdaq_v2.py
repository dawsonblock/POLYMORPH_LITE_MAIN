import numpy as np
import time
from typing import Dict, Any, Optional
from retrofitkit.drivers.base import DeviceKind, DeviceCapabilities
from retrofitkit.drivers.production_base import ProductionHardwareDriver
from retrofitkit.core.registry import registry

class SimDAQv2(ProductionHardwareDriver):
    """
    SimDAQ v2: Deterministic Synthetic Data Generator for CI/CD.
    
    Features:
    - Deterministic noise generation (seeded)
    - Configurable signal shapes (sine, square, pulse)
    - Simulated hardware latency
    - Error injection mode for testing failure paths
    """
    
    KIND = DeviceKind.DAQ
    MODEL = "simdaq_v2"
    VENDOR = "polymorph_sim"
    
    capabilities = DeviceCapabilities(
        kind=DeviceKind.DAQ,
        vendor="PolymorphSim",
        model="SimDAQv2",
        actions=["acquire_waveform", "generate_waveform"]
    )
    
    def __init__(self, config: Any):
        super().__init__(max_workers=1)
        self.config = config
        self.seed = getattr(config, "seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self._error_mode = False
        
    async def health(self) -> Dict[str, Any]:
        return {
            "status": "error" if self._error_mode else "ok",
            "mode": "simulation",
            "version": "2.0.0"
        }
        
    async def acquire_waveform(self, 
                             channels: list[int], 
                             sample_rate: float, 
                             duration: float) -> Dict[str, np.ndarray]:
        """
        Acquire synthetic waveforms.
        """
        if self._error_mode:
            raise RuntimeError("Simulated hardware failure")
            
        # Simulate acquisition latency
        await self._simulate_latency(duration * 0.1)
        
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        results = {}
        for ch in channels:
            # Generate deterministic signal based on channel index
            # Ch0: Sine, Ch1: Square, others: Noise
            if ch == 0:
                signal = np.sin(2 * np.pi * 10 * t) # 10 Hz sine
            elif ch == 1:
                signal = np.sign(np.sin(2 * np.pi * 5 * t)) # 5 Hz square
            else:
                signal = np.zeros_like(t)
                
            # Add deterministic noise
            noise = self.rng.normal(0, 0.1, num_samples)
            results[f"ch{ch}"] = signal + noise
            
        return results

    async def generate_waveform(self, channel: int, waveform: np.ndarray, sample_rate: float):
        """
        Simulate waveform generation output.
        """
        if self._error_mode:
            raise RuntimeError("Simulated hardware failure")
            
        # Just simulate the time it takes to "upload" and "play"
        duration = len(waveform) / sample_rate
        await self._simulate_latency(duration)
        return {"status": "generated", "samples": len(waveform)}

    async def _simulate_latency(self, seconds: float):
        # In fast CI mode, we might skip this, but for realism we keep it small
        import asyncio
        await asyncio.sleep(min(seconds, 0.1))

    def set_error_mode(self, enabled: bool):
        """Enable/disable error injection for testing."""
        self._error_mode = enabled

registry.register("simdaq_v2", SimDAQv2)
