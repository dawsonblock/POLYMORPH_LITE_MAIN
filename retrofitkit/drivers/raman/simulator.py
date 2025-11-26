"""
Raman simulator driver with DeviceRegistry integration.

Provides realistic simulation of Raman spectroscopy with drift and noise.
"""
import asyncio
import random
import time
import numpy as np
from typing import Dict, Any

from retrofitkit.drivers.raman.base import RamanBase
from retrofitkit.drivers.base import DeviceCapabilities, SpectrometerDevice, DeviceKind
from retrofitkit.core.data_models import Spectrum
from retrofitkit.core.registry import registry


class SimRaman(RamanBase, SpectrometerDevice):
    """
    Simulated Raman spectrometer with realistic drift and noise.
    
    Capabilities:
    - Configurable peak wavelength
    - Intensity drift over time
    - Gaussian noise
    - Returns typed Spectrum objects
    """
    
    # Class-level capabilities for DeviceRegistry
    capabilities = DeviceCapabilities(
        kind=DeviceKind.SPECTROMETER,
        vendor="simulator",
        model="SimRaman_v1",
        actions=["acquire_spectrum", "read_frame"],
        features={
            "simulation": True,
            "drift_simulation": True,
            "noise_simulation": True,
        }
    )
    
    def __init__(self, cfg=None, **kwargs):
        """
        Initialize simulator.
        
        Args:
            cfg: Configuration object (optional for registry compatibility)
            **kwargs: Allow registry creation with named params
        """
        # Handle both cfg object and kwargs for registry compatibility
        if cfg is not None:
            sim_config = cfg.raman.simulator
            self.id = "sim_raman_0"
            self.peak_nm = float(sim_config.get("peak_nm", 532.0))
            self.intensity = float(sim_config.get("base_intensity", 1000.0))
            self.noise = float(sim_config.get("noise_std", 2.0))
            self.drift = float(sim_config.get("drift_per_s", 0.5))
        else:
            # Registry-style creation
            self.id = kwargs.get("id", "sim_raman_0")
            self.peak_nm = kwargs.get("peak_nm", 532.0)
            self.intensity = kwargs.get("base_intensity", 1000.0)
            self.noise = kwargs.get("noise_std", 2.0)
            self.drift = kwargs.get("drift_per_s", 0.5)
        
        self.cfg = cfg
        self.t0 = time.time()
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to simulated device (no-op)."""
        self._connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from simulated device (no-op)."""
        self._connected = False
    
    async def health(self) -> Dict[str, Any]:
        """Get simulator health status."""
        return {
            "status": "ok",
            "mode": "simulation",
            "peak_nm": self.peak_nm,
            "current_intensity": self.intensity,
            "uptime_s": time.time() - self.t0,
        }
    
    async def acquire_spectrum(self, **kwargs) -> Spectrum:
        """
        Acquire a simulated spectrum (DeviceRegistry-compatible).
        
        Returns:
            Spectrum object with simulated data
        """
        await asyncio.sleep(0.2)  # Simulate acquisition time
        
        # Simulate drift
        self.intensity += self.drift + random.gauss(0, self.noise)
        
        # Create spectrum around peak
        wavelengths = np.array([self.peak_nm])
        intensities = np.array([max(0.0, self.intensity + random.gauss(0, self.noise))])
        
        return Spectrum(
            wavelengths=wavelengths,
            intensities=intensities,
            meta={
                "t": time.time() - self.t0,
                "peak_nm": self.peak_nm,
                "peak_intensity": float(intensities[0]),
                "simulation": True,
                "device_id": self.id,
            }
        )
    
    async def read_frame(self) -> Dict[str, Any]:
        """
        Legacy dict-based interface for backward compatibility.
        
        Returns:
            Dict with wavelengths, intensities, peak info
        """
        spectrum = await self.acquire_spectrum()
        
        return {
            "t": spectrum.meta["t"],
            "wavelengths": spectrum.wavelengths.tolist(),
            "intensities": spectrum.intensities.tolist(),
            "peak_nm": spectrum.meta["peak_nm"],
            "peak_intensity": spectrum.meta["peak_intensity"],
        }


# Backward compatibility alias for tests
SimulatorRaman = SimRaman


# Register with DeviceRegistry
registry.register("simulator", SimRaman)
registry.register("sim_raman", SimRaman)  # Alternative name
