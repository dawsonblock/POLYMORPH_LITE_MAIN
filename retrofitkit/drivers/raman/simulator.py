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
            self.id = "sim_raman_0"
            self.peak_nm = float(cfg.raman.simulator_peak_nm)
            self.intensity = float(cfg.raman.simulator_base_intensity)
            self.noise = float(cfg.raman.simulator_noise_std)
            self.drift = float(cfg.raman.simulator_drift_per_s)
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

        # Golden Run Data Playback
        self._playback_data = None
        self._playback_wavelengths = None
        self._playback_index = 0
        self._load_playback_data()

    def _load_playback_data(self):
        """Load synthetic data for golden run simulation."""
        import os
        import pandas as pd

        csv_path = "data/crystallization_demo.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                self._playback_wavelengths = np.array([float(c) for c in df.columns])
                self._playback_data = df.values
                print(f"✅ SimRaman: Loaded {len(df)} frames of golden run data")
            except Exception as e:
                print(f"⚠️ SimRaman: Failed to load demo data: {e}")

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
            "playback_active": self._playback_data is not None
        }

    async def acquire_spectrum(self, **kwargs) -> Spectrum:
        """
        Acquire a simulated spectrum (DeviceRegistry-compatible).
        
        Returns:
            Spectrum object with simulated data
        """
        await asyncio.sleep(0.2)  # Simulate acquisition time

        # Playback Mode (Golden Run)
        if self._playback_data is not None:
            # Get current frame
            intensities = self._playback_data[self._playback_index]
            wavelengths = self._playback_wavelengths

            # Advance frame (loop)
            self._playback_index = (self._playback_index + 1) % len(self._playback_data)

            # Add some live noise on top of playback
            noise = np.random.normal(0, self.noise, len(intensities))
            intensities = np.maximum(0, intensities + noise)

            # Find peak for metadata
            peak_idx = intensities.argmax()

            return Spectrum(
                wavelengths=wavelengths,
                intensities=intensities,
                meta={
                    "t": time.time() - self.t0,
                    "peak_nm": float(wavelengths[peak_idx]),
                    "peak_intensity": float(intensities[peak_idx]),
                    "simulation": True,
                    "mode": "playback",
                    "frame": self._playback_index,
                    "device_id": self.id,
                }
            )

        # Legacy Simulation Mode (Random Noise)
        # Simulate drift
        self.intensity += self.drift + random.gauss(0, self.noise)

        # Create spectrum around peak
        wavelengths = np.linspace(400, 1000, 1024)
        # Gaussian peak
        intensities = self.intensity * np.exp(-((wavelengths - self.peak_nm) ** 2) / (2 * 10 ** 2))
        # Add noise
        intensities += np.random.normal(0, self.noise, len(wavelengths))
        intensities = np.maximum(0, intensities)

        return Spectrum(
            wavelengths=wavelengths,
            intensities=intensities,
            meta={
                "t": time.time() - self.t0,
                "peak_nm": self.peak_nm,
                "peak_intensity": float(intensities.max()),
                "simulation": True,
                "mode": "synthetic",
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
