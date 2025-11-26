"""
Ocean Optics spectrometer driver using seabreeze library.

This is a reference implementation showing how to:
1. Implement DeviceBase and SpectrometerDevice protocols
2. Return unified Spectrum data model
3. Register with DeviceRegistry for discovery
"""
import asyncio
import time
from typing import Dict, Any

from retrofitkit.drivers.base import DeviceBase, SpectrometerDevice, DeviceCapabilities, DeviceKind
from retrofitkit.core.data_models import Spectrum
from retrofitkit.core.registry import registry

try:
    import seabreeze.spectrometers as sb
except Exception:
    sb = None


class OceanOpticsSpectrometer(SpectrometerDevice):
    """
    Ocean Optics USB spectrometer driver.
    
    Supports all seabreeze-compatible devices (USB2000, HR4000, QE65000, etc.).
    Falls back to simulation mode if no hardware present.
    """
    
    # Class-level capabilities for registry
    capabilities = DeviceCapabilities(
        kind=DeviceKind.SPECTROMETER,
        vendor="ocean_optics",
        model="auto",  # Auto-detected from device
        actions=["acquire_spectrum"],
        features={
            "dark_correction": True,
            "nonlinearity_correction": True,
            "integration_time_range_us": [1000, 60000000],
        }
    )
    
    def __init__(self, device_index: int = 0, integration_time_ms: float = 20.0):
        """
        Initialize Ocean Optics spectrometer.
        
        Args:
            device_index: Index of device if multiple connected (0-based)
            integration_time_ms: Default integration time in milliseconds
        """
        self.id = f"ocean_optics_{device_index}"
        self._device_index = device_index
        self._integration_time_ms = integration_time_ms
        self._device = None
        self._t0 = time.time()
        self._connected = False
    
    async def connect(self) -> None:
        """
        Connect to Ocean Optics device.
        
        Raises:
            RuntimeError: If device not found
        """
        if sb is None:
            # Simulation mode
            self._connected = True
            return
        
        devices = sb.list_devices()
        if not devices:
            # Simulation mode
            self._connected = True
            return
        
        if self._device_index >= len(devices):
            raise RuntimeError(
                f"Device index {self._device_index} out of range. "
                f"Found {len(devices)} device(s)."
            )
        
        self._device = sb.Spectrometer(devices[self._device_index])
        self._device.integration_time_micros(int(self._integration_time_ms * 1000))
        self._connected = True
        
        # Update capabilities with actual model
        if hasattr(self._device, 'model'):
            self.capabilities.model = self._device.model
    
    async def disconnect(self) -> None:
        """Close connection to device."""
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
        self._device = None
        self._connected = False
    
    async def health(self) -> Dict[str, Any]:
        """
        Get device health status.
        
        Returns:
            Dict with status, integration time, and model info
        """
        if not self._connected:
            return {"status": "disconnected"}
        
        if self._device is None:
            return {
                "status": "ok",
                "mode": "simulation",
                "integration_time_ms": self._integration_time_ms,
            }
        
        return {
            "status": "ok",
            "mode": "hardware",
            "model": getattr(self._device, 'model', 'unknown'),
            "integration_time_ms": self._integration_time_ms,
            "serial_number": getattr(self._device, 'serial_number', 'unknown'),
        }
    
    async def acquire_spectrum(
        self,
        integration_time_ms: float | None = None,
        **kwargs
    ) -> Spectrum:
        """
        Acquire a spectrum.
        
        Args:
            integration_time_ms: Optional integration time override (milliseconds)
            **kwargs: Additional device-specific parameters
            
        Returns:
            Spectrum object with wavelengths and intensities
        """
        # Update integration time if specified
        if integration_time_ms is not None:
            if self._device is not None:
                self._device.integration_time_micros(int(integration_time_ms * 1000))
            self._integration_time_ms = integration_time_ms
        
        # Simulate acquisition delay
        await asyncio.sleep(self._integration_time_ms / 1000.0)
        
        # Simulation mode if no hardware
        if self._device is None:
            # Simple simulated Raman peak
            import numpy as np
            wavelengths = np.linspace(400, 900, 1024)
            # Gaussian peak at 532 nm
            intensities = 100 * np.exp(-((wavelengths - 532) ** 2) / (2 * 10 ** 2)) + 50
            
            return Spectrum(
                wavelengths=wavelengths,
                intensities=intensities,
                meta={
                    "integration_time_ms": self._integration_time_ms,
                    "mode": "simulation",
                    "device_id": self.id,
                }
            )
        
        # Real hardware acquisition
        wavelengths = self._device.wavelengths()
        intensities = self._device.intensities(
            correct_dark_counts=True,
            correct_nonlinearity=True
        )
        
        # Find peak
        peak_idx = intensities.argmax()
        peak_wavelength = float(wavelengths[peak_idx])
        peak_intensity = float(intensities[peak_idx])
        
        return Spectrum(
            wavelengths=wavelengths,
            intensities=intensities,
            meta={
                "integration_time_ms": self._integration_time_ms,
                "mode": "hardware",
                "device_id": self.id,
                "model": getattr(self._device, 'model', 'unknown'),
                "peak_wavelength_nm": peak_wavelength,
                "peak_intensity": peak_intensity,
            }
        )


# Register driver with global registry
registry.register("ocean_optics", OceanOpticsSpectrometer)
