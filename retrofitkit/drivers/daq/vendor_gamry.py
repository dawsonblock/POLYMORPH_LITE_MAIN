"""
Gamry potentiostat driver with simulation detection.

Requires gamry_sdk (fictitious). Falls back to simulation when unavailable.
"""
import time
import warnings
from typing import Dict, Any
from retrofitkit.drivers.daq.base import DAQBase
from retrofitkit.drivers.base import DeviceCapabilities, DAQDevice, DeviceKind
from retrofitkit.core.registry import registry

try:
    import gamry_sdk
except ImportError:
    gamry_sdk = None


class GamryPotentiostat(DAQBase, DAQDevice):
    """
    Gamry electrochemistry potentiostat driver.
    
    Requires gamry_sdk. Falls back to simulation if unavailable.
    """

    # Class-level capabilities
    capabilities = DeviceCapabilities(
        kind=DeviceKind.DAQ,
        vendor="gamry",
        model="Reference600+",
        actions=["set_voltage", "read_ai", "read_eis"],
        features={
            "simulation": gamry_sdk is None,
            "sdk_available": gamry_sdk is not None,
            "electrochemistry": True,
        }
    )

    def __init__(self, cfg=None, **kwargs):
        """Initialize Gamry driver with simulation detection."""
        self.cfg = cfg
        self.simulation_mode = (gamry_sdk is None)

        if self.simulation_mode:
            warnings.warn(
                "Gamry SDK not available - running in SIMULATION mode. "
                "Install gamry_sdk for real hardware control.",
                RuntimeWarning,
                stacklevel=2
            )

        if cfg is not None:
            self.dev = cfg.daq.gamry.get("device_id", "Gamry0")
            self.id = f"gamry_{self.dev}"
        else:
            self.dev = kwargs.get("device_id", "Gamry0")
            self.id = kwargs.get("id", "gamry_0")

        self._voltage = 0.0
        self._connected = False
        self.device = None
        self._is_connected = False

    def _connect_blocking(self):
        if gamry_sdk:
            self.device = gamry_sdk.Potentiostat(self.cfg.get("port", "USB0"))
            self.device.open()
            self._is_connected = True
        else:
            # Simulation mode
            self._is_connected = True

    def _acquire_blocking(self, duration: float) -> Dict[str, Any]:
        if not self._is_connected:
            raise RuntimeError("Device not connected")

        if gamry_sdk and self.device:
            return self.device.run_curve(duration)
        else:
            # Simulation
            time.sleep(duration)
            return {
                "voltage": [0.1, 0.2, 0.3],
                "current": [1e-6, 2e-6, 3e-6],
                "timestamp": time.time()
            }

    async def connect(self) -> None:
        """Connect to Gamry device."""
        if not self.simulation_mode and gamry_sdk:
            # Real SDK initialization would go here
            pass
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Gamry device."""
        self._connected = False
        self._voltage = 0.0

    async def health(self) -> Dict[str, Any]:
        """Get device health with clear simulation indicator."""
        return {
            "status": "ok" if self._connected else "disconnected",
            "mode": "simulation" if self.simulation_mode else "hardware",
            "device": self.dev,
            "voltage_v": self._voltage,
            "sdk_available": gamry_sdk is not None,
            "simulation": self.simulation_mode,
        }

    async def acquire(self, duration: float = 1.0) -> Dict[str, Any]:
        return await self._run_blocking(self._acquire_blocking, duration, timeout=duration + 5.0)

    async def set_voltage(self, volts: float):
        """Set voltage (simulated if SDK unavailable)."""
        self._voltage = float(volts)
        if not self.simulation_mode and gamry_sdk:
            # Real SDK call would go here
            pass

    async def read_ai(self) -> float:
        """Read analog input (simulated if SDK unavailable)."""
        return self._voltage

    async def read_eis(self, freq_range=(0.1, 100000)) -> Dict[str, Any]:
        """
        Read electrochemical impedance spectroscopy.
        
        Returns simulated data if SDK unavailable.
        """
        if self.simulation_mode:
            import numpy as np

            # Simulated EIS data
            freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 50)
            z_real = 100 + 50 * np.random.randn(50)
            z_imag = -80 + 30 * np.random.randn(50)

            return {
                "frequencies_hz": freqs.tolist(),
                "z_real_ohm": z_real.tolist(),
                "z_imag_ohm": z_imag.tolist(),
                "simulation": self.simulation_mode,
                "device": self.dev,
            }
        else:
            # Real SDK call for EIS
            return self.device.run_eis(freq_range)


# Register with DeviceRegistry
registry.register("gamry", GamryPotentiostat)
registry.register("gamry_potentiostat", GamryPotentiostat)
