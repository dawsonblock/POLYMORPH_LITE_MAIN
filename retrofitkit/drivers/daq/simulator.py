import random, time
"""
Simulated DAQ driver with DeviceRegistry integration.

Provides realistic simulation of analog I/O for testing and development.
"""
import asyncio
from typing import Dict, Any

from retrofitkit.drivers.daq.base import DAQBase
from retrofitkit.drivers.base import DeviceCapabilities, DAQDevice, DeviceKind
from retrofitkit.core.data_models import DAQTrace
from retrofitkit.core.registry import registry


class SimDAQ(DAQBase, DAQDevice):
    """
    Simulated DAQ with realistic voltage simulation.
    
    Capabilities:
    - Analog input/output
    - Digital input/output
    - Voltage range: -10V to +10V
    - Simulation mode always active
    """
    
    # Class-level capabilities for DeviceRegistry
    capabilities = DeviceCapabilities(
        kind=DeviceKind.DAQ,
        vendor="simulator",
        model="SimDAQ_v1",
        actions=["set_voltage", "read_ai", "write_ao", "read_di", "write_do"],
        features={
            "simulation": True,
            "voltage_range_v": [-10.0, 10.0],
            "ai_channels": 8,
            "ao_channels": 4,
            "di_lines": 8,
            "do_lines": 8,
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
        # Handle both cfg object and kwargs for registry compatibility
        if cfg is not None:
            if isinstance(cfg, dict):
                # Legacy dict configuration
                self._noise = cfg.get("noise_std", 0.01)
                self.estop_active = cfg.get("estop", False)
                self.door_open = cfg.get("door_open", False)
            else:
                # AppContext Config object
                sim_config = cfg.daq.simulator
                self._noise = sim_config.get("noise_v", 0.01)
                self.estop_active = False
                self.door_open = False
            self.id = "sim_daq_0"
        else:
            # Registry-style creation
            self.id = kwargs.get("id", "sim_daq_0")
            self._noise = kwargs.get("noise_v", kwargs.get("noise_std", 0.01))
            self.estop_active = kwargs.get("estop", False)
            self.door_open = kwargs.get("door_open", False)
        
        self.cfg = cfg
        self._voltage = 0.0
        self._di_state = [False] * 8
        self._do_state = [False] * 8
        self._connected = False
        self._t0 = time.time()
        
        # Map legacy attributes
        self.noise_std = self._noise
    
    async def connect(self) -> None:
        """Connect to simulated device (no-op)."""
        await asyncio.sleep(0.1)  # Simulate connection time
        self._connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from simulated device."""
        self._connected = False
        self._voltage = 0.0
    
    async def health(self) -> Dict[str, Any]:
        """Get simulator health status."""
        return {
            "status": "ok",
            "mode": "simulation",
            "voltage_v": self._voltage,
            "uptime_s": time.time() - self._t0,
            "connected": self._connected,
        }
    
    async def set_voltage(self, volts: float):
        """Set simulated output voltage."""
        # Clamp to +/- 10V
        self._voltage = max(-10.0, min(10.0, float(volts)))
        await asyncio.sleep(0.01)
    
    async def read_ai(self, channel: int = 0) -> float:
        """Read simulated analog input with noise."""
        await asyncio.sleep(0.01)
        return self._voltage + random.gauss(0, self._noise)
    
    async def write_ao(self, channel: int, value: float) -> None:
        """Write analog output (alias for set_voltage)."""
        await self.set_voltage(value)
    
    async def read_di(self, line: int) -> bool:
        """Read digital input line."""
        if 0 <= line < len(self._di_state):
            return self._di_state[line]
        return False
    
    async def write_do(self, line: int, state: bool) -> None:
        """Write digital output line."""
        if 0 <= line < len(self._do_state):
            self._do_state[line] = bool(state)

    async def read_interlocks(self) -> Dict[str, bool]:
        """Read safety interlocks."""
        return {
            "estop": self.estop_active,
            "door": self.door_open
        }

    # Compatibility aliases for tests
    @property
    def output_voltage(self) -> float:
        return self._voltage

    @output_voltage.setter
    def output_voltage(self, value: float):
        self._voltage = float(value)

    async def read_voltage(self) -> float:
        return await self.read_ai(0)


# Backward compatibility alias for tests
SimulatorDAQ = SimDAQ


# Register with DeviceRegistry
registry.register("sim_daq", SimDAQ)
registry.register("daq_simulator", SimDAQ)  # Alternative name
