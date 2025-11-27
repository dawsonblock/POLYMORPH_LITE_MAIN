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
        if cfg is not None:
            sim_config = cfg.daq.simulator
            self.id = "sim_daq_0"
            self._noise = sim_config.get("noise_v", 0.01)
        else:
            # Registry-style creation
            self.id = kwargs.get("id", "sim_daq_0")
            self._noise = kwargs.get("noise_v", 0.01)
        
        self.cfg = cfg
        self._voltage = 0.0
        self._di_state = [False] * 8
        self._do_state = [False] * 8
        self._connected = False
        self._t0 = time.time()
    
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
        # Clamp to range
        volts = max(-10.0, min(10.0, float(volts)))
        self._voltage = volts
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

    # Helpers for tests/compatibility
    async def read_voltage(self) -> float:
        """Alias for read_ai(0)."""
        return await self.read_ai(0)

    async def read_interlocks(self) -> Dict[str, bool]:
        """Read safety interlocks (simulated on DI lines)."""
        # Assume DI 0 is Estop (Active High for test), DI 1 is Door
        estop = await self.read_di(0)
        door = await self.read_di(1)
        
        # If estop was set via property in test, respect it
        if hasattr(self, 'estop_active'):
            estop = self.estop_active
        if hasattr(self, 'door_open'):
            door = self.door_open
            
        return {"estop": estop, "door": door}


# Backward compatibility alias for tests
SimulatorDAQ = SimDAQ


# Register with DeviceRegistry
registry.register("sim_daq", SimDAQ)
registry.register("daq_simulator", SimDAQ)  # Alternative name
