"""
National Instruments DAQ driver with DeviceRegistry integration.

Supports NI-DAQmx hardware with simulation fallback when SDK unavailable.
"""
from typing import Dict, Any
from retrofitkit.drivers.daq.base import DAQBase
from retrofitkit.drivers.production_base import ProductionHardwareDriver
from retrofitkit.drivers.base import DeviceCapabilities, DAQDevice, DeviceKind
from retrofitkit.core.registry import registry

try:
    import nidaqmx
    from nidaqmx.constants import LineGrouping
except Exception:
    nidaqmx = None


class NIDAQ(ProductionHardwareDriver, DAQBase, DAQDevice):
    """
    National Instruments DAQ driver.
    
    Requires nidaqmx SDK and Runtime. Falls back to simulation when unavailable.
    
    Setup:
    1. Install NI-DAQmx Runtime from ni.com/drivers
    2. Install python bindings: `pip install nidaqmx`
    """
    
    # Class-level capabilities for DeviceRegistry
    capabilities = DeviceCapabilities(
        kind=DeviceKind.DAQ,
        vendor="national_instruments",
        model="NI-DAQmx",
        actions=["set_voltage", "read_ai", "write_ao", "read_di", "write_do"],
        features={
            "simulation": nidaqmx is None,
            "sdk_available": nidaqmx is not None,
            "voltage_range_v": [-10.0, 10.0],
            "supports_watchdog": True,
        }
    )
    
    def __init__(self, cfg=None, **kwargs):
        """
        Initialize NI DAQ.
        
        Args:
            cfg: Configuration object (optional for registry compatibility)
            **kwargs: Allow registry creation with named params
        """
        # Handle both cfg object and kwargs for registry compatibility
        # Handle both cfg object and kwargs for registry compatibility
        if cfg is not None:
            super().__init__(max_workers=1)
            self.cfg = cfg
            if isinstance(cfg, dict):
                # Legacy dict configuration
                self.dev = cfg.get("device_name", "Dev1")
                self.ao = cfg.get("ao_voltage_channel", "ao0")
                self.ai = cfg.get("ai_voltage_channel", "ai0")
                self.di_lines = cfg.get("di_lines", ["port0/line0","port0/line1"])
                self.do_watchdog = cfg.get("do_watchdog_line", "port0/line2")
            else:
                # AppContext Config object
                self.dev = cfg.daq.ni["device_name"]
                self.ao = cfg.daq.ni["ao_voltage_channel"]
                self.ai = cfg.daq.ni["ai_voltage_channel"]
                self.di_lines = cfg.daq.ni.get("di_lines", ["port0/line0","port0/line1"])
                self.do_watchdog = cfg.daq.ni.get("do_watchdog_line", "port0/line2")
            self.id = f"ni_daq_{self.dev}"
        else:
            # Registry-style creation
            super().__init__(max_workers=1)
            self.id = kwargs.get("id", "ni_daq_0")
            self.dev = kwargs.get("device_name", "Dev1")
            self.ao = kwargs.get("ao_channel", "ao0")
            self.ai = kwargs.get("ai_channel", "ai0")
            self.di_lines = kwargs.get("di_lines", ["port0/line0","port0/line1"])
            self.do_watchdog = kwargs.get("do_watchdog_line", "port0/line2")
            self.cfg = None
        
        self._last_v = 0.0
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to NI DAQ (marks as connected)."""
        self._connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from NI DAQ."""
        self._connected = False
        self._last_v = 0.0
    
    async def health(self) -> Dict[str, Any]:
        """Get NI DAQ health status."""
        return {
            "status": "ok" if self._connected else "disconnected",
            "mode": "simulation" if nidaqmx is None else "hardware",
            "device": self.dev,
            "last_voltage_v": self._last_v,
            "sdk_available": nidaqmx is not None,
        }

    def _set_voltage_blocking(self, volts: float):
        self._last_v = float(volts)
        if nidaqmx is None:
            return
        with nidaqmx.Task() as t:
            t.ao_channels.add_ao_voltage_chan(f"{self.dev}/{self.ao}")
            t.write(self._last_v)

    async def set_voltage(self, volts: float):
        await self._run_blocking(self._set_voltage_blocking, volts, timeout=1.0)

    def _read_ai_blocking(self) -> float:
        if nidaqmx is None:
            return self._last_v
        with nidaqmx.Task() as t:
            t.ai_channels.add_ai_voltage_chan(f"{self.dev}/{self.ai}")
            return float(t.read())

    async def read_ai(self) -> float:
        return await self._run_blocking(self._read_ai_blocking, timeout=1.0)

    def _read_di_blocking(self, line: int) -> bool:
        if nidaqmx is None:
            # fallback: read from config simulator flags if present
            sim = self.cfg.daq.simulator
            if line == self.cfg.safety.interlocks["estop_line"]:
                return bool(sim.get("estop", False))
            if line == self.cfg.safety.interlocks["door_line"]:
                return bool(sim.get("door_open", False))
            return False
        name = self.di_lines[line] if line < len(self.di_lines) else self.di_lines[0]
        with nidaqmx.Task() as t:
            t.di_channels.add_di_chan(f"{self.dev}/{name}", line_grouping=LineGrouping.CHAN_PER_LINE)
            val = t.read()
        return bool(val)

    async def read_di(self, line: int) -> bool:
        return await self._run_blocking(self._read_di_blocking, line, timeout=0.5)

    def _write_do_blocking(self, line: int, on: bool):
        if nidaqmx is None:
            return
        name = self.di_lines[line] if line < len(self.di_lines) else self.di_lines[0]
        with nidaqmx.Task() as t:
            t.do_channels.add_do_chan(f"{self.dev}/{name}", line_grouping=LineGrouping.CHAN_PER_LINE)
            t.write(bool(on))

    async def write_do(self, line: int, on: bool):
        await self._run_blocking(self._write_do_blocking, line, on, timeout=1.0)

    # Optional: call this from an async heartbeat to toggle a DO line
    async def toggle_watchdog(self, v: bool):
        def _toggle():
            if nidaqmx is None:
                return
            with nidaqmx.Task() as t:
                t.do_channels.add_do_chan(f"{self.dev}/{self.do_watchdog}", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
                t.write(bool(v))
        await self._run_blocking(_toggle, timeout=1.0)

    # Compatibility aliases for tests
    @property
    def device_name(self) -> str:
        return self.dev
        
    @property
    def ao_channel(self) -> str:
        return self.ao
        
    @property
    def ai_channel(self) -> str:
        return self.ai

    async def read_voltage(self) -> float:
        return await self.read_ai()


# Backward compatibility alias for tests
NI_DAQ = NIDAQ


# Register with DeviceRegistry
registry.register("ni_daq", NIDAQ)
registry.register("national_instruments", NIDAQ)  # Alternative name
