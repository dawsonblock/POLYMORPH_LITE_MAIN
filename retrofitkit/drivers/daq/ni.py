"""
National Instruments DAQ driver with DeviceRegistry integration.

Supports NI-DAQmx hardware with simulation fallback when SDK unavailable.
"""
import os
from typing import Dict, Any
from retrofitkit.drivers.daq.base import DAQBase
from retrofitkit.drivers.production_base import ProductionHardwareDriver
from retrofitkit.drivers.base import DeviceCapabilities, DAQDevice, DeviceKind
from retrofitkit.core.registry import registry
from retrofitkit.core.hardware_utils import hardware_call

try:
    import nidaqmx
    from nidaqmx.constants import LineGrouping
except Exception:
    nidaqmx = None


class NIDAQ(ProductionHardwareDriver, DAQBase, DAQDevice):
    """
    National Instruments DAQ driver.
    
    Requires nidaqmx SDK. 
    Controlled by USE_REAL_HARDWARE env var:
    - If "1": Forces real hardware connection (raises error if missing).
    - If "0" or unset: Forces simulation mode.
    """

    # Class-level capabilities for DeviceRegistry
    capabilities = DeviceCapabilities(
        kind=DeviceKind.DAQ,
        vendor="national_instruments",
        model="NI-DAQmx",
        actions=["set_voltage", "read_ai", "write_ao", "read_di", "write_do"],
        features={
            "simulation": True, # Dynamic based on runtime
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
        if cfg is not None:
            super().__init__(max_workers=1)
            self.cfg = cfg
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
        
        # Check environment flag
        self._use_real_hardware = os.environ.get("USE_REAL_HARDWARE") == "1"
        
        if self._use_real_hardware and nidaqmx is None:
            raise RuntimeError("USE_REAL_HARDWARE=1 but nidaqmx SDK is not installed.")

    async def connect(self) -> None:
        """Connect to NI DAQ (marks as connected)."""
        if self._use_real_hardware:
            # Verify we can actually create a task (simple check)
            try:
                with nidaqmx.Task() as t:
                    pass # Just check if we can instantiate
            except Exception as e:
                raise RuntimeError(f"Failed to connect to real NI hardware: {e}")
        
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from NI DAQ."""
        self._connected = False
        self._last_v = 0.0

    async def health(self) -> Dict[str, Any]:
        """Get NI DAQ health status."""
        return {
            "status": "ok" if self._connected else "disconnected",
            "mode": "hardware" if self._use_real_hardware else "simulation",
            "device": self.dev,
            "last_voltage_v": self._last_v,
            "sdk_available": nidaqmx is not None,
        }

    def _set_voltage_blocking(self, volts: float):
        self._last_v = float(volts)
        if not self._use_real_hardware:
            return
        with nidaqmx.Task() as t:
            t.ao_channels.add_ao_voltage_chan(f"{self.dev}/{self.ao}")
            t.write(self._last_v)

    @hardware_call(timeout=1.0)
    async def set_voltage(self, volts: float):
        await self._run_blocking(self._set_voltage_blocking, volts, timeout=1.0)

    def _read_ai_blocking(self) -> float:
        if not self._use_real_hardware:
            return self._last_v
        with nidaqmx.Task() as t:
            t.ai_channels.add_ai_voltage_chan(f"{self.dev}/{self.ai}")
            return float(t.read())

    @hardware_call(timeout=1.0)
    async def read_ai(self) -> float:
        return await self._run_blocking(self._read_ai_blocking, timeout=1.0)

    def _read_di_blocking(self, line: int) -> bool:
        if not self._use_real_hardware:
            # fallback: read from config simulator flags if present
            if self.cfg:
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

    @hardware_call(timeout=0.5)
    async def read_di(self, line: int) -> bool:
        return await self._run_blocking(self._read_di_blocking, line, timeout=0.5)

    def _write_do_blocking(self, line: int, on: bool):
        if not self._use_real_hardware:
            return
        name = self.di_lines[line] if line < len(self.di_lines) else self.di_lines[0]
        with nidaqmx.Task() as t:
            t.do_channels.add_do_chan(f"{self.dev}/{name}", line_grouping=LineGrouping.CHAN_PER_LINE)
            t.write(bool(on))

    @hardware_call(timeout=1.0)
    async def write_do(self, line: int, on: bool):
        await self._run_blocking(self._write_do_blocking, line, on, timeout=1.0)

    # Optional: call this from an async heartbeat to toggle a DO line
    @hardware_call(timeout=1.0)
    async def toggle_watchdog(self, v: bool):
        def _toggle():
            if not self._use_real_hardware:
                return
            with nidaqmx.Task() as t:
                t.do_channels.add_do_chan(f"{self.dev}/{self.do_watchdog}", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
                t.write(bool(v))
        await self._run_blocking(_toggle, timeout=1.0)


# Backward compatibility alias for tests
NI_DAQ = NIDAQ


# Register with DeviceRegistry
registry.register("ni_daq", NIDAQ)
registry.register("national_instruments", NIDAQ)  # Alternative name
