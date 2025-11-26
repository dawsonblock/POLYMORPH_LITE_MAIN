from retrofitkit.drivers.daq.base import DAQBase
from retrofitkit.drivers.production_base import ProductionHardwareDriver
try:
    import nidaqmx
    from nidaqmx.constants import LineGrouping
except Exception:
    nidaqmx = None

class NIDAQ(ProductionHardwareDriver, DAQBase):
    def __init__(self, cfg):
        super().__init__(max_workers=1)
        self.cfg = cfg
        self.dev = cfg.daq.ni["device_name"]
        self.ao = cfg.daq.ni["ao_voltage_channel"]
        self.ai = cfg.daq.ni["ai_voltage_channel"]
        self.di_lines = cfg.daq.ni.get("di_lines", ["port0/line0","port0/line1"])
        self.do_watchdog = cfg.daq.ni.get("do_watchdog_line", "port0/line2")
        self._last_v = 0.0

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

    # Optional: call this from an async heartbeat to toggle a DO line
    async def toggle_watchdog(self, v: bool):
        def _toggle():
            if nidaqmx is None: return
            with nidaqmx.Task() as t:
                t.do_channels.add_do_chan(f"{self.dev}/{self.do_watchdog}", line_grouping=LineGrouping.CHAN_PER_LINE)
                t.write(bool(v))
        await self._run_blocking(_toggle, timeout=0.5)

# Backwards-compatible alias (tests and legacy code may import NI_DAQ)
NI_DAQ = NIDAQ

# Backwards-compatible alias (tests and legacy code may import NI_DAQ)
NI_DAQ = NIDAQ
