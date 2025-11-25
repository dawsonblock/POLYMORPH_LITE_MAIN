import random, time
from retrofitkit.drivers.daq.base import DAQBase

class SimDAQ(DAQBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = 0.0
        self.t0 = time.time()

    async def set_voltage(self, volts: float):
        self.v = float(volts)

    async def read_ai(self) -> float:
        return self.v + random.gauss(0, self.cfg.daq.simulator.get("noise_std", 0.002))

    async def read_di(self, line: int) -> bool:
        if line == self.cfg.safety.interlocks["estop_line"]:
            return self.cfg.daq.simulator.get("estop", False)
        if line == self.cfg.safety.interlocks["door_line"]:
            return self.cfg.daq.simulator.get("door_open", False)
        return False
