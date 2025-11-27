from retrofitkit.core.app import AppContext
from retrofitkit.drivers.daq.factory import make_daq

class Interlocks:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.daq = make_daq(ctx.config)

    def estop_triggered(self) -> bool:
        line = self.ctx.config.safety.estop_line
        return bool(self.daq.read_di(line))

    def door_open(self) -> bool:
        line = self.ctx.config.safety.door_line
        return bool(self.daq.read_di(line))
