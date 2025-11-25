import asyncio, time
from retrofitkit.drivers.daq.factory import make_daq
class Watchdog:
    def __init__(self, ctx):
        self.ctx = ctx
        self.daq = make_daq(ctx.config)
        self._stop = asyncio.Event()
    async def run(self):
        v = False
        while not self._stop.is_set():
            try:
                if hasattr(self.daq, "toggle_watchdog"):
                    self.daq.toggle_watchdog(v)
                v = not v
            except Exception:
                pass
            await asyncio.sleep(1.0)
    def stop(self):
        self._stop.set()
