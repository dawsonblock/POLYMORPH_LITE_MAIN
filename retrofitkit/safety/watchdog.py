import asyncio
from retrofitkit.drivers.daq.factory import make_daq

class Watchdog:
    """
    Hardware watchdog that toggles a digital output line periodically.
    
    Note: Orchestrator also has its own _watchdog_loop(). Consider consolidating.
    """
    def __init__(self, ctx):
        self.ctx = ctx
        self.daq = make_daq(ctx.config)
        self._stop = asyncio.Event()

    async def run(self):
        v = False
        while not self._stop.is_set():
            try:
                if hasattr(self.daq, "toggle_watchdog"):
                    await self.daq.toggle_watchdog(v)  # FIX: Must await async method
                v = not v
            except Exception as e:
                print(f"Watchdog toggle failed: {e}")
            await asyncio.sleep(1.0)

    def stop(self):
        self._stop.set()
