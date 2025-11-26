import asyncio
from retrofitkit.core.app import AppContext
from retrofitkit.core.events import EventBus
from retrofitkit.drivers.raman.factory import make_raman
from retrofitkit.metrics.exporter import Metrics

class RamanStreamer:
    def __init__(self, ctx: AppContext, bus: EventBus, device=None):
        self.ctx = ctx
        self.bus = bus
        self._task = None
        self._stop = asyncio.Event()
        self.raman = device if device else make_raman(ctx.config)
        self.mx = Metrics.get()

    async def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._stop.set()
            await self._task

    async def frames(self):
        while True:
            frame = await self.raman.read_frame()
            self.mx.set("polymorph_raman_peak_intensity", frame.get("peak_intensity", 0.0))
            yield frame

    async def _run(self):
        while not self._stop.is_set():
            frame = await self.raman.read_frame()
            self.mx.set("polymorph_raman_peak_intensity", frame.get("peak_intensity", 0.0))
            await asyncio.sleep(0.2)
