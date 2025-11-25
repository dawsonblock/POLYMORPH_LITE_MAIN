import asyncio, time
from typing import Dict, Any
from retrofitkit.drivers.raman.base import RamanBase
try:
    import seabreeze.spectrometers as sb
except Exception:
    sb = None

class OceanRaman(RamanBase):
    def __init__(self, cfg):
        self.t0 = time.time()
        self.spec = None
        if sb:
            devs = sb.list_devices()
            if devs:
                self.spec = sb.Spectrometer(devs[0])
                self.spec.integration_time_micros(20000)

    async def read_frame(self) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        if not self.spec:
            return {"t": time.time()-self.t0, "wavelengths":[532.0], "intensities":[1000.0], "peak_nm":532.0, "peak_intensity":1000.0}
        wl = self.spec.wavelengths()
        it = self.spec.intensities(correct_dark_counts=True, correct_nonlinearity=True)
        idx = int(it.argmax()) if hasattr(it, "argmax") else max(range(len(it)), key=lambda i: it[i])
        return {"t": time.time()-self.t0, "wavelengths": wl.tolist(), "intensities": it.tolist(), "peak_nm": float(wl[idx]), "peak_intensity": float(it[idx])}
