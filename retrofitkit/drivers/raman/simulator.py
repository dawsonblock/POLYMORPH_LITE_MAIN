import asyncio, math, random, time
from typing import Dict, Any
from retrofitkit.drivers.raman.base import RamanBase

class SimRaman(RamanBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.t0 = time.time()
        self.peak_nm = float(cfg.raman.simulator.get("peak_nm", 532.0))
        self.intensity = float(cfg.raman.simulator.get("base_intensity", 1000.0))
        self.noise = float(cfg.raman.simulator.get("noise_std", 2.0))
        self.drift = float(cfg.raman.simulator.get("drift_per_s", 0.5))

    async def read_frame(self) -> Dict[str, Any]:
        t = time.time() - self.t0
        # Simulate slow drift and a “reaction” accelerating intensity
        self.intensity += self.drift + random.gauss(0, self.noise)
        intensities = [max(0.0, self.intensity + random.gauss(0, self.noise))]
        wavelengths = [self.peak_nm]
        await asyncio.sleep(0.2)
        return {"t": t, "wavelengths": wavelengths, "intensities": intensities, "peak_nm": self.peak_nm, "peak_intensity": intensities[0]}
