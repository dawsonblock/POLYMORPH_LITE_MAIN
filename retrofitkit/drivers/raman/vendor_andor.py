# retrofitkit/drivers/raman/vendor_andor.py

import math
import time
import numpy as np
from typing import Optional

from retrofitkit.core.data_models import Spectrum
from retrofitkit.drivers.base import DeviceKind, DeviceCapabilities
from retrofitkit.drivers.production_base import ProductionHardwareDriver
from retrofitkit.core.registry import registry


class AndorRaman(ProductionHardwareDriver):
    """
    Andor Raman spectrometer driver.

    This implementation is:
      - SDK-backed if the Andor library is available
      - otherwise falls back to a safe simulator that still returns a valid Spectrum

    This ensures:
      - the module always imports
      - workflows can run in sim mode
      - real hardware can be hooked in later without touching callers
    """

    KIND = DeviceKind.SPECTROMETER
    MODEL = "andor_raman"
    VENDOR = "andor"

    capabilities = DeviceCapabilities(
        kind=DeviceKind.SPECTROMETER,
        vendor="Andor",
        model="Newton",
        actions=["acquire_spectrum"]
    )

    def __init__(self, config):
        # Initialize base with default workers, don't pass config as max_workers
        super().__init__(max_workers=1)
        self.config = config
        self._sdk = None
        self._t0 = time.time()
        self._init_sdk_if_available()

    async def get_temperature(self) -> float:
        """Get the current detector temperature in Celsius."""
        if self._sdk:
            # Placeholder for real SDK call
            # return await self._run_blocking(self._sdk.GetTemperature)
            return -45.0
        else:
            # Simulated temperature
            return -45.0

    # ---------- SDK / simulation setup ----------

    def _init_sdk_if_available(self) -> None:
        """
        Try to import and initialize the real Andor SDK.
        If not available, we silently fall back to simulation.
        """
        try:
            # Replace with real Andor SDK import when available
            import andor_sdk  # type: ignore

            self._sdk = andor_sdk
            self.logger.info("Andor SDK detected, using real hardware backend.")
        except Exception:
            self._sdk = None
            import os
            if os.environ.get("USE_REAL_HARDWARE") == "1":
                raise RuntimeError("USE_REAL_HARDWARE=1 but Andor SDK not found.")
            
            self.logger.warning(
                "Andor SDK not available, AndorRaman running in SIMULATOR mode."
            )

    # ---------- Core spectrometer interface ----------

    async def acquire_spectrum(
        self,
        integration_time_ms: float,
        averages: int = 1,
        center_wavelength_nm: Optional[float] = None,
    ) -> Spectrum:
        """
        Acquire a spectrum from the Andor spectrometer.

        When SDK is present:
          - call real hardware
        Otherwise:
          - return a clean synthetic spectrum with plausible metadata
        """
        if self._sdk is not None:
            return await self._acquire_real(
                integration_time_ms=integration_time_ms,
                averages=averages,
                center_wavelength_nm=center_wavelength_nm,
            )
        else:
            return await self._acquire_simulated(
                integration_time_ms=integration_time_ms,
                averages=averages,
                center_wavelength_nm=center_wavelength_nm,
            )

    async def _acquire_real(
        self,
        integration_time_ms: float,
        averages: int,
        center_wavelength_nm: Optional[float],
    ) -> Spectrum:
        """
        Placeholder for real Andor acquisition logic.

        Implement using the actual Andor API when you have the SDK wired.
        For now, we raise a clear error so no one assumes this is done.
        """
        raise NotImplementedError(
            "Real Andor SDK integration not yet implemented. "
            "Implement _acquire_real() in vendor_andor.py when SDK is available."
        )

    async def _acquire_simulated(
        self,
        integration_time_ms: float,
        averages: int,
        center_wavelength_nm: Optional[float],
    ) -> Spectrum:
        """
        Clean simulator: single Gaussian peak plus noise.

        This guarantees:
          - valid Spectrum schema
          - deterministic behavior across tests
        """
        t = time.time() - self._t0
        n_points = 1024

        # simple synthetic wavelength axis around a 500–600 nm band
        center = center_wavelength_nm or 550.0
        span = 50.0
        start = center - span
        stop = center + span
        step = (stop - start) / (n_points - 1)
        wavelengths = [start + i * step for i in range(n_points)]

        # single Gaussian peak at center
        sigma = span / 6.0
        intensities: list[float] = []
        for wl in wavelengths:
            dx = wl - center
            base = math.exp(-0.5 * (dx / sigma) ** 2)
            noise = 0.02 * math.sin(0.1 * wl)  # soft structure
            intensities.append(max(0.0, base + noise))

        metadata = {
            "integration_time_ms": integration_time_ms,
            "averages": averages,
            "center_wavelength_nm": center,
            "backend": "simulator",
            "device_model": self.MODEL,
            "vendor": self.VENDOR,
            "t": t,
        }

        return Spectrum(
            wavelengths=np.array(wavelengths),
            intensities=np.array(intensities),
            meta=metadata,
        )

    async def close(self) -> None:
        """
        Ensure clean shutdown; close hardware handles if SDK is active.
        """
        # Add real SDK cleanup here when implemented
        self.logger.info("AndorRaman closed.")


# Register driver with the global registry at import time
registry.register("andor_raman", AndorRaman)
