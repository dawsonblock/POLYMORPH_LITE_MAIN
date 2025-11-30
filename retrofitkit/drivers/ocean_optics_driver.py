"""
Ocean Optics Spectrometer Driver for POLYMORPH v8.0.

This driver provides a unified interface for Ocean Optics spectrometers using the
SeaBreeze library. It supports:
- Device discovery and initialization
- Spectrum acquisition with integration time control
- Dark spectrum correction
- Wavelength calibration retrieval
- Simulation mode for development without hardware

Dependencies:
    - seabreeze
    - numpy
"""

import logging
import time
from typing import Optional, List, Dict, Tuple, Union
import numpy as np

# Try to import seabreeze, else use mock for simulation
try:
    import seabreeze.spectrometers as sb
    SEABREEZE_AVAILABLE = True
except ImportError:
    SEABREEZE_AVAILABLE = False

logger = logging.getLogger(__name__)

class OceanOpticsDriver:
    """
    Driver for Ocean Optics spectrometers (e.g., USB2000+, HR4000).
    """

    def __init__(self, device_id: Optional[str] = None, simulate: bool = False):
        """
        Initialize the spectrometer driver.

        Args:
            device_id: Serial number of the device to open. If None, opens the first available.
            simulate: If True, forces simulation mode even if hardware is present.
        """
        self.device_id = device_id
        self.simulate = simulate or not SEABREEZE_AVAILABLE
        self.spec = None
        self._wavelengths = None
        self._dark_spectrum = None

        if not SEABREEZE_AVAILABLE and not self.simulate:
            logger.warning("SeaBreeze not installed. Falling back to simulation mode.")
            self.simulate = True

        self.connect()

    def connect(self):
        """Connect to the spectrometer."""
        if self.simulate:
            logger.info(f"Connected to SIMULATED Ocean Optics Spectrometer (ID: {self.device_id or 'SIM-001'})")
            # Simulate wavelengths (e.g., 200-1100 nm)
            self._wavelengths = np.linspace(200, 1100, 2048)
            return

        try:
            devices = sb.list_devices()
            if not devices:
                raise RuntimeError("No Ocean Optics devices found.")
            
            if self.device_id:
                # Find specific device
                # Note: sb.list_devices() returns objects, need to check serials if possible
                # For simplicity in this driver, we might just try to open it.
                # SeaBreeze API allows opening by serial.
                try:
                    self.spec = sb.Spectrometer.from_serial_number(self.device_id)
                except Exception:
                     raise RuntimeError(f"Device with ID {self.device_id} not found.")
            else:
                self.spec = sb.Spectrometer(devices[0])

            self.device_id = self.spec.serial_number
            self._wavelengths = self.spec.wavelengths()
            logger.info(f"Connected to Ocean Optics Spectrometer: {self.device_id}")

        except Exception as e:
            logger.error(f"Failed to connect to spectrometer: {e}")
            raise

    def disconnect(self):
        """Close the connection to the spectrometer."""
        if self.spec:
            self.spec.close()
            self.spec = None
        logger.info("Disconnected from spectrometer.")

    def set_integration_time(self, micros: int):
        """
        Set the integration time in microseconds.
        
        Args:
            micros: Integration time in microseconds.
        """
        if self.simulate:
            logger.debug(f"SIM: Set integration time to {micros} us")
            return

        if self.spec:
            # SeaBreeze expects microseconds
            limits = self.spec.integration_time_micros_limits
            if not (limits[0] <= micros <= limits[1]):
                logger.warning(f"Integration time {micros} out of bounds {limits}. Clamping.")
                micros = max(limits[0], min(micros, limits[1]))
            self.spec.integration_time_micros(micros)

    def acquire_spectrum(self, correct_dark: bool = True, average: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire a spectrum.

        Args:
            correct_dark: If True, subtracts the stored dark spectrum (if available).
            average: Number of scans to average.

        Returns:
            Tuple of (wavelengths, intensities)
        """
        if average < 1:
            raise ValueError("Average count must be >= 1")

        intensities_accum = None

        for i in range(average):
            if self.simulate:
                # Generate synthetic Raman-like spectrum
                # Base fluorescence
                x = np.linspace(0, 1, len(self._wavelengths))
                base = 100 * np.exp(-x) + 50
                # Add some peaks
                peaks = 500 * np.exp(-0.5 * ((self._wavelengths - 500) / 10)**2)  # Peak at 500nm
                peaks += 300 * np.exp(-0.5 * ((self._wavelengths - 800) / 15)**2) # Peak at 800nm
                noise = np.random.normal(0, 5, len(self._wavelengths))
                intensities = base + peaks + noise
                # Simulate integration time effect (linear scaling)
                intensities *= 1.0 
            else:
                intensities = self.spec.intensities()

            if intensities_accum is None:
                intensities_accum = intensities.astype(float)
            else:
                intensities_accum += intensities

        avg_intensities = intensities_accum / average

        # Dark correction
        if correct_dark and self._dark_spectrum is not None:
            if len(self._dark_spectrum) == len(avg_intensities):
                avg_intensities -= self._dark_spectrum
                # Clip negatives? Usually yes for physical meaning, but raw data might be useful.
                # Let's clip to 0 for now.
                avg_intensities = np.maximum(avg_intensities, 0)
            else:
                logger.warning("Dark spectrum length mismatch. Skipping correction.")

        return self._wavelengths, avg_intensities

    def acquire_dark_spectrum(self, average: int = 10):
        """
        Acquire and store a dark spectrum for later correction.
        User should ensure the shutter is closed or light source is off.
        """
        logger.info("Acquiring dark spectrum...")
        _, dark = self.acquire_spectrum(correct_dark=False, average=average)
        self._dark_spectrum = dark
        logger.info("Dark spectrum acquired and stored.")

    def get_wavelengths(self) -> np.ndarray:
        """Return the wavelength calibration array."""
        return self._wavelengths

    def is_simulated(self) -> bool:
        return self.simulate

# Mock driver for testing if needed explicitly, though the main class handles simulation.
class MockOceanOpticsDriver(OceanOpticsDriver):
    def __init__(self):
        super().__init__(simulate=True)
