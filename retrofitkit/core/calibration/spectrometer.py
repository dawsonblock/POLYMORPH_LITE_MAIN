"""
Spectrometer Calibration Utilities.

Provides methods for wavelength calibration and baseline subtraction.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CalibrationError(Exception):
    pass

class SpectrometerCalibrator:
    """
    Handles spectrometer calibration tasks.
    """
    def __init__(self, config):
        self.config = config
        # Load coefficients from config or default
        # Polynomial coefficients for pixel -> nm mapping: nm = c0 + c1*x + c2*x^2 + ...
        self.coeffs = config.raman.calibration_coeffs or [0, 1, 0] 

    def pixel_to_nm(self, pixel_indices: List[int]) -> List[float]:
        """Convert pixel indices to nanometers using stored coefficients."""
        p = np.poly1d(self.coeffs[::-1]) # numpy expects highest power first
        return p(pixel_indices).tolist()

    def calibrate_wavelength(self, 
                             measured_peaks_pixels: List[float], 
                             known_peaks_nm: List[float], 
                             order: int = 2) -> List[float]:
        """
        Recalibrate wavelength mapping based on measured peaks.
        
        Args:
            measured_peaks_pixels: List of pixel positions where peaks were found.
            known_peaks_nm: List of known wavelengths for those peaks.
            order: Polynomial order (usually 1 or 2).
            
        Returns:
            New coefficients [c0, c1, c2, ...]
        """
        if len(measured_peaks_pixels) != len(known_peaks_nm):
            raise CalibrationError("Mismatch in number of measured and known peaks.")
        
        if len(measured_peaks_pixels) < order + 1:
            raise CalibrationError(f"Need at least {order+1} points for order {order} fit.")

        # Fit polynomial: nm = f(pixel)
        # polyfit returns [cn, ..., c1, c0] (highest power first)
        coeffs_high_first = np.polyfit(measured_peaks_pixels, known_peaks_nm, order)
        
        # We store as [c0, c1, c2...] (lowest power first) for consistency with some legacy systems
        # or just stick to numpy standard. Let's stick to numpy standard internally but check config expectation.
        # Let's assume config expects [c0, c1, c2] (standard math notation).
        self.coeffs = coeffs_high_first[::-1].tolist()
        
        logger.info(f"Calibration updated. New coefficients: {self.coeffs}")
        return self.coeffs

    def subtract_baseline(self, 
                          spectrum: List[float], 
                          dark_frame: List[float]) -> List[float]:
        """
        Subtract dark frame from spectrum.
        """
        if len(spectrum) != len(dark_frame):
            raise ValueError("Spectrum and dark frame must have same length.")
            
        return (np.array(spectrum) - np.array(dark_frame)).tolist()

    def find_peaks(self, spectrum: List[float], threshold: float = 100.0) -> List[int]:
        """
        Simple peak finder. Returns pixel indices of peaks.
        """
        # Very basic implementation. In production, use scipy.signal.find_peaks
        peaks = []
        arr = np.array(spectrum)
        for i in range(1, len(arr)-1):
            if arr[i] > threshold and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                peaks.append(i)
        return peaks
