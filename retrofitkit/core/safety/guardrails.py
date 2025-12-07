"""
Safety Guardrails for Raman Spectroscopy.

Provides protection against:
- Over-intensity (sensor damage prevention)
- Dead sensor detection (flatline detection)
- Invalid spectrum detection (NaN, zeros, corruption)
"""

from typing import List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SafetyGuardrails:
    """
    Safety guardrails for spectroscopy operations.
    
    All checks return (is_safe: bool, message: str)
    """
    
    def __init__(
        self,
        max_intensity: float = 65535.0,  # 16-bit max
        intensity_spike_factor: float = 10.0,
        flatline_threshold: float = 1e-6,
        flatline_samples: int = 10,
        nan_tolerance: float = 0.01,  # 1% NaN allowed
        zero_tolerance: float = 0.50,  # 50% zeros allowed
    ):
        self.max_intensity = max_intensity
        self.intensity_spike_factor = intensity_spike_factor
        self.flatline_threshold = flatline_threshold
        self.flatline_samples = flatline_samples
        self.nan_tolerance = nan_tolerance
        self.zero_tolerance = zero_tolerance
        
        # State tracking
        self._last_intensity: Optional[float] = None
        self._flatline_count: int = 0
        self._trip_count: int = 0
    
    def check_over_intensity(self, signal: np.ndarray) -> Tuple[bool, str]:
        """
        Check for over-intensity conditions.
        
        Args:
            signal: Intensity array
            
        Returns:
            (is_safe, message)
        """
        max_val = np.nanmax(signal)
        
        # Check absolute maximum
        if max_val > self.max_intensity:
            self._trip_count += 1
            logger.warning(f"Over-intensity detected: {max_val:.0f} > {self.max_intensity}")
            return False, f"Over-intensity: {max_val:.0f} exceeds max {self.max_intensity}"
        
        # Check spike relative to last reading
        if self._last_intensity is not None and self._last_intensity > 0:
            ratio = max_val / self._last_intensity
            if ratio > self.intensity_spike_factor:
                self._trip_count += 1
                logger.warning(f"Intensity spike detected: {ratio:.1f}x increase")
                return False, f"Intensity spike: {ratio:.1f}x sudden increase"
        
        self._last_intensity = max_val
        return True, "OK"
    
    def check_dead_sensor(self, samples: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Check for dead sensor (flatline detection).
        
        Args:
            samples: List of recent intensity arrays
            
        Returns:
            (is_safe, message)
        """
        if len(samples) < self.flatline_samples:
            return True, "Insufficient samples for flatline detection"
        
        # Check variance across recent samples
        recent = samples[-self.flatline_samples:]
        stacked = np.vstack(recent)
        variance = np.var(stacked)
        
        if variance < self.flatline_threshold:
            self._flatline_count += 1
            if self._flatline_count >= 3:  # Require 3 consecutive checks
                self._trip_count += 1
                logger.warning(f"Dead sensor detected: variance={variance:.2e}")
                return False, f"Dead sensor: variance {variance:.2e} below threshold"
        else:
            self._flatline_count = 0
        
        return True, "OK"
    
    def check_invalid_spectrum(self, spectrum: np.ndarray) -> Tuple[bool, str]:
        """
        Check for invalid spectrum (NaN, zeros, corruption).
        
        Args:
            spectrum: Intensity or full spectrum array
            
        Returns:
            (is_safe, message)
        """
        total = len(spectrum)
        if total == 0:
            self._trip_count += 1
            return False, "Empty spectrum"
        
        # Check NaN ratio
        nan_count = np.isnan(spectrum).sum()
        nan_ratio = nan_count / total
        if nan_ratio > self.nan_tolerance:
            self._trip_count += 1
            logger.warning(f"High NaN ratio: {nan_ratio:.1%}")
            return False, f"Invalid spectrum: {nan_ratio:.1%} NaN values"
        
        # Check zero ratio
        zero_count = (spectrum == 0).sum()
        zero_ratio = zero_count / total
        if zero_ratio > self.zero_tolerance:
            self._trip_count += 1
            logger.warning(f"High zero ratio: {zero_ratio:.1%}")
            return False, f"Invalid spectrum: {zero_ratio:.1%} zero values"
        
        # Check for inf
        inf_count = np.isinf(spectrum).sum()
        if inf_count > 0:
            self._trip_count += 1
            return False, f"Invalid spectrum: {inf_count} infinite values"
        
        return True, "OK"
    
    def check_all(
        self,
        spectrum: np.ndarray,
        recent_samples: Optional[List[np.ndarray]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Run all safety checks.
        
        Returns:
            (all_safe, list of messages)
        """
        messages = []
        all_safe = True
        
        # Over-intensity check
        safe, msg = self.check_over_intensity(spectrum)
        if not safe:
            all_safe = False
            messages.append(msg)
        
        # Invalid spectrum check
        safe, msg = self.check_invalid_spectrum(spectrum)
        if not safe:
            all_safe = False
            messages.append(msg)
        
        # Dead sensor check (if samples available)
        if recent_samples is not None:
            safe, msg = self.check_dead_sensor(recent_samples)
            if not safe:
                all_safe = False
                messages.append(msg)
        
        return all_safe, messages
    
    def get_stats(self) -> dict:
        """Return safety statistics."""
        return {
            "trip_count": self._trip_count,
            "flatline_count": self._flatline_count,
            "last_intensity": self._last_intensity
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self._last_intensity = None
        self._flatline_count = 0
        # Don't reset trip_count - it's cumulative
