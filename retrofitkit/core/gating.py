from typing import Dict, Any, List
from collections import deque


class GatingEngine:
    """Optimized gating engine for spectral threshold detection."""
    
    __slots__ = ('rules', 'window', '_peak_threshold_rules', '_slope_rules')
    
    def __init__(self, rules: List[Dict[str, Any]]) -> None:
        self.rules = rules
        self.window: deque[Dict[str, Any]] = deque(maxlen=50)
        
        # Pre-categorize rules for faster lookup
        self._peak_threshold_rules = [r for r in rules if r["name"] == "peak_threshold"]
        self._slope_rules = [r for r in rules if r["name"] == "slope_stop"]

    def update(self, spectrum: Dict[str, Any]) -> bool:
        """
        Update gating engine with new spectrum data.
        
        Args:
            spectrum: Dict with t, wavelengths, intensities, peak_nm, peak_intensity
            
        Returns:
            True if stop condition is triggered, False otherwise.
        """
        self.window.append(spectrum)
        peak_intensity = spectrum.get("peak_intensity", 0)
        
        # Check peak threshold rules (fast path - no window needed)
        for r in self._peak_threshold_rules:
            threshold = r["threshold"]
            if r["direction"] == "above":
                if peak_intensity >= threshold:
                    return True
            elif r["direction"] == "below":
                if peak_intensity <= threshold:
                    return True
        
        # Check slope rules (need at least 3 samples)
        if len(self.window) >= 3 and self._slope_rules:
            # Get last 5 samples (or all if fewer)
            samples = list(self.window)[-5:] if len(self.window) >= 5 else list(self.window)
            
            if len(samples) >= 2:
                # Calculate slope using first and last points (linear approximation)
                y_start = samples[0].get("peak_intensity", 0)
                y_end = samples[-1].get("peak_intensity", 0)
                t_start = samples[0].get("t", 0)
                t_end = samples[-1].get("t", 0)
                
                dt = t_end - t_start
                if dt > 1e-6:  # Avoid division by zero
                    slope = (y_end - y_start) / dt
                    
                    for r in self._slope_rules:
                        if slope <= r["slope_threshold"]:
                            return True
        
        return False
