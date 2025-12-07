"""
Gating Engine with Hysteresis, Cooldown, and Safety Features.

Provides stateful gating logic for crystallization control with:
- Peak threshold with consecutive sample requirement (hysteresis)
- Slope detection with moving window smoothing
- Cooldown periods after triggers
"""

from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class GatingState:
    """Tracks stateful gating information."""
    consecutive_above: int = 0
    consecutive_below: int = 0
    last_trigger_time: Optional[float] = None
    trigger_count: int = 0
    cooldown_remaining: float = 0.0


class GatingEngine:
    """
    Enhanced gating engine with hysteresis and cooldown.
    
    Features:
    - Peak threshold with N consecutive samples requirement
    - Slope detection with configurable window and smoothing
    - Cooldown period between triggers
    - Detailed trigger logging
    """
    
    def __init__(
        self,
        rules: List[Dict[str, Any]],
        window_size: int = 50,
        cooldown_sec: float = 5.0
    ) -> None:
        self.rules = rules
        self.window: deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.cooldown_sec = cooldown_sec
        self.states: Dict[str, GatingState] = {}
        
        # Initialize state for each rule
        for r in rules:
            rule_name = r.get("name", f"rule_{len(self.states)}")
            self.states[rule_name] = GatingState()
    
    def update(self, spectrum: Dict[str, Any]) -> bool:
        """
        Process new spectrum and check gating conditions.
        
        Args:
            spectrum: Dict with keys:
                - t: timestamp in seconds
                - peak_intensity: peak intensity value
                - wavelengths: (optional) list of wavelengths
                - intensities: (optional) list of intensities
        
        Returns:
            True if any gating condition triggers
        """
        self.window.append(spectrum)
        current_time = spectrum.get("t", 0.0)
        
        # Update cooldowns
        for state in self.states.values():
            if state.last_trigger_time is not None:
                elapsed = current_time - state.last_trigger_time
                state.cooldown_remaining = max(0, self.cooldown_sec - elapsed)
        
        for rule in self.rules:
            rule_name = rule.get("name", "unnamed")
            state = self.states.get(rule_name, GatingState())
            
            # Skip if in cooldown
            if state.cooldown_remaining > 0:
                continue
            
            triggered = self._check_rule(rule, spectrum, state)
            
            if triggered:
                state.last_trigger_time = current_time
                state.trigger_count += 1
                state.cooldown_remaining = self.cooldown_sec
                # Reset consecutive counters after trigger
                state.consecutive_above = 0
                state.consecutive_below = 0
                logger.info(f"Gating triggered: {rule_name} (count={state.trigger_count})")
                return True
        
        return False
    
    def _check_rule(
        self,
        rule: Dict[str, Any],
        spectrum: Dict[str, Any],
        state: GatingState
    ) -> bool:
        """Check a single rule with hysteresis."""
        rule_type = rule.get("name", "")
        
        if rule_type == "peak_threshold":
            return self._check_peak_threshold(rule, spectrum, state)
        elif rule_type == "slope_stop":
            return self._check_slope_stop(rule, state)
        
        return False
    
    def _check_peak_threshold(
        self,
        rule: Dict[str, Any],
        spectrum: Dict[str, Any],
        state: GatingState
    ) -> bool:
        """
        Check peak threshold with hysteresis.
        
        Requires N consecutive samples above/below threshold.
        """
        threshold = rule.get("threshold", 0)
        direction = rule.get("direction", "above")
        consecutive_required = rule.get("consecutive", 1)  # Hysteresis: default 1
        
        intensity = spectrum.get("peak_intensity", 0)
        
        if direction == "above":
            if intensity >= threshold:
                state.consecutive_above += 1
                state.consecutive_below = 0
            else:
                state.consecutive_above = 0
            
            return state.consecutive_above >= consecutive_required
        
        elif direction == "below":
            if intensity <= threshold:
                state.consecutive_below += 1
                state.consecutive_above = 0
            else:
                state.consecutive_below = 0
            
            return state.consecutive_below >= consecutive_required
        
        return False
    
    def _check_slope_stop(
        self,
        rule: Dict[str, Any],
        state: GatingState
    ) -> bool:
        """
        Check slope condition with moving window smoothing.
        """
        if len(self.window) < 5:
            return False
        
        slope_threshold = rule.get("slope_threshold", 0)
        window_size = rule.get("window_size", 5)
        smoothing = rule.get("smoothing", False)
        
        # Get recent samples
        recent = list(self.window)[-window_size:]
        y = np.array([x.get("peak_intensity", 0) for x in recent])
        t = np.array([x.get("t", i) for i, x in enumerate(recent)])
        
        # Optional smoothing
        if smoothing and len(y) >= 3:
            y = np.convolve(y, np.ones(3)/3, mode='valid')
            t = t[:len(y)]
        
        if len(t) < 2 or (t[-1] - t[0]) < 1e-6:
            return False
        
        # Compute slope
        slope = (y[-1] - y[0]) / (t[-1] - t[0])
        
        return slope <= slope_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Return gating statistics."""
        return {
            "window_size": len(self.window),
            "rules": [
                {
                    "name": r.get("name"),
                    "trigger_count": self.states.get(r.get("name"), GatingState()).trigger_count,
                    "cooldown_remaining": self.states.get(r.get("name"), GatingState()).cooldown_remaining
                }
                for r in self.rules
            ]
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self.window.clear()
        for state in self.states.values():
            state.consecutive_above = 0
            state.consecutive_below = 0
            state.last_trigger_time = None
            state.cooldown_remaining = 0.0
