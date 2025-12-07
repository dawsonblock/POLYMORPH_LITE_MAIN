"""
Tests for Gating Engine with Hysteresis.

Tests peak threshold with consecutive samples, slope detection,
cooldown periods, and state management.
"""

import pytest
import numpy as np
from retrofitkit.core.gating import GatingEngine, GatingState


class TestGatingHysteresis:
    """Test gating hysteresis behavior."""

    @pytest.fixture
    def basic_rules(self):
        """Basic threshold rules."""
        return [
            {
                "name": "peak_threshold",
                "direction": "above",
                "threshold": 100,
                "consecutive": 3,  # Require 3 consecutive samples
            }
        ]

    @pytest.fixture
    def slope_rules(self):
        """Slope-based rules."""
        return [
            {
                "name": "slope_stop",
                "slope_threshold": -0.1,
                "window_size": 5,
                "smoothing": True,
            }
        ]

    def test_single_sample_does_not_trigger(self, basic_rules):
        """Test that single sample above threshold doesn't trigger."""
        engine = GatingEngine(basic_rules)
        
        result = engine.update({"t": 0, "peak_intensity": 150})
        assert result is False
        
        result = engine.update({"t": 1, "peak_intensity": 50})
        assert result is False

    def test_consecutive_samples_trigger(self, basic_rules):
        """Test that consecutive samples above threshold trigger."""
        engine = GatingEngine(basic_rules)
        
        # First two don't trigger
        assert engine.update({"t": 0, "peak_intensity": 150}) is False
        assert engine.update({"t": 1, "peak_intensity": 150}) is False
        
        # Third consecutive triggers
        assert engine.update({"t": 2, "peak_intensity": 150}) is True

    def test_broken_consecutive_resets_count(self, basic_rules):
        """Test that breaking the streak resets counter."""
        engine = GatingEngine(basic_rules)
        
        engine.update({"t": 0, "peak_intensity": 150})
        engine.update({"t": 1, "peak_intensity": 150})
        
        # Break the streak
        engine.update({"t": 2, "peak_intensity": 50})
        
        # Start again - should not trigger yet
        assert engine.update({"t": 3, "peak_intensity": 150}) is False
        assert engine.update({"t": 4, "peak_intensity": 150}) is False
        assert engine.update({"t": 5, "peak_intensity": 150}) is True

    def test_cooldown_prevents_retrigger(self, basic_rules):
        """Test that cooldown prevents immediate re-trigger."""
        engine = GatingEngine(basic_rules, cooldown_sec=10.0)
        
        # Trigger at t=2 (after 3 consecutive)
        engine.update({"t": 0, "peak_intensity": 150})
        engine.update({"t": 1, "peak_intensity": 150})
        assert engine.update({"t": 2, "peak_intensity": 150}) is True
        
        # Try to trigger again during cooldown (t=2 + 10 = 12 is when cooldown ends)
        engine.update({"t": 3, "peak_intensity": 150})
        engine.update({"t": 4, "peak_intensity": 150})
        assert engine.update({"t": 5, "peak_intensity": 150}) is False
        
        # After cooldown expires (t > 12), need 3 NEW consecutive samples
        engine.update({"t": 13, "peak_intensity": 150})  # 1st after cooldown
        engine.update({"t": 14, "peak_intensity": 150})  # 2nd after cooldown
        assert engine.update({"t": 15, "peak_intensity": 150}) is True  # 3rd triggers

    def test_slope_detection(self, slope_rules):
        """Test slope-based gating."""
        engine = GatingEngine(slope_rules)
        
        # Build up window with decreasing values
        for t in range(10):
            intensity = 100 - t * 5  # Decreasing
            result = engine.update({"t": t, "peak_intensity": intensity})
        
        # Should eventually trigger due to negative slope
        # Last few should trigger
        assert result is True

    def test_get_stats(self, basic_rules):
        """Test stats reporting."""
        engine = GatingEngine(basic_rules)
        
        # Trigger once
        for t in range(3):
            engine.update({"t": t, "peak_intensity": 150})
        
        stats = engine.get_stats()
        assert "window_size" in stats
        assert "rules" in stats
        assert stats["rules"][0]["trigger_count"] == 1

    def test_reset(self, basic_rules):
        """Test state reset."""
        engine = GatingEngine(basic_rules)
        
        engine.update({"t": 0, "peak_intensity": 150})
        engine.update({"t": 1, "peak_intensity": 150})
        
        engine.reset()
        
        # Should require full consecutive count again
        assert engine.update({"t": 2, "peak_intensity": 150}) is False


class TestGatingSafety:
    """Test gating safety integration."""

    def test_empty_rules(self):
        """Test engine with no rules."""
        engine = GatingEngine([])
        result = engine.update({"t": 0, "peak_intensity": 1000})
        assert result is False

    def test_below_threshold(self):
        """Test below threshold with consecutive requirement."""
        rules = [
            {
                "name": "peak_threshold",
                "direction": "below",
                "threshold": 50,
                "consecutive": 2,
            }
        ]
        engine = GatingEngine(rules)
        
        assert engine.update({"t": 0, "peak_intensity": 30}) is False
        assert engine.update({"t": 1, "peak_intensity": 30}) is True
