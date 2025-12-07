"""
Tests for PMM Checkpointing.

Tests save_state(), load_state(), reset_state(), and state persistence.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "bentoml_service"))

from pmm_brain import StaticPseudoModeMemory


class TestPMMCheckpointing:
    """Test PMM checkpoint save/load functionality."""

    @pytest.fixture
    def pmm(self):
        """Create a fresh PMM instance."""
        return StaticPseudoModeMemory(latent_dim=128, max_modes=32, init_modes=4)

    @pytest.fixture
    def tmp_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_save_state_creates_file(self, pmm, tmp_dir):
        """Test that save_state creates a checkpoint file."""
        path = pmm.save_state(tmp_dir, org_id="test_org")
        assert Path(path).exists()
        assert path.endswith(".npz")

    def test_save_state_with_explicit_path(self, pmm, tmp_dir):
        """Test saving to an explicit file path."""
        explicit_path = tmp_dir / "my_checkpoint.npz"
        path = pmm.save_state(explicit_path)
        assert Path(path) == explicit_path
        assert explicit_path.exists()

    def test_load_state_restores_values(self, pmm, tmp_dir):
        """Test that load_state restores all state correctly."""
        # Modify PMM state
        with torch.no_grad():
            pmm.mu.data[0] = torch.ones(128) * 0.5
            pmm.occupancy[0] = 0.99
            pmm.risk[0] = 0.42
            pmm.age[0] = 100
            pmm.poly_id = 5
        pmm.poly_tracker = {12345: {"name": "Polymorph-01", "first_seen": "2024-01-01"}}

        # Save
        path = pmm.save_state(tmp_dir, org_id="test_org")

        # Create new PMM and load
        pmm2 = StaticPseudoModeMemory(latent_dim=128, max_modes=32)
        metadata = pmm2.load_state(path)

        # Verify restoration
        assert torch.allclose(pmm.mu[0], pmm2.mu[0])
        assert pmm2.occupancy[0].item() == pytest.approx(0.99)
        assert pmm2.risk[0].item() == pytest.approx(0.42)
        assert pmm2.age[0].item() == 100
        assert pmm2.poly_id == 5
        assert 12345 in pmm2.poly_tracker or "12345" in pmm2.poly_tracker

        # Verify metadata
        assert metadata["org_id"] == "test_org"
        assert "timestamp" in metadata
        assert metadata["version"] == "1.0"

    def test_load_state_from_bytes(self, pmm, tmp_dir):
        """Test loading state from bytes (for HTTP upload)."""
        path = pmm.save_state(tmp_dir)
        
        with open(path, "rb") as f:
            checkpoint_bytes = f.read()
        
        pmm2 = StaticPseudoModeMemory(latent_dim=128, max_modes=32)
        metadata = pmm2.load_state(checkpoint_bytes)
        
        assert pmm2.n_active == pmm.n_active

    def test_load_state_file_not_found(self, pmm):
        """Test that load_state raises for missing file."""
        with pytest.raises(FileNotFoundError):
            pmm.load_state("/nonexistent/path/checkpoint.npz")

    def test_reset_state(self, pmm):
        """Test that reset_state clears all learned state."""
        # Modify state
        pmm.poly_id = 100
        pmm.poly_tracker = {"a": {"b": "c"}}
        with torch.no_grad():
            pmm.occupancy.fill_(0.5)
            pmm.risk.fill_(0.5)
            pmm.age.fill_(50)

        # Reset
        pmm.reset_state(init_modes=2)

        # Verify reset
        assert pmm.n_active == 2
        assert pmm.poly_id == 1
        assert len(pmm.poly_tracker) == 0
        assert pmm.occupancy[0].item() == pytest.approx(0.5, abs=0.01)

    def test_get_mode_stats(self, pmm):
        """Test get_mode_stats returns correct format."""
        stats = pmm.get_mode_stats()
        
        assert "n_active" in stats
        assert "max_modes" in stats
        assert "active_indices" in stats
        assert "occupancy" in stats
        assert "risk" in stats
        assert "age" in stats
        assert "poly_count" in stats
        
        assert stats["n_active"] == 4
        assert stats["max_modes"] == 32
        assert len(stats["active_indices"]) == 4

    def test_get_poly_ids(self, pmm):
        """Test get_poly_ids returns copy of tracker."""
        pmm.poly_tracker = {"hash1": {"name": "P1"}}
        
        ids = pmm.get_poly_ids()
        assert ids == pmm.poly_tracker
        
        # Verify it's a copy
        ids["new"] = {"name": "P2"}
        assert "new" not in pmm.poly_tracker


class TestPMMStability:
    """Test PMM numerical stability."""

    @pytest.fixture
    def pmm(self):
        return StaticPseudoModeMemory(latent_dim=128, max_modes=32, init_modes=4)

    def test_forward_with_zero_vector(self, pmm):
        """Test forward pass handles zero vectors."""
        zero_latent = torch.zeros(1, 128)
        recon, comp = pmm(zero_latent)
        
        # Should not produce NaN
        assert not torch.isnan(recon).any()
        assert not torch.isnan(comp["alpha"]).any()

    def test_forward_with_large_values(self, pmm):
        """Test forward pass handles large values."""
        large_latent = torch.ones(1, 128) * 1000
        recon, comp = pmm(large_latent)
        
        assert not torch.isnan(recon).any()
        assert not torch.isinf(recon).any()

    def test_merge_normalizes_mu(self, pmm):
        """Test that merge operation maintains normalized mu."""
        # Set two modes to be very similar
        with torch.no_grad():
            pmm.mu.data[0] = torch.ones(128)
            pmm.mu.data[1] = torch.ones(128) * 1.01
            pmm.occupancy[0] = 0.5
            pmm.occupancy[1] = 0.5
        
        # Process to trigger merge check
        latent = torch.randn(10, 128)
        pmm(latent)
        pmm.apply_explicit_updates()
        
        # Check mu norms are reasonable
        for i in range(pmm.n_active):
            idx = torch.where(pmm.active_mask)[0][i]
            norm = pmm.mu[idx].norm().item()
            assert 0 < norm < 1000, f"Mode {idx} has abnormal norm: {norm}"
