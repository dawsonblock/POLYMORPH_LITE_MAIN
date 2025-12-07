# service.py
# BentoML Service for POLYMORPH-LITE AI Brain
import bentoml
import torch
import numpy as np
import scipy.signal
import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pmm_brain import StaticPseudoModeMemory, RamanPreprocessor
from bentoml.io import JSON
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Environment configuration
CHECKPOINT_DIR = Path(os.environ.get("PMM_CHECKPOINT_DIR", "/app/checkpoints"))
AUTO_LOAD_CHECKPOINT = os.environ.get("PMM_AUTO_LOAD", "true").lower() == "true"


@bentoml.service(
    name="polymorph-crystallization-ai",
    resources={"gpu": 1, "memory": "8Gi"},
    traffic={"timeout": 300, "concurrency": 8},
    workers=1  # Critical: stateful model → 1 worker
)
class PolymorphService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.brain = StaticPseudoModeMemory(latent_dim=128, max_modes=32).to(self.device)
        self.brain.eval()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128)
        ).to(self.device)
        self.encoder.eval()
        
        # Tracking for honest predictions
        self.poly_tracker: Dict[str, Dict[str, Any]] = {}
        self._session_created_at = datetime.now(timezone.utc)
        self._last_poly_change_at = datetime.now(timezone.utc)
        self._prev_intensity = 0.0
        
        # Auto-load checkpoint on startup
        self._try_load_latest_checkpoint()

    def _try_load_latest_checkpoint(self) -> None:
        """Attempt to load the latest checkpoint on startup."""
        if not AUTO_LOAD_CHECKPOINT:
            logger.info("Auto-load checkpoint disabled")
            return
            
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(CHECKPOINT_DIR.glob("*.npz"), reverse=True)
        
        if checkpoints:
            latest = checkpoints[0]
            try:
                metadata = self.brain.load_state(latest)
                logger.info(f"Loaded checkpoint: {latest.name} (org: {metadata.get('org_id', 'N/A')})")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {latest}: {e}")
        else:
            logger.info("No checkpoints found, starting fresh")

    def _compute_polymorph_id(self, latent: np.ndarray) -> str:
        """Stable polymorph ID based on SHA-256 hash of latent vector."""
        h = hashlib.sha256(latent.tobytes()).hexdigest()
        return h[:16]  # 64 bits

    def _estimate_finish_time(self, created_at: datetime, last_change_at: datetime, max_horizon_sec: int = 3600) -> int:
        """Simple, honest heuristic for time estimation."""
        now = datetime.now(timezone.utc)
        total_age = (now - created_at).total_seconds()
        since_change = (now - last_change_at).total_seconds()

        if total_age < 60:
            return min(max_horizon_sec, 1800)

        if since_change > 300:
            return max(60, int(max_horizon_sec * 0.1))

        frac = max(0.1, min(1.0, 1.0 - since_change / max(300.0, total_age)))
        return int(max(60, frac * max_horizon_sec))

    def _build_response(self, active_modes: int, new_poly_id: Optional[str], spectra_slope: float) -> Dict[str, Any]:
        """Build honest AI response."""
        now = datetime.now(timezone.utc)

        if new_poly_id is not None:
            self._last_poly_change_at = now

        # Status logic
        if abs(spectra_slope) < 1e-3:
            status = "stable"
        elif spectra_slope > 0:
            status = "crystallizing"
        else:
            status = "degrading"

        predicted_finish_sec = self._estimate_finish_time(
            self._session_created_at,
            self._last_poly_change_at,
            max_horizon_sec=3600,
        )

        return {
            "active_modes": int(active_modes),
            "polymorphs_found": len(self.poly_tracker),
            "predicted_finish_sec": predicted_finish_sec,
            "new_polymorph": new_poly_id,
            "status": status,
            "timestamp": now.isoformat(),
        }

    # -------------------------------------------------------------------------
    # Core Inference API
    # -------------------------------------------------------------------------

    @bentoml.api
    async def infer(self, spectrum: np.ndarray) -> dict:
        """Process a spectrum and return AI predictions."""
        with torch.no_grad():
            x = RamanPreprocessor.preprocess(spectrum).unsqueeze(0).to(self.device)
            latent = self.encoder(x)
            _, comp = self.brain(latent)

            prev_poly_count = len(self.poly_tracker)
            self.brain.apply_explicit_updates()

            # Detect new polymorph with stable ID
            new_poly_id = None
            if self.brain.n_active > prev_poly_count:
                new_mode = self.brain.mu[self.brain.active_mask][-1].cpu().numpy().astype(np.float32).ravel()
                poly_id = self._compute_polymorph_id(new_mode)
                if poly_id not in self.poly_tracker:
                    self.poly_tracker[poly_id] = {
                        "id": poly_id,
                        "first_seen": datetime.now(timezone.utc).isoformat()
                    }
                    new_poly_id = poly_id

            # Compute simple slope (intensity change)
            current_intensity = float(x.sum())
            slope = current_intensity - self._prev_intensity
            self._prev_intensity = current_intensity

            return self._build_response(self.brain.n_active, new_poly_id, slope)

    # -------------------------------------------------------------------------
    # Memory Management APIs
    # -------------------------------------------------------------------------

    @bentoml.api
    async def reset_memory(self, init_modes: int = 4) -> dict:
        """Reset PMM to initial state."""
        self.brain.reset_state(init_modes)
        self.poly_tracker = {}
        self._session_created_at = datetime.now(timezone.utc)
        self._last_poly_change_at = datetime.now(timezone.utc)
        return {"status": "ok", "message": f"Memory reset with {init_modes} initial modes"}

    @bentoml.api
    async def export_memory(self, org_id: Optional[str] = None) -> dict:
        """Export current memory state to checkpoint."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = self.brain.save_state(CHECKPOINT_DIR, org_id=org_id)
        return {
            "status": "ok",
            "path": path,
            "org_id": org_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @bentoml.api
    async def import_memory(self, checkpoint_path: str) -> dict:
        """Import memory state from checkpoint file."""
        try:
            metadata = self.brain.load_state(checkpoint_path)
            return {"status": "ok", "metadata": metadata}
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": f"Failed to load: {e}"}

    @bentoml.api
    async def modes(self) -> dict:
        """Return live mode statistics."""
        return self.brain.get_mode_stats()

    @bentoml.api
    async def poly_ids(self) -> dict:
        """Return discovered polymorph IDs."""
        return {
            "poly_ids": list(self.poly_tracker.keys()),
            "details": self.poly_tracker,
            "count": len(self.poly_tracker)
        }

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    @bentoml.api
    async def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "device": self.device,
            "active_modes": self.brain.n_active,
            "uptime_sec": (datetime.now(timezone.utc) - self._session_created_at).total_seconds()
        }
