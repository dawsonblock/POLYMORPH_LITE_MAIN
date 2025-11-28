# service.py
# Trigger reload
import bentoml
import torch
import numpy as np
import scipy.signal
import hashlib
from datetime import datetime, timedelta, timezone
from pmm_brain import StaticPseudoModeMemory, RamanPreprocessor  # Import shared preprocessor
from bentoml.io import JSON
from pydantic import BaseModel
from typing import Dict, Any, Optional


@bentoml.service(
    name="polymorph-crystallization-ai",
    resources={"gpu": 1, "memory": "8Gi"},
    traffic={"timeout": 300, "concurrency": 8},
    workers=1  # Critical:stateful model → 1 worker
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

    @bentoml.api
    async def infer(self, spectrum: np.ndarray) -> dict:
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
