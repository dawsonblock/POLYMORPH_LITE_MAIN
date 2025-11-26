# service.py
# Trigger reload
import bentoml
import torch
import numpy as np
import scipy.signal
import hashlib
from datetime import datetime, timedelta
from pmm_brain import StaticPseudoModeMemory
from scipy.signal import savgol_filter
from bentoml.io import JSON
from pydantic import BaseModel
from typing import Dict, Any, Optional

class RamanPreprocessor:
    """Production Raman preprocessing pipeline — identical across training/inference"""
    
    @staticmethod
    def preprocess(x: list) -> torch.Tensor:
        x = np.array(x, dtype=np.float32)
        
        # Validate input length
        if len(x) < 10:
            return torch.zeros(1024, dtype=torch.float32)

        # 1. Denoise
        x = scipy.signal.medfilt(x, kernel_size=5)
        
        # 2. Baseline correction (asymmetric least squares)
        x = RamanPreprocessor._als_baseline(x)
        
        
        # 3. Savitzky-Golay smoothing
        if len(x) >= 15:
            x = savgol_filter(x, window_length=15, polyorder=3, mode='nearest')
        
        # 4. Normalization (area under curve)
        area = np.trapz(x, dx=1.0)
        if area > 1e-6:
            x = x / area
        
        # 5. Range selection (500–1800 cm⁻¹ typical for organics)
        if len(x) > 900:
             x = x[100:900]
        
        # 6. Pad/truncate to fixed size
        target_len = 1024
        if len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)), mode='constant')
        else:
            x = x[:target_len]
            
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction (Eilers & Boelens, 2005).
        Correct version: Z = W + λ D^T D
        """
        L = y.shape[0]
        if L < 3:
            return np.zeros_like(y)

        D = np.diff(np.eye(L), 2, axis=0)  # (L-2, L)
        w = np.ones(L)

        for _ in range(niter):
            W = np.diag(w)
            Z = W + lam * (D.T @ D)
            z = np.linalg. solve(Z, w * y)
            w = p * (y > z) + (1.0 - p) * (y <= z)

        return z

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
        self._session_created_at = datetime.utcnow()
        self._last_poly_change_at = datetime.utcnow()
        self._prev_intensity = 0.0

    def _compute_polymorph_id(self, latent: np.ndarray) -> str:
        """Stable polymorph ID based on SHA-256 hash of latent vector."""
        h = hashlib.sha256(latent.tobytes()).hexdigest()
        return h[:16]  # 64 bits

    def _estimate_finish_time(self, created_at: datetime, last_change_at: datetime, max_horizon_sec: int = 3600) -> int:
        """Simple, honest heuristic for time estimation."""
        now = datetime.utcnow()
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
        now = datetime.utcnow()

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
                        "first_seen": datetime.utcnow().isoformat()
                    }
                    new_poly_id = poly_id

            # Compute simple slope (intensity change)
            current_intensity = float(x.sum())
            slope = current_intensity - self._prev_intensity
            self._prev_intensity = current_intensity

            return self._build_response(self.brain.n_active, new_poly_id, slope)
