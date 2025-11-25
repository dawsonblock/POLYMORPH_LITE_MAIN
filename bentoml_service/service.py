# service.py
# Trigger reload
import bentoml
import torch
import numpy as np
import scipy.signal
from datetime import datetime
from pmm_brain import StaticPseudoModeMemory
from scipy.signal import savgol_filter
from bentoml.io import JSON
from pydantic import BaseModel

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
    def _als_baseline(y, lam=1e4, p=0.001, niter=10):
        """Asymmetric Least Squares baseline (Eilers & Boelens)"""
        L = len(y)
        if L < 3: return y
        D = np.diff(np.eye(L), 2)
        w = np.ones(L)
        for i in range(niter):
            W = np.diag(w)
            # Fix: D @ D.T for correct shape (L, L)
            Z = W + lam * D @ D.T
            z = np.linalg.solve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        return y - z

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
        
        # Simple encoder (trainable part — load your checkpoint or keep fixed)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128)
        ).to(self.device)
        self.encoder.eval()

    @bentoml.api
    async def infer(self, spectrum: np.ndarray) -> dict:
        with torch.no_grad():
            # Preprocess using the production pipeline
            x = RamanPreprocessor.preprocess(spectrum).unsqueeze(0).to(self.device)
            
            # Encode
            latent = self.encoder(x)
            
            # PMM Inference
            _, comp = self.brain(latent)
            
            # Capture state before update
            prev_polys = len(self.brain.poly_tracker)
            
            # Explicit Updates (Stateful)
            self.brain.apply_explicit_updates()
            
            # Check for new polymorphs
            curr_polys = len(self.brain.poly_tracker)
            new_poly = None
            if curr_polys > prev_polys:
                # Find the new entry (heuristic: latest added)
                # In a real scenario, we'd return it from apply_explicit_updates or store it
                # Here we iterate to find the one with the latest timestamp
                sorted_polys = sorted(
                    self.brain.poly_tracker.values(), 
                    key=lambda x: x['first_seen'], 
                    reverse=True
                )
                if sorted_polys:
                    new_poly = sorted_polys[0]['name']
            
            return {
                "active_modes": int(self.brain.n_active_modes),
                "polymorphs_found": curr_polys,
                "predicted_finish_sec": 1800,  # replace with real predictor
                "new_polymorph": new_poly,
                "status": "crystallizing" if new_poly else "stable",
                "timestamp": datetime.utcnow().isoformat()
            }
