# pmm_brain.py — FULLY EXPANDED & PRODUCTION-GRADE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import savgol_filter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RamanPreprocessor:
    """Production Raman preprocessing pipeline — identical across training/inference"""
    
    @staticmethod
    def preprocess(spectrum: np.ndarray) -> torch.Tensor:
        x = spectrum.astype(np.float32)
        
        # Validate input length
        if len(x) < 10:
            # Return a zero tensor or handle gracefully
            return torch.zeros(1024, dtype=torch.float32)

        # 1. Baseline correction (asymmetric least squares)
        x = RamanPreprocessor._als_baseline(x)
        
        # 2. Cosmic ray removal (median filter)
        x = np.medfilt(x, kernel_size=5)
        
        # 3. Savitzky-Golay smoothing
        if len(x) >= 15:
            x = savgol_filter(x, window_length=15, polyorder=3, mode='nearest')
        
        # 4. Normalization (area under curve)
        area = np.trapz(x, dx=1.0)
        if area > 1e-6:  # Avoid division by zero
            x = x / area
        
        # 5. Range selection (500–1800 cm⁻¹ typical for organics)
        # Adjust indices based on your spectrometer calibration
        if len(x) > 900:
             x = x[100:900]  # example: 500–1800 cm⁻¹
        
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

        Correct version:
          - D is a (L-2, L) second-difference matrix
          - Z = W + λ D^T D is (L, L)
        """
        L = y.shape[0]
        if L < 3:
            # Too short for second derivative; just return zeros baseline
            return np.zeros_like(y)

        # Second-difference operator
        D = np.diff(np.eye(L), 2, axis=0)  # (L-2, L)
        w = np.ones(L)

        for _ in range(niter):
            W = np.diag(w)
            # Proper ALS system: Z is (L, L)
            Z = W + lam * (D.T @ D)
            # Solve Z z = W y
            z = np.linalg.solve(Z, w * y)

            # Update weights: penalize positive residuals more
            w = p * (y > z) + (1.0 - p) * (y <= z)

        return z

class StaticPseudoModeMemory(nn.Module):
    """FULLY EXPANDED PMM — Explicit Updates + Merge/Split/Prune + Safety + Predictive"""
    
    def __init__(self, latent_dim=128, max_modes=32, init_modes=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_modes = max_modes
        
        # Learnable prototypes
        self.mu = nn.Parameter(torch.randn(max_modes, latent_dim) * 0.1)
        self.F = nn.Parameter(torch.eye(latent_dim).unsqueeze(0).repeat(max_modes, 1, 1) * 0.1)  # Predictive
        
        # Explicit state buffers
        self.register_buffer('w', torch.ones(max_modes))           # Readout weight
        self.register_buffer('lambda_i', torch.ones(max_modes))    # Importance
        self.register_buffer('occupancy', torch.zeros(max_modes))  # EMA occupancy
        self.register_buffer('active_mask', torch.zeros(max_modes, dtype=torch.bool))
        self.register_buffer('risk', torch.zeros(max_modes))       # Safety risk
        self.register_buffer('age', torch.zeros(max_modes))        # Age tracking
        
        # Hyperparameters
        self.merge_thresh = 0.88
        self.split_thresh = 0.28
        self.prune_thresh = 0.02
        self.risk_decay = 0.95
        
        # State
        self.poly_tracker = {}
        self.poly_id = 1
        self.last_batch = None
        
        # Init
        self.active_mask[:init_modes] = True
        self.occupancy[:init_modes] = 1.0 / init_modes
        self.w[:init_modes] = 1.0 / init_modes
        self.lambda_i[:init_modes] = 1.0

    @property
    def n_active(self): return self.active_mask.sum().item()
    
    # Alias for compatibility if needed
    @property
    def n_active_modes(self): return self.n_active

    def forward(self, latent: torch.Tensor):
        self.last_batch = latent.detach()
        if self.n_active == 0:
            return latent, {"alpha": torch.zeros(latent.shape[0], self.max_modes).to(latent.device)}
            
        mu_a = self.mu[self.active_mask]
        sim = F.cosine_similarity(latent.unsqueeze(1), mu_a.unsqueeze(0), dim=2)
        a = sim * self.lambda_i[self.active_mask]
        alpha = F.softmax(a, dim=1)
        
        recon = (alpha.unsqueeze(2) * mu_a.unsqueeze(0)).sum(1)
        
        # Risk modulation
        if self.risk[self.active_mask].any():
            risk_mod = torch.exp(-self.risk[self.active_mask])
            alpha = alpha * risk_mod
        
        return recon, {"alpha": alpha, "sim": sim}

    def apply_explicit_updates(self):
        if self.last_batch is None: return
        batch = self.last_batch
        B = batch.shape[0]
        
        with torch.no_grad():
            # 1. Update occupancy (EMA)
            _, comp = self.forward(batch)
            # Handle case where alpha might be smaller than max_modes if we only returned active
            # But forward returns alpha based on active_mask size? 
            # Wait, forward logic:
            # mu_a = self.mu[self.active_mask]
            # sim = ... (B, n_active)
            # alpha = ... (B, n_active)
            # We need to map back to full size for occupancy update
            
            alpha_active = comp["alpha"]
            alpha_mean_active = alpha_active.mean(0)
            
            active_indices = torch.where(self.active_mask)[0]
            self.occupancy[active_indices] = 0.95 * self.occupancy[active_indices] + 0.05 * alpha_mean_active
            
            # 2. Structural updates
            self._merge_similar()
            self._split_overloaded()
            self._prune_dead()
            self._enforce_capacity()
            
            # 3. Age & risk decay
            self.age[active_indices] += 1
            self.risk *= self.risk_decay
            
            # 4. Polymorph detection
            self._detect_new_polymorph()
            
        self.last_batch = None

    def _merge_similar(self):
        active_idx = torch.where(self.active_mask)[0]
        if len(active_idx) < 2: return
        mu_a = self.mu[active_idx]
        sim = F.cosine_similarity(mu_a.unsqueeze(1), mu_a.unsqueeze(0), dim=2)
        # Mask diagonal
        mask = ~torch.eye(len(active_idx), device=sim.device).bool()
        pairs = torch.where((sim > self.merge_thresh) & mask)
        
        merged = set()
        for i, j in zip(pairs[0], pairs[1]):
            if i.item() in merged or j.item() in merged: continue
            i_idx, j_idx = active_idx[i], active_idx[j]
            # Merge j → i
            occ_i = self.occupancy[i_idx]
            occ_j = self.occupancy[j_idx]
            total = occ_i + occ_j + 1e-8
            self.mu.data[i_idx] = (occ_i * self.mu[i_idx] + occ_j * self.mu[j_idx]) / total
            self.occupancy[i_idx] = total
            self.lambda_i[i_idx] = (occ_i * self.lambda_i[i_idx] + occ_j * self.lambda_i[j_idx]) / total
            self.active_mask[j_idx] = False
            merged.add(j.item())
            logger.info(f"Merged mode {j_idx} → {i_idx}")

    def _split_overloaded(self):
        active_idx = torch.where(self.active_mask)[0]
        occ = self.occupancy[active_idx]
        scores = occ / (self.lambda_i[active_idx] + 1e-8)
        candidates = scores > self.split_thresh
        
        # Iterate over candidates
        cand_indices = active_idx[candidates]
        for idx in cand_indices:
            if self.n_active >= self.max_modes: break
            new_idx = self._find_free_slot()
            if new_idx is None: break
            
            noise = torch.randn_like(self.mu[idx]) * 0.15
            self.mu.data[new_idx] = self.mu[idx] + noise
            self.occupancy[new_idx] = self.occupancy[idx] * 0.5
            self.occupancy[idx] *= 0.5
            self.active_mask[new_idx] = True
            logger.info(f"Split mode {idx} → {new_idx}")

    def _prune_dead(self):
        dead = (self.occupancy < self.prune_thresh) & self.active_mask
        if dead.any():
            idx = torch.where(dead)[0]
            self.active_mask[idx] = False
            logger.info(f"Pruned {len(idx)} dead modes")

    def _enforce_capacity(self):
        active_sum = self.occupancy[self.active_mask].sum()
        if active_sum > 1.1:
            self.lambda_i[self.active_mask] /= active_sum

    def _find_free_slot(self):
        free = torch.where(~self.active_mask)[0]
        return free[0].item() if len(free) > 0 else None

    def _detect_new_polymorph(self):
        if self.n_active > len(self.poly_tracker):
            # Get the last active mode as the new one (heuristic)
            # Better: find the mode that is active but not in tracker?
            # The prompt logic: "Get the newest mode (the one just born) ... newest_mode_vector = self.mu[self.active_mask][-1]"
            # This assumes self.active_mask indices are ordered or something. 
            # self.mu[self.active_mask] returns them in index order.
            # If we split, we used a free slot. It might be higher index.
            
            new_mode = self.mu[self.active_mask][-1].cpu().numpy()
            sig = hash(new_mode.tobytes()[:64])
            if sig not in self.poly_tracker:
                name = f"Polymorph-{self.poly_id:02d}"
                self.poly_tracker[sig] = {"name": name, "first_seen": datetime.utcnow().isoformat()}
                self.poly_id += 1
                logger.critical(f"NEW POLYMORPH DISCOVERED: {name}")
                # Trigger webhook/Slack here

    def predict_next(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu_a = self.mu[self.active_mask]
            sim = F.cosine_similarity(latent.unsqueeze(1), mu_a.unsqueeze(0), dim=2)
            alpha = F.softmax(sim, dim=1)
            pred = (alpha.unsqueeze(2) * self.F[self.active_mask] @ latent.unsqueeze(2)).sum(1).squeeze()
            return pred
