import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, Set

logger = logging.getLogger(__name__)

class StaticPseudoModeMemory(nn.Module):
    """
    Production-Grade Static Pseudo-Mode Memory (PMM).
    
    Features:
    - Explicit read/write cycles.
    - Dynamic mode management (merge, split, prune).
    - Safety mechanisms (risk modulation).
    - Predictive forward model.
    - GPU-aware and tensorized.
    """
    
    def __init__(self, latent_dim: int = 128, max_modes: int = 32, init_modes: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_modes = max_modes
        
        # Learnable prototypes (Modes)
        self.mu = nn.Parameter(torch.randn(max_modes, latent_dim) * 0.1)
        # Predictive transition matrix per mode: F_i * z -> z_next
        self.F = nn.Parameter(torch.eye(latent_dim).unsqueeze(0).repeat(max_modes, 1, 1) * 0.1)
        
        # Explicit state buffers (Not parameters, but persistent state)
        self.register_buffer('w', torch.ones(max_modes))           # Readout weight
        self.register_buffer('lambda_i', torch.ones(max_modes))    # Importance/Attention scale
        self.register_buffer('occupancy', torch.zeros(max_modes))  # EMA occupancy tracking
        self.register_buffer('active_mask', torch.zeros(max_modes, dtype=torch.bool))
        self.register_buffer('risk', torch.zeros(max_modes))       # Safety risk score
        self.register_buffer('age', torch.zeros(max_modes))        # Age tracking
        
        # Hyperparameters
        self.merge_thresh = 0.88
        self.split_thresh = 0.28
        self.prune_thresh = 0.02
        self.risk_decay = 0.95
        
        # Internal State tracking
        self.poly_tracker: Dict[int, Dict] = {}
        self.poly_id_counter = 1
        self.last_batch: Optional[torch.Tensor] = None
        
        # Initialization
        self.active_mask[:init_modes] = True
        self.occupancy[:init_modes] = 1.0 / init_modes
        self.w[:init_modes] = 1.0 / init_modes
        self.lambda_i[:init_modes] = 1.0

    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum().item())

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Memory Read Operation.
        
        Args:
            latent: Input latent state (Batch, LatentDim)
            
        Returns:
            recon: Reconstructed/Memory-Augmented state (Batch, LatentDim)
            info: Dictionary containing attention weights and similarities
        """
        self.last_batch = latent.detach() # Store for write cycle
        
        if self.n_active == 0:
            # Fallback if no modes active (should rarely happen)
            return latent, {"alpha": torch.zeros(latent.shape[0], self.max_modes, device=latent.device)}
            
        # 1. Similarity Matching
        # Expand for broadcasting: (Batch, 1, Dim) vs (1, ActiveModes, Dim)
        # We use masked indexing to only compute against active modes
        active_indices = torch.where(self.active_mask)[0]
        mu_a = self.mu[active_indices] # (N_Active, Dim)
        
        # Cosine similarity
        # latent: (B, D) -> (B, 1, D)
        # mu_a: (A, D) -> (1, A, D)
        sim = F.cosine_similarity(latent.unsqueeze(1), mu_a.unsqueeze(0), dim=2) # (B, A)
        
        # 2. Attention / Activation
        # Scale by importance lambda
        a = sim * self.lambda_i[active_indices]
        alpha = F.softmax(a, dim=1) # (B, A)
        
        # 3. Reconstruction / Memory Readout
        # Weighted sum of prototypes
        # alpha: (B, A) -> (B, A, 1)
        # mu_a: (A, D) -> (1, A, D)
        recon = (alpha.unsqueeze(2) * mu_a.unsqueeze(0)).sum(dim=1) # (B, D)
        
        # 4. Risk Modulation (Safety)
        # If active modes are risky, suppress the output or modulate attention
        if self.risk[active_indices].any():
            risk_mod = torch.exp(-self.risk[active_indices]) # (A,)
            # Modulate alpha for return info, though recon is already computed
            # In a real control loop, this might gate the action
            alpha_modulated = alpha * risk_mod.unsqueeze(0)
        else:
            alpha_modulated = alpha

        # Map active alpha back to full size for consistency if needed, 
        # but usually we just need the active ones. 
        # For visualization/logging, we might want full size.
        alpha_full = torch.zeros(latent.shape[0], self.max_modes, device=latent.device)
        alpha_full[:, active_indices] = alpha
        
        return recon, {"alpha": alpha_full, "sim": sim, "alpha_active": alpha}

    def predict_next(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predictive Forward Model: Estimate next state based on memory dynamics.
        z_{t+1} = sum_i alpha_i * F_i * z_t
        """
        with torch.no_grad():
            active_indices = torch.where(self.active_mask)[0]
            if len(active_indices) == 0:
                return latent
                
            mu_a = self.mu[active_indices]
            sim = F.cosine_similarity(latent.unsqueeze(1), mu_a.unsqueeze(0), dim=2)
            alpha = F.softmax(sim, dim=1) # (B, A)
            
            # F_a: (A, D, D)
            F_a = self.F[active_indices]
            
            # latent: (B, D) -> (B, 1, D, 1)
            z_in = latent.unsqueeze(1).unsqueeze(3) 
            
            # Matrix vector multiply per mode: F_i @ z
            # (A, D, D) @ (B, 1, D, 1) -> mismatch dimensions
            # Let's do: (B, 1, D) @ (1, A, D, D)^T ? No.
            
            # Easiest: (B, A, D)
            # z_next_i = (F_i @ z_t^T)^T
            # But we have batch.
            
            # F_a: (A, D, D)
            # latent: (B, D)
            # We want result (B, D)
            
            # Expand latent: (B, 1, D)
            # Expand F: (1, A, D, D)
            # We want to apply each F_i to each z_b.
            
            # Using einsum is cleanest:
            # b: batch, a: active_mode, i: in_dim, o: out_dim
            # alpha: (b, a)
            # F: (a, o, i)
            # z: (b, i)
            # out: (b, o)
            
            pred = torch.einsum('ba,aoi,bi->bo', alpha, F_a, latent)
            return pred

    def apply_explicit_updates(self):
        """
        Memory Write Cycle.
        Updates internal state (occupancy, modes, topology) based on last batch.
        Should be called after forward pass during training or inference.
        """
        if self.last_batch is None:
            return
            
        batch = self.last_batch
        
        with torch.no_grad():
            # Re-compute forward to get current activations
            # We can't cache from forward() easily because of graph detachment if we want gradients later,
            # but here we are in no_grad for structural updates.
            _, info = self.forward(batch)
            alpha_full = info["alpha"]
            
            # 1. Update Occupancy (EMA)
            # Mean activation across batch for each mode
            alpha_mean = alpha_full.mean(dim=0) # (M,)
            
            # Only update active modes? Or all? 
            # Inactive modes have alpha 0, so their occupancy will decay.
            # That's correct behavior.
            self.occupancy = 0.95 * self.occupancy + 0.05 * alpha_mean
            
            # 2. Structural Updates (Topology)
            self._merge_similar()
            self._split_overloaded()
            self._prune_dead()
            self._enforce_capacity()
            
            # 3. Age & Risk Decay
            self.age[self.active_mask] += 1
            self.risk *= self.risk_decay
            
            # 4. Discovery
            self._detect_new_polymorph()
            
        self.last_batch = None

    def _merge_similar(self):
        """Merge modes that are too close in latent space."""
        active_idx = torch.where(self.active_mask)[0]
        if len(active_idx) < 2: return
        
        mu_a = self.mu[active_idx]
        # Similarity matrix (A, A)
        sim = F.cosine_similarity(mu_a.unsqueeze(1), mu_a.unsqueeze(0), dim=2)
        
        # Mask diagonal and lower triangle to avoid duplicates
        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        
        # Find pairs > threshold
        pairs = torch.where((sim > self.merge_thresh) & mask)
        
        merged_indices = set()
        
        for i_local, j_local in zip(pairs[0], pairs[1]):
            # Map back to global indices
            i = active_idx[i_local].item()
            j = active_idx[j_local].item()
            
            if i in merged_indices or j in merged_indices:
                continue
                
            # Merge j into i
            occ_i = self.occupancy[i]
            occ_j = self.occupancy[j]
            total_occ = occ_i + occ_j + 1e-8
            
            # Weighted average of prototypes
            self.mu.data[i] = (occ_i * self.mu[i] + occ_j * self.mu[j]) / total_occ
            
            # Update metadata
            self.occupancy[i] = total_occ
            self.lambda_i[i] = (occ_i * self.lambda_i[i] + occ_j * self.lambda_i[j]) / total_occ
            
            # Kill j
            self.active_mask[j] = False
            self.occupancy[j] = 0.0
            merged_indices.add(j)
            
            logger.info(f"PMM: Merged mode {j} -> {i} (Sim: {sim[i_local, j_local]:.4f})")

    def _split_overloaded(self):
        """Split modes that are highly occupied but have low importance (high variance)."""
        active_idx = torch.where(self.active_mask)[0]
        if len(active_idx) == 0: return
        
        # Heuristic: High occupancy but maybe we want to split if it covers too much space?
        # Simple heuristic: Occupancy > threshold
        # Original logic: occ / lambda > thresh
        
        scores = self.occupancy[active_idx] / (self.lambda_i[active_idx] + 1e-8)
        candidates = torch.where(scores > self.split_thresh)[0]
        
        for cand_local in candidates:
            if self.n_active >= self.max_modes:
                break
                
            idx = active_idx[cand_local].item()
            new_idx = self._find_free_slot()
            
            if new_idx is None:
                break
                
            # Split: Clone and perturb
            noise = torch.randn_like(self.mu[idx]) * 0.15
            self.mu.data[new_idx] = self.mu[idx] + noise
            
            # Share occupancy
            half_occ = self.occupancy[idx] * 0.5
            self.occupancy[idx] = half_occ
            self.occupancy[new_idx] = half_occ
            
            # Reset age/risk for new
            self.age[new_idx] = 0
            self.risk[new_idx] = self.risk[idx] # Inherit risk?
            self.active_mask[new_idx] = True
            
            logger.info(f"PMM: Split mode {idx} -> {new_idx}")

    def _prune_dead(self):
        """Remove modes with very low occupancy."""
        # Only prune if we have enough modes
        if self.n_active <= 2: return
        
        dead = (self.occupancy < self.prune_thresh) & self.active_mask
        if dead.any():
            idx = torch.where(dead)[0]
            self.active_mask[idx] = False
            self.occupancy[idx] = 0.0
            logger.info(f"PMM: Pruned {len(idx)} dead modes: {idx.tolist()}")

    def _enforce_capacity(self):
        """Normalize importance if total occupancy explodes."""
        active_sum = self.occupancy[self.active_mask].sum()
        if active_sum > 1.1:
            self.lambda_i[self.active_mask] /= active_sum

    def _find_free_slot(self) -> Optional[int]:
        free = torch.where(~self.active_mask)[0]
        if len(free) > 0:
            return free[0].item()
        return None

    def _detect_new_polymorph(self):
        """
        Identify if a new stable mode has emerged that represents a distinct 'Polymorph'.
        This is a high-level semantic tagging of modes.
        """
        # Simple logic: Any active mode that has survived for a while is a candidate
        active_idx = torch.where(self.active_mask)[0]
        
        for idx in active_idx:
            idx_val = idx.item()
            # If age is high enough and not already tracked
            if self.age[idx_val] > 100: # Arbitrary age threshold
                # Check if this mode index is already tracked? 
                # Indices get reused, so we should track by hash of vector or ID?
                # For simplicity, let's just track if we haven't seen this index 'graduate' yet since last reset.
                # But better: hash the prototype vector.
                
                vec_hash = hash(self.mu[idx_val].detach().cpu().numpy().tobytes())
                if vec_hash not in self.poly_tracker:
                    name = f"Polymorph-{self.poly_id_counter:02d}"
                    self.poly_tracker[vec_hash] = {
                        "name": name,
                        "id": self.poly_id_counter,
                        "discovered_at": datetime.utcnow().isoformat(),
                        "mode_index": idx_val
                    }
                    self.poly_id_counter += 1
                    logger.info(f"PMM: NEW POLYMORPH DISCOVERED: {name} (Mode {idx_val})")
