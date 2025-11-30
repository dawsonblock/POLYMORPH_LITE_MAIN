import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .memory import StaticPseudoModeMemory

class HybridRecurrentEncoder(nn.Module):
    """
    Hybrid Encoder using GRU (or Mamba) backbone.
    Currently defaults to GRU for maximum stability and compatibility.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature extractor (MLP)
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Recurrent Backbone
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, Seq, InputDim) or (Batch, InputDim)
            h: (Layers, Batch, HiddenDim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add seq dim
            
        features = self.feature_net(x)
        out, h_next = self.rnn(features, h)
        
        # Return last time step features
        return out[:, -1, :], h_next

class CyborgModel(nn.Module):
    """
    The CyborgMind Model:
    Observation -> Encoder -> Latent -> PMM -> Augmented State -> Heads
    """
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_dim: int = 128,
        memory_modes: int = 32
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 1. Encoder (Observation -> Latent)
        self.encoder = HybridRecurrentEncoder(obs_dim, hidden_dim)
        
        # 2. Memory (Latent -> Memory Augmented)
        self.memory = StaticPseudoModeMemory(
            latent_dim=hidden_dim,
            max_modes=memory_modes
        )
        
        # 3. Heads (Augmented State -> Policy/Value)
        # Input to heads is Latent + Memory Reconstruction (concatenated)
        combined_dim = hidden_dim * 2
        
        self.actor_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self, 
        obs: torch.Tensor, 
        rnn_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass for PPO training/inference.
        
        Returns:
            logits: Action logits
            value: State value
            rnn_state: Next RNN state
            memory_recon: Memory reconstruction (for auxiliary loss)
            info: Debug info
        """
        # 1. Encode
        latent, next_rnn_state = self.encoder(obs, rnn_state)
        
        # 2. Memory Interaction
        # We use the latent state to query memory
        memory_recon, mem_info = self.memory(latent)
        
        # 3. Fusion
        # Concatenate latent state and memory reconstruction
        # This gives the agent access to "what it is seeing" AND "what it remembers/predicts"
        combined = torch.cat([latent, memory_recon], dim=1)
        
        # 4. Heads
        logits = self.actor_head(combined)
        value = self.critic_head(combined)
        
        return logits, value, next_rnn_state, memory_recon, mem_info

    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.encoder.num_layers, batch_size, self.hidden_dim, device=device)
