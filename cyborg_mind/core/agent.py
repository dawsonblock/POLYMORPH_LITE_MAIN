import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from .model import CyborgModel

class Agent:
    """
    Agent wrapper for interaction with environments.
    Handles device placement, sampling, and state management.
    """
    def __init__(self, model: CyborgModel, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
    def get_action(
        self, 
        obs: torch.Tensor, 
        rnn_state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[int, float, torch.Tensor, float, Dict]:
        """
        Select action for a single observation (inference/rollout).
        
        Args:
            obs: (ObsDim,) or (1, ObsDim)
            rnn_state: (Layers, 1, Hidden)
            
        Returns:
            action: Selected action index
            log_prob: Log probability of action
            next_rnn_state: Updated RNN state
            value: Estimated value
            info: Memory info
        """
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0) # (1, D)
            
            obs = obs.to(self.device)
            if rnn_state is not None:
                rnn_state = rnn_state.to(self.device)
                
            logits, value, next_rnn_state, _, mem_info = self.model(obs, rnn_state)
            
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
            log_prob = torch.log(probs[0, action.item()] + 1e-8)
            
            return action.item(), log_prob.item(), next_rnn_state, value.item(), mem_info

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
