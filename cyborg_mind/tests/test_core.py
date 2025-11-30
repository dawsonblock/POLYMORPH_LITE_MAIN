import torch
import pytest
from cyborg_mind.core.memory import StaticPseudoModeMemory
from cyborg_mind.core.model import CyborgModel
from cyborg_mind.core.agent import Agent

def test_memory_forward():
    mem = StaticPseudoModeMemory(latent_dim=32, max_modes=4)
    x = torch.randn(2, 32)
    recon, info = mem(x)
    assert recon.shape == x.shape
    assert "alpha" in info
    
def test_memory_update():
    mem = StaticPseudoModeMemory(latent_dim=32, max_modes=4)
    x = torch.randn(2, 32)
    mem(x) # Prime
    mem.apply_explicit_updates()
    # Check occupancy updated
    assert mem.occupancy.sum() > 0

def test_model_forward():
    model = CyborgModel(obs_dim=10, action_dim=2, hidden_dim=32)
    obs = torch.randn(1, 10)
    logits, val, next_rnn, mem_recon, info = model(obs)
    assert logits.shape == (1, 2)
    assert val.shape == (1, 1)
    assert next_rnn.shape == (2, 1, 32) # 2 layers

def test_agent_action():
    model = CyborgModel(obs_dim=10, action_dim=2, hidden_dim=32)
    agent = Agent(model)
    obs = torch.randn(10)
    action, log_prob, _, _, _ = agent.get_action(obs, None)
    assert isinstance(action, int)
    assert isinstance(log_prob, float)
