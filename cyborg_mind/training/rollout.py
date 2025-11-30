import torch
import numpy as np
from typing import List, Dict

class RolloutBuffer:
    """
    Stores trajectories for PPO updates.
    """
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.rnn_states = []
        
    def add(self, obs, action, reward, done, log_prob, value, rnn_state):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rnn_states.append(rnn_state)
        
    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.rnn_states = []
        
    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation.
        """
        advantages = []
        returns = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        return torch.tensor(returns), torch.tensor(advantages)
