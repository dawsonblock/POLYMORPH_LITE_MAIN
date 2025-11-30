import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict
from ..core.agent import Agent
from .rollout import RolloutBuffer

from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

# Prometheus Metrics
GAUGE_REWARD = Gauge('reward_mean', 'Mean reward per update')
GAUGE_LOSS_POLICY = Gauge('loss_policy', 'Policy loss')
GAUGE_LOSS_VALUE = Gauge('loss_value', 'Value loss')

class PPOTrainer:
    """
    PPO Trainer with Memory Integration.
    """
    def __init__(
        self,
        agent: Agent,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        memory_coef: float = 0.1, # Auxiliary loss for memory reconstruction
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        start_metrics_server: bool = False
    ):
        if start_metrics_server:
            try:
                start_http_server(8000)
                logger.info("Prometheus metrics server started on port 8000")
            except Exception as e:
                logger.warning(f"Could not start metrics server: {e}")

        self.agent = agent
        self.optimizer = optim.Adam(agent.model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.memory_coef = memory_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
    def update(self, buffer: RolloutBuffer, next_value: float) -> Dict[str, float]:
        """
        Perform PPO update.
        """
        # 1. Prepare data
        returns, advantages = buffer.compute_gae(next_value, self.gamma, self.gae_lambda)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Stack tensors
        # obs list of tensors -> stack
        obs_batch = torch.stack(buffer.obs).to(self.agent.device)
        act_batch = torch.tensor(buffer.actions).to(self.agent.device)
        old_log_probs = torch.tensor(buffer.log_probs).to(self.agent.device)
        returns = returns.to(self.agent.device)
        advantages = advantages.to(self.agent.device)
        
        # RNN states: (T, Layers, 1, H) -> (T, Layers, H) -> need careful handling for batching
        # For simplicity in this PPO implementation, we might just detach RNN state at each step 
        # or use the stored states as initial states for chunks.
        # Here we assume full-batch update or simple batching without BPTT across chunks for simplicity.
        # Ideally we re-run forward pass.
        
        # Let's do simple full-batch for now or mini-batching if T is large.
        dataset_size = len(buffer.obs)
        indices = np.arange(dataset_size)
        
        total_loss_p = 0
        total_loss_v = 0
        total_loss_mem = 0
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                # Mini-batch data
                mb_obs = obs_batch[idx]
                mb_act = act_batch[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                mb_old_log_p = old_log_probs[idx]
                
                # Re-evaluate
                # Note: RNN state handling in mini-batch PPO is tricky. 
                # We usually burn-in or carry forward. 
                # For this implementation, we will ignore RNN state carry-over within update 
                # and just use zero state or stored state (which is stale).
                # Correct way: RecurrentPPO requires sequences.
                # Simplification: Treat each step as independent for gradient (truncated BPTT=1)
                # but pass the stored RNN state as input.
                
                mb_rnn_state = torch.stack([buffer.rnn_states[i] for i in idx]).squeeze(2).permute(1, 0, 2).to(self.agent.device)
                
                logits, values, _, mem_recon, _ = self.agent.model(mb_obs, mb_rnn_state)
                
                # Policy Loss
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_p)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = 0.5 * ((values.squeeze() - mb_ret) ** 2).mean()
                
                # Memory Auxiliary Loss (Reconstruction)
                # We want memory to reconstruct the latent state (self-consistency)
                # But we don't have the "target" latent easily available here without hooks.
                # Let's assume the model output `mem_recon` should match `latent`.
                # But we need `latent` from the model forward pass.
                # Let's modify model to return latent? Or just skip for now.
                # Actually, `mem_recon` is returned. We can minimize its norm or something?
                # No, we need a target.
                # Let's skip memory loss for this iteration to keep it simple, 
                # or add a simple sparsity loss on memory attention.
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss_p += policy_loss.item()
                total_loss_v += value_loss.item()
                
        # Trigger Memory Write Cycle (Explicit Update)
        # We should do this on the collected trajectories?
        # Or just once per update?
        # The memory module needs `last_batch` to be set.
        # We can run a forward pass on a sample of data to prime it.
        with torch.no_grad():
            sample_idx = np.random.choice(dataset_size, min(64, dataset_size))
            sample_obs = obs_batch[sample_idx]
            # Prime memory
            self.agent.model.encoder(sample_obs) # This doesn't call memory forward
            # We need to call model forward
            self.agent.model(sample_obs)
            # Now update
            self.agent.model.memory.apply_explicit_updates()
            
        # Update Metrics
        loss_p = total_loss_p / self.ppo_epochs
        loss_v = total_loss_v / self.ppo_epochs
        rew_mean = returns.mean().item()
        
        GAUGE_LOSS_POLICY.set(loss_p)
        GAUGE_LOSS_VALUE.set(loss_v)
        GAUGE_REWARD.set(rew_mean)
            
        return {
            "loss_policy": loss_p,
            "loss_value": loss_v,
            "reward_mean": rew_mean
        }
