import torch
import logging
import os
import numpy as np
from cyborg_mind.core.model import CyborgModel
from cyborg_mind.core.agent import Agent
from cyborg_mind.env.minerl_adapter import MineRLAdapter
from cyborg_mind.training.rollout import RolloutBuffer
from cyborg_mind.training.trainer import PPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_minerl():
    # Hyperparams
    env_id = "MineRLNavigateDense-v0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_timesteps = 1000
    steps_per_epoch = 200
    
    try:
        env = MineRLAdapter(env_id, device=device)
    except ImportError:
        logger.error("MineRL not installed. Skipping.")
        return
        
    # Obs: (C, H, W) -> 3 * 64 * 64 = 12288
    obs_dim = 3 * 64 * 64
    
    # MineRL Action Space Handling
    # For navigation, we can simplify to a discrete set of actions:
    # 0: forward
    # 1: jump
    # 2: left
    # 3: right
    # 4: attack
    # 5: camera left
    # 6: camera right
    # ...
    # However, MineRLAdapter should ideally handle this mapping.
    # For this example, we assume the environment wrapper provides a discrete action space or we map it here.
    # Let's assume 10 discrete actions.
    action_dim = 10 
    
    # Model with larger hidden dim for visual inputs
    model = CyborgModel(obs_dim, action_dim, hidden_dim=256, memory_modes=32)
    agent = Agent(model, device=device)
    
    # Trainer
    trainer = PPOTrainer(agent, lr=1e-4, batch_size=32, start_metrics_server=True)
    buffer = RolloutBuffer()
    
    logger.info("Starting MineRL training loop...")
    
    obs = env.reset()
    rnn_state = model.get_initial_state(1, device)
    
    episode_reward = 0
    episode_len = 0
    
    for step in range(total_timesteps):
        # Select Action
        action_idx, log_prob, next_rnn_state, value, _ = agent.get_action(obs, rnn_state)
        
        # Map discrete action index to MineRL dictionary action
        # This is a simplified mapping for demonstration
        minerl_action = env.env.action_space.no_op()
        if action_idx == 0: minerl_action['forward'] = 1
        elif action_idx == 1: minerl_action['jump'] = 1
        elif action_idx == 2: minerl_action['camera'] = [0, -10]
        elif action_idx == 3: minerl_action['camera'] = [0, 10]
        elif action_idx == 4: minerl_action['attack'] = 1
        # ... add more mappings as needed
        
        # Step Env
        # Note: MineRLAdapter.step expects the raw action or handles mapping?
        # Our MineRLAdapter.step calls env.step(action).
        # So we pass the dict.
        
        next_obs, reward, done, info = env.step(minerl_action)
        
        # Store
        buffer.add(obs, action_idx, reward, done, log_prob, value, rnn_state)
        
        obs = next_obs
        rnn_state = next_rnn_state
        episode_reward += reward
        episode_len += 1
        
        if done:
            logger.info("Episode finished: Reward={}, Len={}".format(episode_reward, episode_len))
            obs = env.reset()
            rnn_state = model.get_initial_state(1, device)
            episode_reward = 0
            episode_len = 0
            
        # Update
        if (step + 1) % steps_per_epoch == 0:
            logger.info("Update at step {}".format(step+1))
            # Get next value
            _, _, _, next_val, _ = agent.get_action(obs, rnn_state)
            metrics = trainer.update(buffer, next_val)
            logger.info("Metrics: {}".format(metrics))
            buffer.clear()
            
            # Save Checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            agent.save("checkpoints/minerl_agent.pt")
            
    env.close()

if __name__ == "__main__":
    train_minerl()
