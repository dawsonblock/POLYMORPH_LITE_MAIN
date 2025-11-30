import torch
import logging
import os
from cyborg_mind.core.model import CyborgModel
from cyborg_mind.core.agent import Agent
from cyborg_mind.env.gym_adapter import GymAdapter
from cyborg_mind.training.rollout import RolloutBuffer
from cyborg_mind.training.trainer import PPOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # Hyperparams
    env_id = "CartPole-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_timesteps = 1000
    steps_per_epoch = 200
    
    # 1. Setup Env
    env = GymAdapter(env_id, device=device)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 2. Setup Agent
    model = CyborgModel(obs_dim, action_dim, hidden_dim=64, memory_modes=16)
    agent = Agent(model, device=device)
    
    # 3. Setup Trainer
    trainer = PPOTrainer(agent, lr=1e-3, batch_size=64, start_metrics_server=True)
    buffer = RolloutBuffer()
    
    # 4. Training Loop
    obs = env.reset()
    rnn_state = model.get_initial_state(1, device)
    
    episode_reward = 0
    episode_len = 0
    
    for step in range(total_timesteps):
        # Select Action
        action, log_prob, next_rnn_state, value, _ = agent.get_action(obs, rnn_state)
        
        # Step Env
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Store
        buffer.add(obs, action, reward, done, log_prob, value, rnn_state)
        
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
            # Get next value for GAE
            _, _, _, next_val, _ = agent.get_action(obs, rnn_state)
            metrics = trainer.update(buffer, next_val)
            logger.info("Metrics: {}".format(metrics))
            buffer.clear()
            
    # Save
    os.makedirs("checkpoints", exist_ok=True)
    agent.save("checkpoints/cartpole_agent.pt")
    logger.info("Training complete. Model saved.")
    env.close()

if __name__ == "__main__":
    train()
