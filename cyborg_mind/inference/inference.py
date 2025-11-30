import torch
import logging
import argparse
from cyborg_mind.core.model import CyborgModel
from cyborg_mind.core.agent import Agent
from cyborg_mind.env.gym_adapter import GymAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference(checkpoint_path: str, env_id: str = "CartPole-v1"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Env
    env = GymAdapter(env_id, device=device)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Setup Agent
    model = CyborgModel(obs_dim, action_dim, hidden_dim=64, memory_modes=16)
    agent = Agent(model, device=device)
    
    # Load
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    agent.load(checkpoint_path)
    model.eval() # Set to eval mode
    
    # Run Loop
    obs = env.reset()
    rnn_state = model.get_initial_state(1, device)
    
    total_reward = 0
    done = False
    
    while not done:
        # Deterministic action for inference
        action, _, next_rnn_state, _, info = agent.get_action(obs, rnn_state, deterministic=True)
        
        obs, reward, done, truncated, _ = env.step(action)
        rnn_state = next_rnn_state
        total_reward += reward
        
        # Optional: Visualize memory
        # print(f"Active Modes: {info['alpha_active'].shape}")
        
        if done or truncated:
            break
            
    logger.info(f"Inference Episode Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    args = parser.parse_args()
    
    run_inference(args.ckpt, args.env)
