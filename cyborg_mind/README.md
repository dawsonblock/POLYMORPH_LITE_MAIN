# CyborgMind: Production RL + Memory System

This package contains the core intelligence for the Polymorph system, featuring a hybrid recurrent neural network with Static Pseudo-Mode Memory (PMM).

## Structure

- **core/**: Core components.
  - `memory.py`: `StaticPseudoModeMemory` implementation with explicit read/write cycles and safety mechanisms.
  - `model.py`: `CyborgModel` combining encoder and memory.
  - `agent.py`: Agent wrapper for interaction.
  - `preprocessing.py`: Raman spectroscopy preprocessing pipeline.
- **env/**: Environment adapters.
  - `gym_adapter.py`: Standard Gym wrapper.
  - `minerl_adapter.py`: MineRL wrapper.
- **training/**: Training pipeline.
  - `trainer.py`: PPO trainer with GAE and Prometheus metrics.
  - `rollout.py`: Trajectory buffer.
- **inference/**: Inference scripts.
- **scripts/**: Training scripts (CartPole, MineRL).

## Usage

### Training
```bash
python scripts/train_gym_cartpole.py
```

### Inference
```bash
python inference/inference.py --ckpt checkpoints/cartpole_agent.pt
```

### BentoML Integration
The `bentoml_service` uses `cyborg_mind.core` directly. Ensure this package is in the python path or installed.
