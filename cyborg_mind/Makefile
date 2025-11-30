.PHONY: install test train-cartpole train-minerl inference build clean

install:
	pip install -r requirements.txt

test:
	export PYTHONPATH=$PYTHONPATH:. && pytest tests/

train-cartpole:
	export PYTHONPATH=$PYTHONPATH:. && python scripts/train_gym_cartpole.py

train-minerl:
	export PYTHONPATH=$PYTHONPATH:. && python scripts/train_minerl_navigate.py

inference:
	export PYTHONPATH=$PYTHONPATH:. && python inference/inference.py --ckpt checkpoints/cartpole_agent.pt

build:
	docker build -t cyborg-mind .

clean:
	rm -rf __pycache__
	rm -rf checkpoints/
