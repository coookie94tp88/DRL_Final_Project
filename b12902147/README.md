# DooRL Environment v2.0

A pure Python Deep Reinforcement Learning environment built from scratch for the DooRL specification. This project simulates a 10-player, 4-door betting game with dynamic payouts, managed by a central Host.

## Project Structure

- `config/default_config.yaml`: Contains all hyperparameters for the environment, training, and checkpointing.
- `env.py`: The core environment implementation (`DooRLEnv`) adhering to the 3-phase execution model and dynamic payout engine.
- `agents.py`: Contains basic PyTorch neural network modules for both `HostAgent` and `PlayerAgent`.
- `train.py`: The main training pipeline that ties everything together. It supports TensorBoard logging, checkpointing, and resuming training from a saved state.

## Setup

1. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Training Pipeline

To start a new training session:
```bash
python train.py --config config/default_config.yaml
```

To resume training from a specific checkpoint:
```bash
python train.py --config config/default_config.yaml --resume checkpoints/checkpoint_ep_1000.pt
```

## Monitoring Training

The training script automatically logs metrics to the `logs/` directory. You can visualize them using TensorBoard:
```bash
tensorboard --logdir=logs/
```

Metrics tracked:
- `Episode/Host_Total_Profit`
- `Episode/Active_Players`
- `Episode/Bankrupt_Players`
