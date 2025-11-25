# TicTacToe Reinforcement Learning Zoo

This project implements six reinforcement learning agents to play TicTacToe against a random opponent. The algorithms range from entry-level baselines to more advanced variants, all sharing a lightweight custom environment.

## Environment
- `tictactoe_rl/env.py`: 3x3 board where the agent (X) moves first and the opponent (O) plays a random valid move.
- Rewards: `+1` win, `-1` loss or illegal move, `0.5` draw, `0` otherwise.

## Algorithms
- **DQN** (`tictactoe_rl/agents/dqn.py`)
- **Double DQN** (`tictactoe_rl/agents/ddqn.py`)
- **A2C** (`tictactoe_rl/agents/a2c.py`)
- **Rainbow DQN** (`tictactoe_rl/agents/rainbow.py`)
- **PPO** (`tictactoe_rl/agents/ppo.py`)
- **SAC (discrete)** (`tictactoe_rl/agents/sac.py`)

Each agent exposes a `train(episodes)` method and selects actions that respect the current valid moves.

## Quickstart

1. Install dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

2. Train any agent:

```bash
python train.py dqn --episodes 200
python train.py rainbow --episodes 400
python train.py sac --episodes 400
```

GPU acceleration is used when available. Set `--seed` for reproducible runs.
