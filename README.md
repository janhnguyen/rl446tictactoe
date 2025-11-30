# TicTacToe Reinforcement Learning Project

This project implements six reinforcement learning agents to play TicTacToe against the user. The algorithms range from entry-level baselines to more advanced variants, all sharing a lightweight custom environment.

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

1. Install dependencies :

```bash
pip install -r requirements.txt
```

2. Train agents. Checkpoints are saved to `checkpoints/<algo>.pt` by default:

```bash
python train.py --episodes 800
python train.py --algo dqn --episodes 800
python train.py --algo rainbow --episodes 800
python train.py --algo sac --episodes 800
```

Set `--seed` for reproducible runs.

3. Play against a trained policy (agent moves first as **X**, you play **O**). You'll be prompted to pick one interactively:

```bash
python play.py --algo dqn --checkpoint checkpoints/dqn.pt
python play.py                                 # prompt to choose an algorithm
```
