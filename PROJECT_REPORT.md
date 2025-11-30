# TicTacToe RL Project Report

## Overview
This project implements a compact reinforcement-learning sandbox for TicTacToe featuring six distinct agents (DQN, Double DQN, A2C, Rainbow DQN, PPO, and discrete SAC) built on a shared environment and CLI tooling. The goal is to provide runnable baselines plus stronger variants while keeping the code easy to extend and experiment with.

## Environment
- **Game mechanics:** `TicTacToeEnv` models a 3x3 board with the agent playing **X** first and an automatic random opponent playing **O** second. Rewards are +1 for a win, -1 for a loss or illegal move, 0.5 for a draw, and 0 otherwise, with helper utilities to render boards and detect wins. 【F:tictactoe_rl/env.py†L18-L129】
- **Action masking:** Agents rely on `available_actions` and related masking helpers to ensure policy/value estimates only consider legal moves, reducing instability from impossible actions. 【F:tictactoe_rl/env.py†L67-L104】

## Agents
All agents follow a simple interface: initialize with the environment/device, call `train(episodes)` to produce per-episode rewards, `select_action` for policy decisions, and `save`/`load` for checkpoint persistence.
- **Value-based baselines:** DQN and Double DQN use Q-networks with experience replay; Rainbow extends them with prioritized replay and distributional/dueling components. 【F:tictactoe_rl/agents/dqn.py†L1-L173】【F:tictactoe_rl/agents/ddqn.py†L1-L155】【F:tictactoe_rl/agents/rainbow.py†L1-L169】
- **On-policy actor-critic:** A2C couples policy and value heads with entropy regularization and normalized advantages to encourage exploration. 【F:tictactoe_rl/agents/a2c.py†L1-L205】
- **Policy-gradient improvements:** PPO adds clipped surrogate objectives and GAE-style advantage computation for more stable updates. 【F:tictactoe_rl/agents/ppo.py†L1-L236】
- **Off-policy entropy-regularized:** Discrete SAC trains twin critics and a stochastic policy with automatic temperature tuning to balance exploration and exploitation. 【F:tictactoe_rl/agents/sac.py†L1-L233】

## Training Workflow
- **CLI entry point:** `train.py` trains one or all algorithms with configurable episodes, seed, and checkpoint directory. By default it cycles through every registered agent, printing average rewards and saving learned policies. 【F:train.py†L18-L66】
- **Shared utilities:** Seeding, tensor helpers, and replay buffers provide consistent behavior across agents, minimizing duplicate code. 【F:tictactoe_rl/utils.py†L1-L90】【F:tictactoe_rl/replay_buffer.py†L1-L127】
- **Checkpoints:** Models are saved as `<algo>.pt` files under the chosen directory, enabling later evaluation or gameplay. 【F:train.py†L60-L65】

## Evaluation & Gameplay
- **Interactive play:** `play.py` loads a chosen checkpoint (or prompts for an algorithm if `--algo` is omitted) and runs greedy inference against a human opponent. Boards render after each move, and the session can be repeated without reloading. 【F:play.py†L21-L152】
- **Human turns:** The script enforces legal human moves and reports outcomes (win/draw/lose) after each game. 【F:play.py†L60-L104】

## Usage Summary
1. Install dependencies with `pip install -r requirements.txt`. 【F:README.md†L21-L25】
2. Train all or specific agents (default 800 episodes) and emit checkpoints. 【F:train.py†L28-L65】【F:README.md†L27-L45】
3. Play against a trained policy via `play.py`, using the interactive algorithm prompt when `--algo` is omitted. 【F:play.py†L106-L152】【F:README.md†L38-L45】

## Notes on Performance
Average rewards depend on episodes, seeds, and the inherent randomness of the opponent. Agents are designed to respect legal moves and leverage masking to stabilize learning, but outcomes will vary between runs given the short horizon and small action space. 【F:tictactoe_rl/env.py†L67-L104】【F:train.py†L50-L65】
