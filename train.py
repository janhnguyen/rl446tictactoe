from __future__ import annotations

import argparse
import torch

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.agents.dqn import DQNAgent
from tictactoe_rl.agents.ddqn import DoubleDQNAgent
from tictactoe_rl.agents.a2c import A2CAgent
from tictactoe_rl.agents.rainbow import RainbowAgent
from tictactoe_rl.agents.ppo import PPOAgent
from tictactoe_rl.agents.sac import SACAgent
from tictactoe_rl.utils import set_seed


ALGORITHMS = {
    "dqn": DQNAgent,
    "ddqn": DoubleDQNAgent,
    "a2c": A2CAgent,
    "rainbow": RainbowAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
}


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on TicTacToe")
    parser.add_argument(
        "--algo",
        choices=list(ALGORITHMS.keys()) + ["all"],
        default="all",
        help="which algorithm to train (default: all)",
    )
    parser.add_argument("--episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_algos = list(ALGORITHMS.keys()) if args.algo == "all" else [args.algo]
    for algo in selected_algos:
        env = TicTacToeEnv()
        agent_cls = ALGORITHMS[algo]
        agent = agent_cls(env, device=device)
        rewards = agent.train(episodes=args.episodes)
        avg_reward = sum(rewards) / len(rewards)
        print(
            f"Trained {algo} for {args.episodes} episodes. Avg reward: {avg_reward:.3f}"
        )


if __name__ == "__main__":
    main()
