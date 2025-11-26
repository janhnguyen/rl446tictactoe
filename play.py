from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Type

import numpy as np
import torch

from tictactoe_rl.agents import (
    A2CAgent,
    DQNAgent,
    DoubleDQNAgent,
    PPOAgent,
    RainbowAgent,
    SACAgent,
)
from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.utils import to_tensor, set_seed

ALGORITHMS: Dict[str, Type] = {
    "dqn": DQNAgent,
    "ddqn": DoubleDQNAgent,
    "a2c": A2CAgent,
    "rainbow": RainbowAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
}


def masked_argmax(values: np.ndarray, available):
    masked = np.full_like(values, -np.inf, dtype=float)
    for a in available:
        masked[a] = values[a]
    return int(np.nanargmax(masked))


def greedy_action(agent, state: np.ndarray, device: torch.device) -> int:
    if hasattr(agent, "policy_net"):
        with torch.no_grad():
            q_values = agent.policy_net(to_tensor(state, device).unsqueeze(0)).cpu().numpy().flatten()
        return masked_argmax(q_values, agent.env.available_actions())
    if hasattr(agent, "model"):
        with torch.no_grad():
            logits, _ = agent.model(to_tensor(state, device).unsqueeze(0))
        return masked_argmax(logits.cpu().numpy().flatten(), agent.env.available_actions())
    if hasattr(agent, "policy"):
        with torch.no_grad():
            probs, _ = agent.policy(to_tensor(state, device).unsqueeze(0))
        probs_np = probs.cpu().numpy().flatten()
        available = agent.env.available_actions()
        if not available:
            return int(np.argmax(probs_np))
        masked = np.zeros_like(probs_np)
        masked[available] = probs_np[available]
        return int(available[int(np.argmax(masked[available]))])
    raise ValueError("Unsupported agent type for greedy inference")


def prompt_human_move(env: TicTacToeEnv) -> int:
    valid = env.available_actions()
    while True:
        try:
            move = int(input(f"Your move (0-8) valid {valid}: "))
        except ValueError:
            print("Please enter a number between 0 and 8.")
            continue
        if move in valid:
            return move
        print("Invalid move. Try again.")


def play_round(agent, device: torch.device) -> None:
    env = agent.env
    state = env.reset()
    done = False
    print("New game! Agent plays as X (positions 0-8 left-to-right, top-to-bottom).")
    print(env.render())

    while not done:
        action = greedy_action(agent, state, device)
        agent_result = env.step(action, auto_opponent=False)
        state = agent_result.state
        print(f"\nAgent plays {action} (X):")
        print(env.render())
        if agent_result.done:
            outcome = "Agent wins!" if agent_result.reward > 0 else "Draw!"
            print(outcome)
            break

        human_action = prompt_human_move(env)
        opp_result = env.opponent_step(human_action)
        state = opp_result.state
        print(f"\nYou play {human_action} (O):")
        print(env.render())
        done = opp_result.done
        if done:
            if opp_result.reward < 0:
                print("You win!")
            elif opp_result.reward > 0:
                print("Agent wins!")
            else:
                print("Draw!")


def main():
    parser = argparse.ArgumentParser(description="Play against a trained TicTacToe agent")
    parser.add_argument("--algo", choices=list(ALGORITHMS.keys()), default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the trained checkpoint (defaults to checkpoints/<algo>.pt)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def prompt_algorithm_choice() -> str:
        options = list(ALGORITHMS.keys())
        print("Select an algorithm to play against:")
        for idx, name in enumerate(options, start=1):
            print(f"  {idx}. {name}")
        while True:
            choice = input("Enter number or name: ").strip().lower()
            if choice in ALGORITHMS:
                return choice
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(options):
                    return options[idx - 1]
            print("Invalid selection. Please choose a listed algorithm.")

    algo = args.algo or prompt_algorithm_choice()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToeEnv()
    agent_cls = ALGORITHMS[algo]
    agent = agent_cls(env, device=device)

    ckpt_path = Path(args.checkpoint or Path("checkpoints") / f"{algo}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py to create it.")
    agent.load(str(ckpt_path))
    print(f"Loaded {algo} checkpoint from {ckpt_path}")

    while True:
        play_round(agent, device)
        again = input("Play again? (y/n): ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
