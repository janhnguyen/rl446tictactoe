from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict


def check_winner(board: np.ndarray) -> Optional[int]:
    lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]
    for indices in lines:
        values = board[indices]
        if abs(values.sum()) == 3:
            return int(np.sign(values[0]))
    if not (board == 0).any():
        return 0
    return None


class TicTacToeEnv:
    """Simple TicTacToe environment with random opponent.

    The agent always plays as ``1`` (X) and moves first. The opponent plays as
    ``-1`` (O) and selects a random valid move. Rewards:
    * ``+1`` for an agent win
    * ``-1`` for a loss or illegal action
    * ``0.5`` for a draw
    * ``0`` otherwise
    """

    def __init__(self, illegal_move_penalty: float = -1.0):
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1
        self.illegal_move_penalty = illegal_move_penalty

    @property
    def action_space(self) -> int:
        return 9

    @property
    def observation_space(self) -> int:
        return 9

    def reset(self) -> np.ndarray:
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1
        return self.board.copy()

    def available_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(
        self, action: int, *, opponent_action: Optional[int] = None, auto_opponent: bool = True
    ) -> StepResult:
        if self.board[action] != 0:
            return StepResult(self.board.copy(), self.illegal_move_penalty, True, {"illegal": True})

        self.board[action] = 1
        result = check_winner(self.board)
        if result is not None:
            reward = 1.0 if result == 1 else 0.5
            done = True
            return StepResult(self.board.copy(), reward, done, {})

        if not auto_opponent:
            return StepResult(self.board.copy(), 0.0, False, {"awaiting_opponent": True})

        if opponent_action is None:
            opponent_action = random.choice(self.available_actions())

        if self.board[opponent_action] != 0:
            return StepResult(self.board.copy(), self.illegal_move_penalty, True, {"illegal_opponent": True})

        self.board[opponent_action] = -1
        result = check_winner(self.board)
        if result is not None:
            if result == 0:
                reward = 0.5
            elif result == -1:
                reward = -1.0
            else:
                reward = 1.0
            return StepResult(self.board.copy(), reward, True, {})

        return StepResult(self.board.copy(), 0.0, False, {})

    def opponent_step(self, action: int) -> StepResult:
        if self.board[action] != 0:
            return StepResult(self.board.copy(), self.illegal_move_penalty, True, {"illegal_opponent": True})

        self.board[action] = -1
        result = check_winner(self.board)
        if result is not None:
            if result == 0:
                reward = 0.5
            elif result == -1:
                reward = -1.0
            else:
                reward = 1.0
            return StepResult(self.board.copy(), reward, True, {})

        return StepResult(self.board.copy(), 0.0, False, {})

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: " "}
        rows = []
        for r in range(3):
            row = " | ".join(symbols[self.board[c]] for c in range(r * 3, (r + 1) * 3))
            rows.append(row)
        return "\n------\n".join(rows)


__all__ = ["TicTacToeEnv", "StepResult", "check_winner"]
