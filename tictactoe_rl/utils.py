from __future__ import annotations

import math
import random
from typing import Iterable, Tuple

import numpy as np
import torch


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def epsilon_by_frame(frame_idx: int, eps_start: float, eps_final: float, eps_decay: int) -> float:
    return eps_final + (eps_start - eps_final) * math.exp(-1.0 * frame_idx / eps_decay)


def action_mask(states: np.ndarray, action_space: int, device: torch.device) -> torch.Tensor:
    """Return a mask tensor with ``0`` for valid moves and ``-inf`` for invalid ones."""

    states_np = np.array(states)
    if states_np.ndim == 1:
        states_np = states_np.reshape(1, -1)
    mask = np.full((states_np.shape[0], action_space), -np.inf, dtype=np.float32)
    for idx, state in enumerate(states_np):
        available = np.where(state == 0)[0]
        mask[idx, available] = 0.0
    return torch.as_tensor(mask, device=device)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def sample_multistep(replay, batch_size: int, n_step: int, gamma: float):
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    n_rewards = []
    n_next_states = []
    n_dones = []
    for idx in range(batch_size):
        reward = 0.0
        discount = 1.0
        end_state = next_states[idx]
        done_flag = dones[idx]
        for step in range(1, n_step):
            if idx + step >= len(replay.buffer):
                break
            state_s, action_s, reward_s, next_state_s, done_s = replay.buffer[(replay.position - step) % len(replay.buffer)]
            if done_flag:
                break
            reward += discount * reward_s
            discount *= gamma
            end_state = next_state_s
            done_flag = done_flag or done_s
        n_rewards.append(reward + rewards[idx])
        n_next_states.append(end_state)
        n_dones.append(done_flag)
    return states, actions, np.array(n_rewards, dtype=np.float32), np.array(n_next_states), np.array(n_dones)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
