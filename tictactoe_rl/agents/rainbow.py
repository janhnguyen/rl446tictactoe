from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.networks import NoisyDuelingDQN
from tictactoe_rl.replay_buffer import PrioritizedReplayBuffer
from tictactoe_rl.utils import action_mask, hard_update, to_tensor


@dataclass
class RainbowConfig:
    gamma: float = 0.99
    batch_size: int = 64
    lr: float = 1e-3
    replay_size: int = 5000
    min_buffer_size: int = 100
    target_update: int = 100
    n_step: int = 3
    hidden: tuple = (128, 128)


class RainbowAgent:
    def __init__(self, env: TicTacToeEnv, config: Optional[RainbowConfig] = None, device: Optional[torch.device] = None):
        self.env = env
        self.config = config or RainbowConfig()
        self.device = device or torch.device("cpu")

        self.policy_net = NoisyDuelingDQN(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.target_net = NoisyDuelingDQN(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        hard_update(self.target_net, self.policy_net)
        self.replay = PrioritizedReplayBuffer(self.config.replay_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.frame_idx = 0

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def select_action(self, state: np.ndarray) -> int:
        state_v = to_tensor(state, self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_v).cpu().numpy().flatten()
        masked_q = np.full_like(q_values, -np.inf)
        for a in self.env.available_actions():
            masked_q[a] = q_values[a]
        return int(masked_q.argmax())

    def optimize(self) -> Optional[float]:
        if len(self.replay) < self.config.min_buffer_size:
            return None
        states, actions, rewards, next_states, dones, indices, weights = self.replay.sample(self.config.batch_size)
        states_v = to_tensor(states, self.device)
        next_states_v = to_tensor(next_states, self.device)
        actions_v = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_v = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_v = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_v = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_v)
        state_action_values = q_values.gather(1, actions_v.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            mask = action_mask(next_states, self.env.action_space, self.device)
            policy_q = self.policy_net(next_states_v) + mask
            next_actions = policy_q.argmax(1)
            target_q = self.target_net(next_states_v) + mask
            next_q = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_v + self.config.gamma * next_q * (1 - dones_v)

        td_errors = state_action_values - targets
        loss = (td_errors.pow(2) * weights_v).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.replay.update_priorities(indices, priorities)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        return float(loss.item())

    def train(self, episodes: int = 800):
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.select_action(state)
                result = self.env.step(action)
                self.replay.push(state, action, result.reward, result.state, result.done)
                state = result.state
                done = result.done
                total_reward += result.reward
                loss = self.optimize()
                self.frame_idx += 1
                if self.frame_idx % self.config.target_update == 0:
                    hard_update(self.target_net, self.policy_net)
            rewards.append(total_reward)
        return rewards
