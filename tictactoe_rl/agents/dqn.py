from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.networks import mlp
from tictactoe_rl.replay_buffer import ReplayBuffer
from tictactoe_rl.utils import epsilon_by_frame, to_tensor


torch.autograd.set_detect_anomaly(False)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    batch_size: int = 64
    lr: float = 1e-3
    replay_size: int = 5000
    min_buffer_size: int = 500
    eps_start: float = 1.0
    eps_final: float = 0.05
    eps_decay: int = 5000
    target_update: int = 200
    hidden: tuple = (128, 128)


class DQNAgent:
    def __init__(self, env: TicTacToeEnv, config: Optional[DQNConfig] = None, device: Optional[torch.device] = None):
        self.env = env
        self.config = config or DQNConfig()
        self.device = device or torch.device("cpu")

        self.policy_net = mlp(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.target_net = mlp(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.replay = ReplayBuffer(self.config.replay_size)
        self.frame_idx = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        epsilon = epsilon_by_frame(self.frame_idx, self.config.eps_start, self.config.eps_final, self.config.eps_decay)
        self.frame_idx += 1
        if random.random() < epsilon and not evaluate:
            return random.choice(self.env.available_actions())
        state_v = to_tensor(state, self.device).unsqueeze(0)
        q_values = self.policy_net(state_v)
        q_values = q_values.cpu().detach().numpy().flatten()
        masked_q = np.full_like(q_values, -np.inf)
        for a in self.env.available_actions():
            masked_q[a] = q_values[a]
        return int(masked_q.argmax())

    def optimize(self) -> Optional[float]:
        if len(self.replay) < self.config.min_buffer_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)
        states_v = to_tensor(states, self.device)
        next_states_v = to_tensor(next_states, self.device)
        actions_v = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_v = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_v = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_v)
        state_action_values = q_values.gather(1, actions_v.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_v).max(1)[0]
            expected = rewards_v + self.config.gamma * next_q_values * (1 - dones_v)

        loss = F.mse_loss(state_action_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def train(self, episodes: int = 1000):
        history = []
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
                if self.frame_idx % self.config.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            history.append(total_reward)
        return history
