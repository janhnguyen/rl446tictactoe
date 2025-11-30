from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.networks import mlp
from tictactoe_rl.replay_buffer import ReplayBuffer
from tictactoe_rl.utils import soft_update, to_tensor


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.01
    lr: float = 3e-4
    alpha: float = 0.2
    batch_size: int = 64
    replay_size: int = 5000
    min_buffer_size: int = 100
    hidden: tuple = (128, 128)


class DiscretePolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: tuple):
        super().__init__()
        self.net = mlp(input_dim, action_dim, hidden)

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample(self, x):
        probs, _ = self.forward(x)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: tuple):
        super().__init__()
        self.net = mlp(input_dim, action_dim, hidden)

    def forward(self, x):
        return self.net(x)


class SACAgent:
    def __init__(self, env: TicTacToeEnv, config: Optional[SACConfig] = None, device: Optional[torch.device] = None):
        self.env = env
        self.config = config or SACConfig()
        self.device = device or torch.device("cpu")

        self.policy = DiscretePolicy(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.q1 = QNetwork(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.q2 = QNetwork(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.target_q1 = QNetwork(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.target_q2 = QNetwork(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        soft_update(self.target_q1, self.q1, 1.0)
        soft_update(self.target_q2, self.q2, 1.0)

        self.replay = ReplayBuffer(self.config.replay_size)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=self.config.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=self.config.lr)

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        with torch.no_grad():
            state_v = to_tensor(state, self.device).unsqueeze(0)
            probs, _ = self.policy(state_v)
            probs_np = probs.squeeze(0).cpu().numpy()
        available = self.env.available_actions()
        masked_probs = np.zeros_like(probs_np)
        if available:
            masked_probs[available] = probs_np[available]
            if masked_probs.sum() == 0:
                action = np.random.choice(available)
            else:
                masked_probs = masked_probs / masked_probs.sum()
                action = np.random.choice(len(probs_np), p=masked_probs)
        else:
            action = int(np.argmax(probs_np))
        if evaluate:
            action = int(np.argmax(probs_np))
        return int(action)

    def optimize(self):
        if len(self.replay) < self.config.min_buffer_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)
        states_v = to_tensor(states, self.device)
        next_states_v = to_tensor(next_states, self.device)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            availability = torch.tensor((next_states == 0), device=self.device, dtype=torch.float32)
            next_probs, _ = self.policy(next_states_v)
            masked_next_probs = next_probs * availability
            masked_next_probs = masked_next_probs / masked_next_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
            masked_next_log_probs = torch.log(masked_next_probs.clamp(min=1e-8))

            next_q1 = self.target_q1(next_states_v).masked_fill(~availability.bool(), -1e9)
            next_q2 = self.target_q2(next_states_v).masked_fill(~availability.bool(), -1e9)
            min_next_q = torch.minimum(next_q1, next_q2)
            next_value = (masked_next_probs * (min_next_q - self.config.alpha * masked_next_log_probs)).sum(dim=1)
            target_q = rewards_v + self.config.gamma * (1 - dones_v) * next_value

        q1_pred = self.q1(states_v).gather(1, actions_v.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(states_v).gather(1, actions_v.unsqueeze(1)).squeeze(1)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        availability = torch.tensor((states == 0), device=self.device, dtype=torch.float32)
        probs, _ = self.policy(states_v)
        masked_probs = probs * availability
        masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        masked_log_probs = torch.log(masked_probs.clamp(min=1e-8))

        q1_values = self.q1(states_v).masked_fill(~availability.bool(), -1e9)
        q2_values = self.q2(states_v).masked_fill(~availability.bool(), -1e9)
        min_q = torch.minimum(q1_values, q2_values)
        policy_loss = (masked_probs * (self.config.alpha * masked_log_probs - min_q)).sum(dim=1).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        soft_update(self.target_q1, self.q1, self.config.tau)
        soft_update(self.target_q2, self.q2, self.config.tau)
        return float((q1_loss + q2_loss + policy_loss).item())

    def train(self, episodes: int = 800):
        total_rewards = []
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
                self.optimize()
            total_rewards.append(total_reward)
        return total_rewards
