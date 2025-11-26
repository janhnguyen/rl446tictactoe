from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.networks import mlp
from tictactoe_rl.utils import to_tensor


@dataclass
class A2CConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    entropy_beta: float = 0.01
    value_coef: float = 0.5
    rollout_length: int = 10
    hidden: tuple = (128, 128)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: tuple):
        super().__init__()
        self.feature = mlp(input_dim, hidden[-1], hidden[:-1] if len(hidden) > 1 else ())
        self.policy = nn.Linear(hidden[-1], action_dim)
        self.value = nn.Linear(hidden[-1], 1)

    def forward(self, x):
        features = self.feature(x)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value


class A2CAgent:
    def __init__(self, env: TicTacToeEnv, config: Optional[A2CConfig] = None, device: Optional[torch.device] = None):
        self.env = env
        self.config = config or A2CConfig()
        self.device = device or torch.device("cpu")
        self.model = ActorCritic(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def select_action(self, state: np.ndarray):
        state_v = to_tensor(state, self.device).unsqueeze(0)
        logits, _ = self.model(state_v)
        mask = torch.full_like(logits, float("-inf"))
        mask[0, self.env.available_actions()] = 0
        logits = logits + mask
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action, probs[:, action]

    def train(self, episodes: int = 500):
        returns = []
        for ep in range(episodes):
            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            dones = []
            entropies = []
            total_reward = 0.0
            done = False
            while not done:
                action, prob = self.select_action(state)
                logits, value = self.model(to_tensor(state, self.device).unsqueeze(0))
                mask = torch.full_like(logits, float("-inf"))
                mask[0, [i for i, v in enumerate(state) if v == 0]] = 0
                logits = logits + mask
                log_prob = torch.log_softmax(logits, dim=-1)[0, action]
                policy_probs = torch.softmax(logits, dim=-1)
                entropies.append(-(policy_probs * torch.log(policy_probs + 1e-8)).sum())
                result = self.env.step(action)
                log_probs.append(log_prob)
                values.append(value.squeeze(0))
                rewards.append(result.reward)
                dones.append(result.done)
                state = result.state
                done = result.done
                total_reward += result.reward

            Qval = torch.tensor([0.0], device=self.device)
            returns_list = []
            for r, d in zip(reversed(rewards), reversed(dones)):
                Qval = torch.tensor([r], device=self.device) + self.config.gamma * Qval * (1 - int(d))
                returns_list.insert(0, Qval)
            returns_t = torch.cat(returns_list)
            log_probs_t = torch.stack(log_probs)
            values_t = torch.cat(values)

            advantage = returns_t - values_t
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            actor_loss = -(log_probs_t * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            entropy = torch.stack(entropies).mean()
            loss = actor_loss + self.config.value_coef * critic_loss - self.config.entropy_beta * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            returns.append(total_reward)
        return returns
