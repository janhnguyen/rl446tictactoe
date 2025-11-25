from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tictactoe_rl.env import TicTacToeEnv
from tictactoe_rl.networks import mlp
from tictactoe_rl.utils import to_tensor


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    batch_size: int = 64
    rollout_size: int = 512
    hidden: tuple = (128, 128)


class PPOActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: tuple):
        super().__init__()
        self.feature = mlp(input_dim, hidden[-1], hidden[:-1] if len(hidden) > 1 else ())
        self.policy = nn.Linear(hidden[-1], action_dim)
        self.value = nn.Linear(hidden[-1], 1)

    def forward(self, x):
        feat = self.feature(x)
        return self.policy(feat), self.value(feat)


class PPOAgent:
    def __init__(self, env: TicTacToeEnv, config: Optional[PPOConfig] = None, device: Optional[torch.device] = None):
        self.env = env
        self.config = config or PPOConfig()
        self.device = device or torch.device("cpu")
        self.model = PPOActorCritic(env.observation_space, env.action_space, self.config.hidden).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

    def select_action(self, state: np.ndarray):
        state_v = to_tensor(state, self.device).unsqueeze(0)
        logits, value = self.model(state_v)
        mask = torch.full_like(logits, float("-inf"))
        mask[0, self.env.available_actions()] = 0
        logits = logits + mask
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)
        log_prob = torch.log_softmax(logits, dim=-1)[0, action]
        return int(action.item()), log_prob.squeeze(), value.squeeze()

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        next_value = 0.0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + self.config.gamma * next_value * (1 - d) - v
            gae = delta + self.config.gamma * self.config.lam * (1 - d) * gae
            advantages.insert(0, gae)
            next_value = v
        returns = [a + v for a, v in zip(advantages, values)]
        return torch.tensor(advantages, dtype=torch.float32, device=self.device), torch.tensor(returns, dtype=torch.float32, device=self.device)

    def train(self, episodes: int = 500):
        total_rewards = []
        states: List[np.ndarray] = []
        actions: List[int] = []
        log_probs: List[torch.Tensor] = []
        values: List[float] = []
        rewards: List[float] = []
        dones: List[int] = []

        episode = 0
        state = self.env.reset()
        while episode < episodes:
            action, log_prob, value = self.select_action(state)
            result = self.env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.item())
            rewards.append(result.reward)
            dones.append(int(result.done))
            state = result.state

            if result.done:
                total_rewards.append(sum(rewards[-(len(dones)) :]))
                state = self.env.reset()
                episode += 1

            if len(states) >= self.config.rollout_size:
                advs, returns = self.compute_advantages(rewards, values, dones)
                states_t = to_tensor(np.array(states), self.device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
                old_log_probs = torch.stack(log_probs).detach()

                for _ in range(self.config.epochs):
                    idx = np.random.permutation(len(states))
                    for start in range(0, len(states), self.config.batch_size):
                        batch_idx = idx[start : start + self.config.batch_size]
                        logits, value_pred = self.model(states_t[batch_idx])
                        mask = torch.full_like(logits, float("-inf"))
                        for i, b in enumerate(batch_idx):
                            avail = [a for a in range(self.env.action_space) if states[b][a] == 0]
                            mask[i, avail] = 0
                        logits = logits + mask
                        new_log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions_t[batch_idx].unsqueeze(1)).squeeze(1)
                        ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                        surr1 = ratio * advs[batch_idx]
                        surr2 = torch.clamp(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip) * advs[batch_idx]
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = (returns[batch_idx] - value_pred.squeeze()).pow(2).mean()
                        loss = actor_loss + 0.5 * critic_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                states.clear()
                actions.clear()
                log_probs.clear()
                values.clear()
                rewards.clear()
                dones.clear()
        return total_rewards
