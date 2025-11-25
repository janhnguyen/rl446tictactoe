from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from tictactoe_rl.agents.dqn import DQNAgent, DQNConfig
from tictactoe_rl.utils import to_tensor


class DoubleDQNAgent(DQNAgent):
    def optimize(self):
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
            next_actions = self.policy_net(next_states_v).argmax(1)
            next_q_values = self.target_net(next_states_v).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected = rewards_v + self.config.gamma * next_q_values * (1 - dones_v)

        loss = F.mse_loss(state_action_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
