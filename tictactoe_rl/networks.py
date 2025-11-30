from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


def mlp(input_dim: int, output_dim: int, hidden: Tuple[int, ...] = (128, 128)) -> nn.Module:
    layers = []
    last_dim = input_dim
    for h in hidden:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.feature = mlp(input_dim, hidden[-1], hidden[:-1] if len(hidden) > 1 else ())
        self.advantage = mlp(hidden[-1], output_dim, ())
        self.value = mlp(hidden[-1], 1, ())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        adv = self.advantage(features)
        val = self.value(features)
        return val + adv - adv.mean(1, keepdim=True)


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class NoisyDuelingDQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.feature = mlp(input_dim, hidden[0], ())
        self.advantage = nn.Sequential(
            NoisyLinear(hidden[0], hidden[1]),
            nn.ReLU(),
            NoisyLinear(hidden[1], output_dim),
        )
        self.value = nn.Sequential(
            NoisyLinear(hidden[0], hidden[1]),
            nn.ReLU(),
            NoisyLinear(hidden[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        adv = self.advantage(features)
        val = self.value(features)
        return val + adv - adv.mean(1, keepdim=True)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
