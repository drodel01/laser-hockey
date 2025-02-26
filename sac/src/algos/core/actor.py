"""
CrossQ and Soft Actor-Critic (SAC) Implementation in Python

This module implements CrossQ and Soft Actor-Critic (SAC) for reinforcement learning.
The SAC implementation is partially based on OpenAI Spinning Up's SAC implementation
(https://spinningup.openai.com/), with modifications and enhancements.

References:
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
  "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor."
  arXiv preprint: https://arxiv.org/abs/1801.01290.
- Bhatt, A., Palenicek, D., Belousov, B., Argus, M., Amiranashvili, A., Brox, T., & Peters, J. (2024).
  "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity."
  arXiv preprint: https://arxiv.org/abs/1902.05605.
- OpenAI Spinning Up: https://spinningup.openai.com/

Author: Niclas Lietzow
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus

from src.algos.core.utils import mlp

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        act_limit: float,
    ):
        super().__init__()
        self.net = mlp(
            sizes=(obs_dim, *hidden_sizes),
            activation=nn.ReLU,
            output_activation=nn.ReLU,
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return tuple(
            layer.out_features for layer in self.net if isinstance(layer, nn.Linear)
        )

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True
    ):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            log_p_pi = pi_distribution.log_prob(pi_action).sum(dim=-1) - (
                2 * (np.log(2) - pi_action - softplus(-2 * pi_action))
            ).sum(dim=1)
        else:
            log_p_pi = None

        pi_action = self.act_limit * torch.tanh(pi_action)
        return pi_action, log_p_pi
