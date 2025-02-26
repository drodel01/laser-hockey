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
import torch.nn as nn


def mlp(
    sizes: tuple[int, ...],
    activation: type[nn.Module],
    output_activation: type[nn.Module],
) -> nn.Sequential:
    def build():
        for j in range(len(sizes) - 2):
            yield nn.Linear(sizes[j], sizes[j + 1])
            yield activation()

        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())


def mlp_bn(
    sizes: tuple[int, ...],
    batch_norm_eps: float,
    batch_norm_momentum: float,
    activation: type[nn.Module],
    output_activation: type[nn.Module],
) -> nn.Sequential:
    def build():
        for j in range(len(sizes) - 2):
            yield nn.Linear(sizes[j], sizes[j + 1])
            yield nn.BatchNorm1d(
                sizes[j + 1],
                eps=batch_norm_eps,
                momentum=batch_norm_momentum,
            )
            yield activation()

        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())
