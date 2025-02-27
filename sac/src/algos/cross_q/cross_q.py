"""
CrossQ Implementation in Python

This module implements CrossQ for reinforcement learning.

References:
- Bhatt, A., Palenicek, D., Belousov, B., Argus, M., Amiranashvili, A., Brox, T., & Peters, J. (2024).
  "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity."
  arXiv preprint: https://arxiv.org/abs/1902.05605.

Author: Niclas Lietzow
"""
import gymnasium as gym
import torch

from src.algos.core import AlgorithmBase, Batch
from src.algos.cross_q.core import CrossQActorCritic


class CrossQ(AlgorithmBase):
    actor_critic_class = CrossQActorCritic

    def __init__(
        self,
        env: gym.Env,
        replay_size: int = int(1e6),
        init_alpha: float = 1,
        alpha_trainable: bool = True,
        actor_hidden_sizes: tuple[int, ...] = (256, 256),
        critic_hidden_sizes: tuple[int, ...] = (2048, 2048),
        batch_size: int = 256,
        gamma: float = 0.99,
        betas: tuple[float, float] = (0.5, 0.999),
        lr: float = 1e-3,
        policy_delay: int = 3,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.99,
        device: str = "auto",
    ):
        super().__init__(
            env=env,
            replay_size=replay_size,
            init_alpha=init_alpha,
            alpha_trainable=alpha_trainable,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            batch_size=batch_size,
            gamma=gamma,
            betas=betas,
            lr=lr,
            policy_delay=policy_delay,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            device=device,
        )

    def compute_loss_q(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            a2, log_p_a2 = self.ac.pi(batch.obs2)

        q1_current, q1_next = self.ac.q1.forward_joint(
            batch.obs, batch.act, batch.obs2, a2
        )
        q2_current, q2_next = self.ac.q2.forward_joint(
            batch.obs, batch.act, batch.obs2, a2
        )

        with torch.no_grad():
            q_pi = torch.min(q1_next, q2_next)
            backup = batch.reward + self.gamma * (1 - batch.done) * (
                q_pi - self.ac.log_alpha.exp() * log_p_a2
            )

        loss_q1 = ((q1_current - backup) ** 2).mean()
        loss_q2 = ((q2_current - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q
