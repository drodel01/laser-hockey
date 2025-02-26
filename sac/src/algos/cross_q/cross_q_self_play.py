"""
CrossQ Implementation in Python

This module implements CrossQ for reinforcement learning.

References:
- Bhatt, A., Palenicek, D., Belousov, B., Argus, M., Amiranashvili, A., Brox, T., & Peters, J. (2024).
  "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity."
  arXiv preprint: https://arxiv.org/abs/1902.05605.

Author: Niclas Lietzow
"""
import math
import random
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch

from src.algos.core.algorithm import SquashedGaussianMLPActor
from src.algos.cross_q.cross_q import CrossQ
from src.environment.hockey_env import BasicOpponent, OpponentBase

WEAK_OPPONENT = BasicOpponent(weak=True)
STRONG_OPPONENT = BasicOpponent(weak=False)


class PolicyOpponent(OpponentBase):
    def __init__(self, policy: SquashedGaussianMLPActor):
        self.policy = policy

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.policy.device,
        )
        with torch.no_grad():
            a, _ = self.policy(obs, deterministic=False, with_logprob=False)
            return a.cpu().numpy()


class CrossQForSelfPlay(CrossQ):
    def __init__(
        self,
        env: gym.Env,
        opponent_pool_size: int = 10,
        weak_opponent_sample_prob: float = 0.05,
        strong_opponent_sample_prob: float = 0.5,
        opponent_update_frequency: int = 200_000,
        **kwargs,
    ):
        super().__init__(env=env, **kwargs)
        self.pool: deque[PolicyOpponent] = deque(maxlen=opponent_pool_size)
        self.weak_opponent_sample_prob = weak_opponent_sample_prob
        self.strong_opponent_sample_prob = strong_opponent_sample_prob
        self.opponent_update_frequency = opponent_update_frequency

    def sample_opponent(self) -> OpponentBase:
        p = random.random()
        if p < self.weak_opponent_sample_prob:
            return WEAK_OPPONENT
        if p < self.weak_opponent_sample_prob + self.strong_opponent_sample_prob:
            return STRONG_OPPONENT
        if len(self.pool) == 0:
            return STRONG_OPPONENT
        return random.choice(self.pool)

    def learn_via_self_play(
        self,
        total_steps,
        test_env,
        wandb_run,
        **kwargs,
    ):
        num_iterations = math.ceil(total_steps / self.opponent_update_frequency)
        for iteration in range(num_iterations):
            continue_from_step = iteration * self.opponent_update_frequency
            self.learn(
                total_steps=self.opponent_update_frequency,
                test_env=test_env,
                continue_from_step=continue_from_step,
                wandb_run=wandb_run,
                **kwargs,
            )

            # We sample the new opponent first, that way we get a delay so that the model
            # never play against an exact same copy
            opponent = self.sample_opponent()
            self.env.unwrapped.set_opponent(opponent=opponent)

            policy = SquashedGaussianMLPActor(
                obs_dim=self.env.observation_space.shape[0],
                act_dim=self.env.action_space.shape[0],
                act_limit=self.env.action_space.high[0],
                hidden_sizes=self.ac.pi.hidden_sizes,
            )
            state_dict = deepcopy(policy.state_dict())
            policy.load_state_dict(state_dict)
            policy.to(self.ac.pi.device)
            policy.eval()
            self.pool.append(PolicyOpponent(policy=policy))
