import os
import sys
import yaml

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.algos.td3.td3 import TD3
from src.algos.td3.distributional_critic import DistributionalCritic, scalar_to_categorical_target
from src.algos.common.replay_buffer import Batch
from src.algos.common.utils import polyak_update


class TD3CategoricalQ(TD3):
    def __init__(
        self,
        env: gym.Env,
        tau: float = 0.005,
        gamma: float = 0.99,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = int(1e6),
        num_bins: int = 51,
        vmin=-10,
        vmax=10,
        sigma_ratio: float = 0.75,
        actor_hidden_sizes: tuple[int, ...] = (256, 256),
        critic_hidden_sizes: tuple[int, ...] = (256, 256),
        device="auto",
    ):
        super().__init__(env, tau, gamma, policy_noise, noise_clip, policy_delay, buffer_size, actor_hidden_sizes, critic_hidden_sizes, device)

        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.sigma = (vmax - vmin) / num_bins * sigma_ratio

        # Replace the standard Q-functions with DistributionalCritics
        self.ac.q1 = DistributionalCritic(self.obs_dim[0], self.act_dim, num_bins, vmin, vmax, critic_hidden_sizes).to(self.device)
        self.ac.q2 = DistributionalCritic(self.obs_dim[0], self.act_dim, num_bins, vmin, vmax, critic_hidden_sizes).to(self.device)
        self.ac_targ.q1 = deepcopy(self.ac.q1).to(self.device)
        self.ac_targ.q2 = deepcopy(self.ac.q2).to(self.device)

        self.critic_params = tuple(self.ac.q1.parameters()) + tuple(self.ac.q2.parameters())

        # Optimizers
        self._critic_optimizer = Adam(self.critic_params, lr=1e-3)
        self._actor_optimizer = Adam(self.ac.policy.parameters(), lr=1e-3)

    def compute_loss_critic(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            policy_targ = self.ac_targ.policy(batch.obs2)

            # Target policy smoothing
            epsilon = torch.randn_like(policy_targ) * self.policy_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = torch.clamp(policy_targ + epsilon, -1, 1)

            # Target Q-values (logits)
            logits_q1_target = self.ac_targ.q1(batch.obs2, a2)
            logits_q2_target = self.ac_targ.q2(batch.obs2, a2)

            # Compute expected Q-values from logits
            q1_target = self.ac_targ.q1.get_q_value(logits_q1_target)
            q2_target = self.ac_targ.q2.get_q_value(logits_q2_target)
            next_q_values = torch.min(q1_target, q2_target)

            # Compute categorical target distribution
            target_q_values = batch.reward.unsqueeze(-1) + self.gamma * (1 - batch.done.unsqueeze(-1)) * next_q_values
            target_probs = scalar_to_categorical_target(target_q_values, self.ac.q1, self.sigma)

        # Compute logits for current Q-values
        logits_q1 = self.ac.q1(batch.obs, batch.act)
        logits_q2 = self.ac.q2(batch.obs, batch.act)

        # Compute loss using KL divergence
        loss_q1 = F.kl_div(F.log_softmax(logits_q1, dim=-1), target_probs, reduction="batchmean")
        loss_q2 = F.kl_div(F.log_softmax(logits_q2, dim=-1), target_probs, reduction="batchmean")

        return loss_q1 + loss_q2

    def compute_loss_policy(self, batch: Batch) -> torch.Tensor:
        # Get logits for current policy actions
        logits_q1 = self.ac.q1(batch.obs, self.ac.policy(batch.obs))

        # Compute expected Q-value
        q1_policy = self.ac.q1.get_q_value(logits_q1)
        return -q1_policy.mean()

    def update(self, batch: Batch, update_policy: bool):
        # Optimize critics
        self._critic_optimizer.zero_grad()
        loss_critic = self.compute_loss_critic(batch)
        loss_critic.backward()
        self._critic_optimizer.step()
        self._loss_critic = loss_critic.item()

        # Delayed policy updates
        if update_policy:
            for p in self.critic_params:
                p.requires_grad = False

            # Optimize actor
            self._actor_optimizer.zero_grad()
            loss_policy = self.compute_loss_policy(batch)
            loss_policy.backward()
            self._actor_optimizer.step()

            for p in self.critic_params:
                p.requires_grad = True

            self._loss_policy = loss_policy.item()

            polyak_update(self.ac.q1.parameters(), self.ac_targ.q1.parameters(), self.tau)
            polyak_update(self.ac.q2.parameters(), self.ac_targ.q2.parameters(), self.tau)
            polyak_update(self.ac.policy.parameters(), self.ac_targ.policy.parameters(), self.tau)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hockey-v0")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file with hyperparameters")
    args = parser.parse_args()

    if args.env == "Hockey-v0":
        gym.register(
            id="Hockey-v0",
            entry_point="src.environment.environment:HockeyEnvRandom",
        )

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    model_params = config.get("model", {})
    learn_params = config.get("learn", {})

    env = gym.make(args.env)
    test_env = gym.make(args.env)

    model = TD3CategoricalQ(env=env, **model_params)
    try:
        model.learn(test_env=test_env, **learn_params)
    except (KeyboardInterrupt, Exception) as e:
        raise e