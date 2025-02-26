import sys
import time
from pathlib import Path
from typing import Optional
import os
import yaml

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.algos.td3.actor_critic import MLPActorCritic
from src.algos.common.replay_buffer import Batch, ReplayBuffer
from src.algos.common.utils import polyak_update
from copy import deepcopy

class TD3:
    def __init__(
        self,
        env: gym.Env,
		tau: float = 0.005,
        gamma: float = 0.99,
		policy_noise: float = 0.2,
		noise_clip: float = 0.5,
		policy_delay: int = 2,
        buffer_size: int = int(1e6),
        actor_hidden_sizes: tuple[int, ...] = (256, 256),
        critic_hidden_sizes: tuple[int, ...] = (256, 256),
        device: str = "auto",
    ):
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        if not isinstance(self.env.observation_space, spaces.Box):
            raise TypeError(
                f"Expected Box observation space, got {type(self.env.observation_space)}"
            )

        if not isinstance(self.env.action_space, spaces.Box):
            raise TypeError(
                f"Expected Box action space, got {type(self.env.action_space)}"
            )

        observation_space: spaces.Box = self.env.observation_space
        action_space: spaces.Box = self.env.action_space

        self.obs_dim = observation_space.shape
        self.act_dim = action_space.shape[0]

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Device: {self.device}")

        self.ac = MLPActorCritic(
            observation_space=observation_space,
            action_space=action_space,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
        ).to(self.device)

        self.ac_targ = deepcopy(self.ac).to(self.device)
        self.critic_params = tuple(self.ac.q1.parameters()) + tuple(self.ac.q2.parameters())

        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            max_size=buffer_size,
            device=self.device,
        )

        # Init optimizers to None
        self._critic_optimizer = None
        self._actor_optimizer = None

        # Losses
        self._loss_policy = 0.0
        self._loss_critic = 0.0

        # Additional metrics
        self._q1_mean = 0.0
        self._q1_std = 0.0
        self._q2_mean = 0.0
        self._q2_std = 0.0
        self._target_q_mean = 0.0
        self._target_q_std = 0.0

        # Logger
        self._logger = None

        @property
        def critic_optimizer(self) -> Adam:
            if self._critic_optimizer is None:
                raise ValueError("Optimizer not initialized")
            return self._critic_optimizer
        
        @property
        def actor_optimizer(self) -> Adam:
            if self._actor_optimizer is None:
                raise ValueError("Optimizer not initialized")
            return self._actor_optimizer


    def compute_loss_critic(self, batch: Batch) -> torch.Tensor:
        current_q1 = self.ac.q1(batch.obs, batch.act)
        current_q2 = self.ac.q2(batch.obs, batch.act)

        with torch.no_grad():
            policy_targ = self.ac_targ.policy(batch.obs2)

            # Target policy smoothing
            epsilon = torch.randn_like(policy_targ) * self.policy_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = torch.clamp(policy_targ + epsilon, -1, 1)

            # Target Q-values
            q1_target = self.ac_targ.q1(batch.obs2, a2)
            q2_target = self.ac_targ.q2(batch.obs2, a2)
            next_q_values = torch.min(q1_target, q2_target)
            target_q_values = batch.reward + self.gamma * (1 - batch.done) * next_q_values

            self._q1_mean = current_q1.mean().item()
            self._q1_std = current_q1.std().item()
            self._q2_mean = current_q2.mean().item()
            self._q2_std = current_q2.std().item()
            self._target_q_mean = next_q_values.mean().item()
            self._target_q_std = next_q_values.std().item()

        loss_q1 = ((current_q1 - target_q_values) ** 2).mean()
        loss_q2 = ((current_q2 - target_q_values) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q
    
    def compute_loss_policy(self, batch: Batch) -> torch.Tensor:
        q1_policy = self.ac.q1(batch.obs, self.ac.policy(batch.obs))
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
    

    def test_agent(self, test_env: gym.Env, num_test_episodes: int) -> None:
        returns, lengths, success = (
            np.zeros(num_test_episodes, dtype=np.float32),
            np.zeros(num_test_episodes, dtype=np.int32),
            np.zeros(num_test_episodes, dtype=np.bool),
        )
        for ep in range(num_test_episodes):
            obs, info = test_env.reset()
            ep_ret, ep_len = 0, 0
            done, truncated = False, False
            while not (done or truncated):
                action = self.ac.act(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
                obs, reward, done, truncated, info = test_env.step(action)
                ep_ret += reward
                ep_len += 1

            returns[ep] = ep_ret
            lengths[ep] = ep_len
            success[ep] = info.get("is_success", False)

        self._test_ep_ret = float(returns.mean())
        self._test_ep_len = float(lengths.mean())
        self._test_success = float(success.mean())

    def save_model(self, model_path: Path, save_buffer: bool = False) -> None:
        if model_path.is_dir():
            raise ValueError(f"Path {model_path} is a directory")

        model_path = model_path.with_suffix(".pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.ac.state_dict(), model_path)

        if save_buffer:
            buffer_path = model_path.with_suffix(".buffer.pth")
            self.replay_buffer.save(buffer_path)

    @classmethod
    def load_model(
        cls,
        env: gym.Env,
        model_path: Path,
        buffer_path: Optional[Path] = None,
        device: str = "cpu",
        **kwargs,
    ) -> "TD3":
        model_obj = cls(env, **kwargs)
        model_obj.ac.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        if buffer_path is not None:
            model_obj.replay_buffer.load(buffer_path)
        return model_obj


    def learn(
        self,
        test_env: gym.Env,
        total_timesteps: int = 100_000,
        num_test_episodes: int = 10,
        learning_starts: int = 1_000,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        save_freq: int = 10_000,
        eval_freq: int = 1_000,
        betas: tuple[float, float] = (0.5, 0.999),
        seed: Optional[int] = None,
        log_dir: str = "./logs/td3",
    ) -> None:
        self._logger = SummaryWriter(log_dir)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._critic_optimizer = Adam(self.critic_params, lr=learning_rate, betas=betas)
        self._actor_optimizer = Adam(self.ac.policy.parameters(), lr=learning_rate, betas=betas)

        # Main interaction loop
        start_time = time.time()
        obs, _ = self.env.reset()
        ep_ret, ep_len = 0, 0

        for t in range(total_timesteps):
            #self._logger.set_step(t)

            if t >= learning_starts:
                action = self.ac.act(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
            else:
                action = self.env.action_space.sample()

            obs2, reward, done, truncated, info = self.env.step(action)
            ep_ret += reward
            ep_len += 1

            # Store if episode ended. However, if done only because of truncation,
            # we don't store the done transition because it may continue.
            self.replay_buffer.store(
                obs=obs,
                act=action,
                rew=reward,
                obs2=obs2,
                done= done and not truncated,
            )
            obs = obs2

            if done or truncated:
                self._logger.add_scalar("Ep/EpRet", ep_ret, t)
                self._logger.add_scalar("Ep/EpLen", ep_len, t)
                obs, info = self.env.reset()
                ep_ret, ep_len = 0, 0

            if self.replay_buffer.size >= batch_size:
                batch = self.replay_buffer.sample_batch(batch_size)
                update_policy = (t + 1) % self.policy_delay == 0
                self.update(
                    batch,
                    update_policy,
                )
            elif self.replay_buffer.max_size < batch_size:
                raise ValueError("Batch size must be less than replay buffer size")

            if (t + 1) % save_freq == 0:
                self.save_model(Path(f"{log_dir}/checkpoints/td3_{t+1}"))
            
            if (t + 1) % eval_freq == 0:
                self.test_agent(test_env, num_test_episodes)
                self._logger.add_scalar("Test/TestEpRet", self._test_ep_ret, t)
                self._logger.add_scalar("Test/TestEpLen", self._test_ep_len, t)
                self._logger.add_scalar("Test/TestSuccess", self._test_success, t)
                self._logger.add_scalar("Loss/LossPolicy", self._loss_policy, t)
                self._logger.add_scalar("Loss/LossCritic", self._loss_critic, t)
                self._logger.add_scalar("TotalEnvInteracts", t, t)
                # Log additional metrics
                self._logger.add_scalar("Metrics/Q1Mean", self._q1_mean, t)
                self._logger.add_scalar("Metrics/Q1Std", self._q1_std, t)
                self._logger.add_scalar("Metrics/Q2Mean", self._q2_mean, t)
                self._logger.add_scalar("Metrics/Q2Std", self._q2_std, t)
                self._logger.add_scalar("Metrics/TargetQMean", self._target_q_mean, t)
                self._logger.add_scalar("Metrics/TargetQStd", self._target_q_std, t)
        self._logger.close()


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

    model = TD3(env=env, **model_params)
    try:
        model.learn(test_env=test_env, **learn_params)
    except (KeyboardInterrupt, Exception) as e:
        raise e

