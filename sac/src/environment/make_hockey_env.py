from typing import Optional

import gymnasium as gym
from gymnasium.envs.registration import register
from src.environment.hockey_env import OpponentBase

ENV_ID = "Hockey-v0"


def make_hockey_env_gym(opponent: Optional[OpponentBase] = None):
    if ENV_ID not in gym.envs.registry.keys():
        register(
            id=ENV_ID,
            entry_point="src.environment.hockey_env:HockeyEnv",
        )
    return gym.make(ENV_ID, opponent=opponent)
