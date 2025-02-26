import numpy as np
from gymnasium import spaces

from src.environment.core import BasicOpponent, HockeyEnvCore


class HockeyEnv(HockeyEnvCore):
    def __init__(self, weak: bool = False):
        super().__init__()
        self.opponent = BasicOpponent(weak=weak, keep_mode=self.keep_mode)
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

    def step(self, action):
        obs2 = self.obs_agent_two()
        action2 = self.opponent.act(obs2)
        actions_combined = np.hstack([action, action2])
        return super().step(actions_combined)
    
class HockeyEnvRandom(HockeyEnvCore):
    """
    HockeyEnv that samples opponent difficulty (weak/strong) at each episode.
    
    - ``hard_prob`` probability that the opponent is strong, otherwise weak.
    """
    def __init__(self, hard_prob = 0.5):
        self.hard_prob = hard_prob
        self.opponent_strong = BasicOpponent(weak=False, keep_mode=True)
        self.opponent_weak = BasicOpponent(weak=True, keep_mode=True)
        self.opponent = self.opponent_strong
        super().__init__()
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

    def step(self, action):
        obs2 = self.obs_agent_two()
        action2 = self.opponent.act(obs2)
        actions_combined = np.hstack([action, action2])
        return super().step(actions_combined)
    
    def reset(self, *args, **kwargs):
        self.opponent = self.opponent_strong if np.random.rand() < self.hard_prob else self.opponent_weak
        return super().reset(*args, **kwargs)
