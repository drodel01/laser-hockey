from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from gymnasium import spaces

from src.environment.core import BasicOpponent as BasicOpponentCore, HockeyEnvCore


class OpponentBase(ABC):
    id: str = ""

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        pass


class BasicOpponent(BasicOpponentCore, OpponentBase):
    id = "BasicOpponent"

    def __init__(self, weak: bool = False):
        super().__init__(weak=weak)
        self.id += f"_{weak=}"


class HockeyEnv(HockeyEnvCore):
    def __init__(self, opponent: Optional[OpponentBase]):
        super().__init__()
        self._opponent = opponent or BasicOpponent()
        # linear force in (x,y)-direction, torque, and shooting
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

    def step(self, action):
        ob2 = self.obs_agent_two()
        a2 = self._opponent.act(ob2)
        action2 = np.hstack([action, a2])
        return super().step(action2)

    def set_opponent(self, opponent: OpponentBase):
        self._opponent = opponent

    def opponent_id(self):
        return str(self._opponent.id)
