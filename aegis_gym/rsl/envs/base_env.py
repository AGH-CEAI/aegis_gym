from abc import abstractmethod
from typing import NamedTuple
import torch as th
from tensordict import TensorDict
from rsl_rl.env import VecEnv

from .scene import BaseScene


class ResetReturn(NamedTuple):
    observations: TensorDict
    extras: dict


class StepReturn(NamedTuple):
    observations: TensorDict
    rewards: th.Tensor
    dones: th.Tensor
    extras: dict


class BaseEnv(VecEnv):
    """
    Base class for implementing an environment compatible with rsl_rl's VecEnv.
    See https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/env/vec_env.py
    """

    def __init__(self, scene: BaseScene):
        super().__init__()
        self._scene: BaseScene = scene

    def __del__(self):
        if self._scene:
            self._scene.shutdown()

    @property
    def unwrapped(self) -> "BaseEnv":
        """Required by rsl_rl logger."""
        return self

    @property
    def step_dt(self) -> float:
        """Required by rsl_rl logger."""
        return self.get_policy_dt()

    @property
    def cfg(self) -> dict:
        """Required by rsl_rl logger."""
        return self.get_cfg_as_dict()

    @abstractmethod
    def get_policy_dt(self) -> float:
        """Returns the time period for policy inference."""
        ...

    @abstractmethod
    def get_cfg_as_dict(self) -> dict:
        """Return the environment config as dict."""
        ...

    @abstractmethod
    def get_num_envs(self) -> int:
        """Returns the number of parallel environments (1 for real robot)."""
        ...

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Returns observations at the current state. Derived from VecEnv."""
        ...

    @abstractmethod
    def reset(self) -> ResetReturn:
        """Resets the environment."""
        ...

    @abstractmethod
    def step(self, actions: th.Tensor) -> StepReturn:
        """Perform a step in environment. Derived from VecEnv."""
        ...
