from abc import ABC, abstractmethod
from typing import NamedTuple
import torch as th
from tensordict import TensorDict

from scene import BaseScene


class ResetObservation(NamedTuple):
    observations: TensorDict
    extras: dict


class Observation(NamedTuple):
    observations: TensorDict
    rewards: th.Tensor
    dones: th.Tensor
    extras: dict


class BaseEnv(ABC):
    """
    Base class for implementing an environment compatible with rsl_rl.
    """

    def __init__(self, scene: BaseScene):
        super().__init__()
        self._scene: BaseScene = scene

    def __del__(self):
        self._scene.shutdown()

    @property
    def unwrapped(self) -> "BaseEnv":
        """Required by rsl_rl."""
        return self

    @property
    def step_dt(self) -> float:
        """Required by rsl_rl."""
        return self.get_policy_dt()

    @property
    def cfg(self) -> dict:
        """Required by rsl_rl."""
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
    def reset(self) -> ResetObservation:
        """Resets the environment."""
        ...

    @abstractmethod
    def step(self, actions: th.Tensor) -> Observation:
        """Perform a step in environment."""
        ...

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Returns observations at the current state."""
        ...

    @abstractmethod
    def get_privileged_observations(self) -> TensorDict:
        """Returns privileged observations at the current state."""
        ...

    @abstractmethod
    def is_episode_complete(self) -> th.Tensor:
        """Returns binary vector of length `n_envs` where ones indices complention."""
        ...
