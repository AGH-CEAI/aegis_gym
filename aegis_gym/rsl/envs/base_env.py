from abc import abstractmethod
from enum import auto
from typing import NamedTuple, Optional, Callable


import torch as th
from strenum import StrEnum
from tensordict import TensorDict
from rsl_rl.env import VecEnv

from .scene.base_scene import BaseScene


class ResetReturn(NamedTuple):
    """This return tuple is compatible with the rsl_rl's `OnPolicyRunner`"""

    observations: TensorDict
    extras: dict


class StepReturn(NamedTuple):
    """This return tuple is compatible with the rsl_rl's `OnPolicyRunner`"""

    observations: TensorDict
    rewards: th.Tensor
    dones: th.Tensor
    extras: dict


class Modality(StrEnum):
    TCP_POSE = auto()
    TCP_VELOCITY = auto()
    TCP_WRENCH = auto()
    CAMERA_SCENE_RGB = auto()
    CAMERA_TOOL_LEFT_RGB = auto()
    CAMERA_TOOL_RIGHT_RGB = auto()
    OBJECT_POSE = auto()


class BaseEnv(VecEnv):
    """
    Base class for implementing an environment compatible with rsl_rl's VecEnv.
    See https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/env/vec_env.py
    You need to define the `DEFAULT_MODALITIES` and `_observation_fns` manually.
    """

    DEFAULT_MODALITIES: frozenset[Modality]
    _observation_fns: dict[Modality, Callable[[], th.Tensor]]

    def __init__(self, scene: Optional[BaseScene]):
        super().__init__()
        self._scene: Optional[BaseScene] = scene
        self._obs_cache: TensorDict = TensorDict({}, batch_size=[self.get_num_envs()])

    def __del__(self):
        if self._scene:
            self._scene.shutdown()

    def _obs_cache_get(self, modalities: frozenset[Modality]) -> TensorDict:
        return self._obs_cache.select(*(m.value for m in modalities))

    def _obs_cache_set(self, modality: Modality, value: th.Tensor) -> None:
        self._obs_cache.set(modality.value, value)

    def _obs_cache_keys(self) -> frozenset[Modality]:
        return frozenset(Modality(k) for k in self._obs_cache.keys())

    def _obs_cache_clear(self) -> None:
        self._obs_cache = TensorDict({}, batch_size=[self.get_num_envs()])

    @property
    def unwrapped(self) -> "BaseEnv":
        """Get the underlying environment."""
        return self

    @property
    def step_dt(self) -> float:
        """Required by rsl_rl logger."""
        return self.get_policy_dt()

    @property
    def cfg(self) -> dict:
        """Required by rsl_rl logger."""
        return self.get_cfg_as_dict()

    @property
    def available_modalities(self) -> frozenset[Modality]:
        """Returns set of available observation modalities"""
        return frozenset(self._observation_fns)

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

    def get_observations(
        self, modalities: Optional[frozenset[Modality]] = None
    ) -> TensorDict:
        """
        Returns observations at the current state. Derived from rsl_rl's `VecEnv`.
        Different `modalities` can be obtained in the result `TensorDict`.
        """
        modalities = self._resolve_modalities(modalities)
        for m in modalities - self._obs_cache_keys():
            self._obs_cache_set(m, self._observation_fns[m]())
        return self._obs_cache_get(modalities)

    def _resolve_modalities(
        self, modalities: Optional[frozenset[Modality]]
    ) -> frozenset[Modality]:
        if modalities is None:
            return self.DEFAULT_MODALITIES
        unknown = modalities - self.available_modalities
        if unknown:
            raise ValueError(f"Unsupported modalities: {unknown}")
        return modalities

    def get_agent_observations(self) -> TensorDict:
        """Adapter: produces (policy_obs, extras) in the shape rsl_rl's `OnPolicyRunner` expects."""

        policy_obs = self._agent_observations_mapping()
        # TODO: incorporate more rsl_rl group obs
        return TensorDict({"policy": policy_obs}, batch_size=self.get_num_envs())

    @abstractmethod
    def _agent_observations_mapping(self) -> th.Tensor:
        """
        Maps default modalities from `get_observations()` into agent observation.
        Should be configured via given config.
        """
        ...

    @abstractmethod
    def reset(self) -> ResetReturn:
        """Resets the environment."""
        ...

    @abstractmethod
    def step(self, actions: th.Tensor) -> StepReturn:
        """Perform a step in environment. Derived from VecEnv."""
        ...
