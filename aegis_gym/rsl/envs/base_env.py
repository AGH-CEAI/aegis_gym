from abc import abstractmethod
from enum import auto
from typing import NamedTuple, Optional, Callable, Collection


import torch as th
from strenum import StrEnum
from tensordict import TensorDict
from rsl_rl.env import VecEnv

from .scene.base_scene import BaseScene
from config.types import EnvCfg, CamerasSetup


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


IMAGE_MODALITIES: tuple[Modality, ...] = (
    Modality.CAMERA_SCENE_RGB,
    Modality.CAMERA_TOOL_LEFT_RGB,
    Modality.CAMERA_TOOL_RIGHT_RGB,
)


class BaseEnv(VecEnv):
    """
    Base class for implementing an environment compatible with rsl_rl's VecEnv.
    See https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/env/vec_env.py
    You need to define the `DEFAULT_MODALITIES` and `_observation_fns` manually.
    """

    DEFAULT_MODALITIES: frozenset[Modality]
    _observation_fns: dict[Modality, Callable[[], th.Tensor]]

    def __init__(self, scene: Optional[BaseScene], cfg: EnvCfg):
        super().__init__()
        self._scene: Optional[BaseScene] = scene
        self._cfg = cfg
        self.num_envs = cfg.num_envs
        self._obs_cache: TensorDict = TensorDict({}, batch_size=[self.num_envs])

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
        self._obs_cache = TensorDict({}, batch_size=[self.num_envs])

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

    def get_cameras_setup(self) -> CamerasSetup:
        """The environment cameras setup"""
        return self._cfg.cameras_setup

    def get_modality_observations(
        self, modalities: Optional[Collection[Modality]] = None
    ) -> TensorDict:
        """
        Returns observations at the current state. Not implicitly compatible with rsl_rl.
        Different `modalities` can be obtained in the result `TensorDict`.
        """
        modalities = self._resolve_modalities(modalities)
        for m in modalities - self._obs_cache_keys():
            self._obs_cache_set(m, self._observation_fns[m]())
        return self._obs_cache_get(modalities)

    def _resolve_modalities(
        self, modalities: Optional[Collection[Modality]]
    ) -> frozenset[Modality]:
        if modalities is None:
            return self.DEFAULT_MODALITIES
        modalities = frozenset(modalities)
        unknown = modalities - self.available_modalities
        if unknown:
            raise ValueError(f"Unsupported modalities: {unknown}")
        return modalities

    def get_observations(self) -> TensorDict:
        """
        Adapter: produces (policy_obs, extras) in the shape rsl_rl's `OnPolicyRunner` expects.

        Returns observations at the current state. Derived from rsl_rl's `VecEnv`.
        """

        obs = self.get_modality_observations()
        agent_obs = self._build_agent_observations(obs)
        return self._format_rslrl_observations(agent_obs)

    def _format_rslrl_observations(self, agent_obs: th.Tensor) -> TensorDict:
        """
        Formats the `th.Tensor` observation into rsl_rl's TensorDict format.
        """
        return TensorDict(
            {"policy": agent_obs}, batch_size=self.num_envs, device=self.device
        )

    @abstractmethod
    def _build_agent_observations(self, obs: TensorDict) -> th.Tensor:
        """
        Maps default modalities from `get_modality_observations()` into an agent observation.
        The result `TensorDict` must match the observation groups from the rsl_rl's configuration file.
        """
        ...

    def reset(self) -> ResetReturn:
        """Resets the environment. Derived from VecEnv. Needs the `_reset()` implementation."""
        self._obs_cache_clear()
        return self._reset()

    @abstractmethod
    def _reset(self) -> ResetReturn:
        """Resets the environment."""
        ...

    def step(self, actions: th.Tensor) -> StepReturn:
        """Perform a step in environment. Derived from VecEnv. Needs the `_step()` implementation."""
        self._obs_cache_clear()
        return self._step(actions=actions)

    @abstractmethod
    def _step(self, actions: th.Tensor) -> StepReturn:
        """Perform a step in environment. Derived from VecEnv."""
        ...
