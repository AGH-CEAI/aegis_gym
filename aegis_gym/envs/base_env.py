from abc import abstractmethod
from collections.abc import Callable, Collection
from typing import NamedTuple

import torch as th
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from aegis_gym.config.types import (
    CamerasSetup,
    DomainRandomizationCfg,
    EnvCfg,
    ExpConfig,
    Modality,
)
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .scene.base_scene import BaseScene

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


class BaseEnv(VecEnv):
    """
    Base class for implementing an environment compatible with rsl_rl's VecEnv.
    See https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/env/vec_env.py
    You need to define the `DEFAULT_MODALITIES` and `_observation_fns` manually.
    """

    DEFAULT_MODALITIES: frozenset[Modality]
    _observation_fns: dict[Modality, Callable[[], th.Tensor]]

    def __init__(self, scene: BaseScene, cfg: ExpConfig):
        super().__init__()
        self._scene: BaseScene = scene
        self._cfg_env: EnvCfg = cfg.env_cfg
        self._cfg_dr: DomainRandomizationCfg = cfg.dr_cfg
        self.num_envs = cfg.env_cfg.num_envs
        self.device = cfg.get_device()
        self._obs_cache: TensorDict = TensorDict({}, batch_size=[self.num_envs])

    def __del__(self):
        if self._scene:
            self._scene.shutdown()

    def _obs_cache_get(self, modalities: frozenset[Modality]) -> TensorDict:
        return self._obs_cache.select(*(m.value for m in modalities))

    def _obs_cache_set(self, modality: Modality, value: th.Tensor) -> None:
        self._obs_cache.set(modality.value, value)

    def _obs_cache_keys(self) -> frozenset[Modality]:
        return frozenset(Modality(k) for k in self._obs_cache.keys())  # noqa: SIM118

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

    def get_policy_dt(self) -> float:
        """Returns the time period for policy inference."""
        if self._scene is None:
            raise RuntimeError("Missing scene definition to call get_policy_dt().")
        return self._scene.get_policy_dt()

    @abstractmethod
    def get_cfg_as_dict(self) -> dict:
        """Return the environment config as dict."""
        ...

    def get_cameras_setup(self) -> CamerasSetup:
        """The environment cameras setup"""
        return self._cfg_env.cameras_setup

    def get_modality_observation(self, modality: Modality) -> th.Tensor:
        """
        Returns a selected observation `modality` at the current state.
        """
        if modality not in self.available_modalities:
            raise ValueError(f"Unsupported modality: {modality}")
        if modality not in self._obs_cache:
            self._obs_cache_set(modality, self._observation_fns[modality]())
        return self._obs_cache_get(frozenset({modality}))[modality]

    def get_modality_observations(
        self, modalities: Collection[Modality] | None = None
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
        self, modalities: Collection[Modality] | None
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
