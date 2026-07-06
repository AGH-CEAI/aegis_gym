from typing import Optional, Collection, Any

import torch as th
from tensordict import TensorDict

from ..base_env import BaseEnv, ResetReturn, StepReturn, Modality


class BaseEnvWrapper(BaseEnv):
    """
    An abstract class to implement environment wrappers
    """

    def __init__(self, env: BaseEnv):
        super().__init__(scene=None, num_envs=env.num_envs)
        self._env = env
        del self._obs_cache

    def __getattr__(self, name: str) -> Any:
        if name == "_env":
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> "BaseEnv":
        return self._env

    def get_policy_dt(self) -> float:
        return self._env.get_policy_dt()

    def get_cfg_as_dict(self) -> dict:
        return self._env.get_cfg_as_dict()

    def get_modality_observations(
        self, modalities: Optional[Collection[Modality]] = None
    ) -> TensorDict:
        return self._env.get_modality_observations(modalities=modalities)

    def get_observations(self) -> TensorDict:
        return self._env.get_observations()

    def _build_agent_observations(self, obs: TensorDict) -> th.Tensor:
        return self._env._build_agent_observations(obs)

    def reset(self) -> ResetReturn:
        return self._env.reset()

    def _reset(self) -> ResetReturn:
        return self._env._reset()

    def step(self, actions: th.Tensor) -> StepReturn:
        return self._env.step(actions)

    def _step(self, actions: th.Tensor) -> StepReturn:
        return self._env._step(actions)
