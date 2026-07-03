from typing import Optional

import torch as th
from tensordict import TensorDict

from ..base_env import BaseEnv, ResetReturn, StepReturn, Modality


class BaseEnvWrapper(BaseEnv):
    """
    An abstract class to implement environment wrappers
    """

    def __init__(self, env: BaseEnv):
        super().__init__(scene=None)
        self._env = env
        del self._obs_cache

    @property
    def unwrapped(self) -> "BaseEnv":
        return self._env

    def get_policy_dt(self) -> float:
        return self._env.get_policy_dt()

    def get_cfg_as_dict(self) -> dict:
        return self._env.get_cfg_as_dict()

    def get_num_envs(self) -> int:
        return self._env.get_num_envs()

    def get_observations(
        self, modalities: Optional[frozenset[Modality]] = None
    ) -> TensorDict:
        return self._env.get_observations(modalities=modalities)

    def get_agent_observations(self) -> TensorDict:
        return self._env.get_agent_observations()

    def reset(self) -> ResetReturn:
        return self._env.reset()

    def step(self, actions: th.Tensor) -> StepReturn:
        return self._env.step(actions)
