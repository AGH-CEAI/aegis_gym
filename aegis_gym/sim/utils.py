from typing import Union, Any, NamedTuple

import gymnasium as gym
import torch as th
import numpy as np


class Dimensions(NamedTuple):
    x: float
    y: float
    z: float


class TorchToNumpyWrapper(gym.ObservationWrapper, gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs: Union[th.Tensor, Any]) -> np.ndarray:
        # Convert thTensor observation to numpy array
        if isinstance(obs, th.Tensor):
            return obs.cpu().numpy()
        return obs

    def action(self, action: Union[np.ndarray, Any]) -> th.Tensor:
        # Convert numpy array action to thTensor
        if isinstance(action, np.ndarray):
            return th.from_numpy(action)
        return action
