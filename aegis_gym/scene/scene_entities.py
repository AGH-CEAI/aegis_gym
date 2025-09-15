from abc import ABC, abstractmethod
from enum import auto

import torch as th

# Python 3.11+: Change to builtin StrEnum
from strenum import StrEnum


class EntityType(StrEnum):
    TARGET = auto()
    BOX = auto()


class SceneEntity(ABC):
    def __init__(self, device: str):
        self._device = device

    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_pose(self, pose: th.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_pose(self) -> th.Tensor:
        raise NotImplementedError


class Target(SceneEntity):
    pass


class Box(SceneEntity):
    pass
