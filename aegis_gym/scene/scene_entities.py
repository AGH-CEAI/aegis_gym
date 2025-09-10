from abc import ABC, abstractmethod
from enum import auto
from typing import Any

# Python 3.11+: Change to builtin StrEnum
from strenum import StrEnum


class EntityType(StrEnum):
    TARGET = auto()
    BOX = auto()


class SceneEntity(ABC):
    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_pose(self, pose: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_pose(self) -> Any:
        raise NotImplementedError


class Target(SceneEntity):
    pass


class Box(SceneEntity):
    pass
