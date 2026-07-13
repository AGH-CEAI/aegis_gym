from abc import ABC, abstractmethod
from enum import auto
from dataclasses import dataclass
from strenum import StrEnum
from typing import Optional

import torch as th


@dataclass(frozen=True, slots=True)
class ObjectProperties:
    dims: tuple
    pose: Optional[tuple[float]]
    fixed: bool = False
    collision: bool = True
    color: tuple[float, float, float] = (1.0, 0.0, 0.0)


class ObjectType(StrEnum):
    BOX = auto()


class BaseObject(ABC):
    """
    An abstraction for an any object.
    """

    def __init__(
        self,
        properties: ObjectProperties,
        device: th.device,
    ):
        self.properties = properties
        self.device = device

    @abstractmethod
    def create(
        self,
        **kwargs,
    ) -> None:
        """Create the object."""
        ...

    @abstractmethod
    def get_pose(self, envs_idx: Optional[th.Tensor | int] = None) -> th.Tensor:
        """Get the object pose (from all envs). Returns 7xN_ENVs tensor."""
        ...

    @abstractmethod
    def set_pose(
        self, pose: th.Tensor, envs_idx: Optional[th.Tensor | int] = None
    ) -> None:
        """Set the object pose (for all envs)."""
        ...


class BaseBox(BaseObject):
    """
    The interface for implementing generic box object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
