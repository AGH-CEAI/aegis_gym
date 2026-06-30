from abc import ABC, abstractmethod
from enum import auto
from strenum import StrEnum
from typing import Optional

import torch as th
import genesis as gs

from envs.scene import BaseScene


class ObjectType(StrEnum):
    BOX = auto()


class BaseObject(ABC):
    """
    An abstraction for an any object.
    """

    def __init__(
        self,
        scene: BaseScene | gs.Scene,  # TODO(issue#128) Remove gs.Scene from this API
        device: th.device = th.device("cpu"),
    ):
        """
        The `dims` are given per specialization, `pose` as X,Y,Z,QW,QX,QY,QZ and the `num_envs` as int.
        """
        self._scene = scene
        self.device = device

    @abstractmethod
    def create(
        self,
        dims: tuple,
        pose: tuple,  # TODO(issue#128) unify pose type for create() and set_pose()
        fixed: bool = False,
        collision: bool = True,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
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
    The interface for implementinc generic box object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
