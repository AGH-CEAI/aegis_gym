from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto

import torch as th
from strenum import StrEnum


@dataclass(frozen=True, slots=True)
class ObjectProperties:
    dims: tuple
    pose: tuple[float, ...] | None
    fixed: bool = False
    collision: bool = True
    color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    mesh_path: str | None = None
    urdf_path: str | None = None
    friction: float | None = None
    scale: float | tuple[float, float, float] = 1.0


class ObjectType(StrEnum):
    BOX = auto()
    MESH = auto()
    URDF = auto()


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
    ) -> None:
        """Create the object."""
        ...

    @abstractmethod
    def get_pose(self, envs_idx: th.Tensor | int | None = None) -> th.Tensor:
        """Get the object pose (from all envs). Returns 7xN_ENVs tensor."""
        ...

    @abstractmethod
    def set_pose(
        self, pose: th.Tensor, envs_idx: th.Tensor | int | None = None
    ) -> None:
        """Set the object pose (for all envs)."""
        ...


class BaseBox(BaseObject):
    """
    The interface for implementing generic box object.
    """

    def __init__(self, device: th.device, properties: ObjectProperties):
        super().__init__(device=device, properties=properties)


class BaseMesh(BaseObject):
    """
    The interface for implementing a generic mesh-file-based object.
    """

    def __init__(self, device: th.device, properties: ObjectProperties):
        super().__init__(device=device, properties=properties)


class BaseURDF(BaseObject):
    """
    The interface for implementing a generic URDF-file-based object.
    """

    def __init__(self, device: th.device, properties: ObjectProperties):
        super().__init__(device=device, properties=properties)
