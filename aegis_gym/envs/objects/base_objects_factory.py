from abc import ABC, abstractmethod

import torch as th

from .base_objects import BaseBox, BaseMesh, BaseObject, ObjectProperties, ObjectType


class BaseObjectsFactory(ABC):
    """Template for objects factory."""

    def __init__(self, device: th.device):
        self.device = device

    def create_object(
        self,
        obj_type: ObjectType,
        obj_properties: ObjectProperties,
    ) -> BaseObject:
        """Creates a generic object of `obj_type` type."""
        match obj_type:
            case ObjectType.BOX:
                return self.create_box(
                    properties=obj_properties,
                )
            case ObjectType.MESH:
                return self.create_mesh(
                    properties=obj_properties,
                )

    @abstractmethod
    def create_box(
        self,
        properties: ObjectProperties,
    ) -> BaseBox:
        """Creates a box object."""
        ...

    @abstractmethod
    def create_mesh(
        self,
        properties: ObjectProperties,
    ) -> BaseMesh:
        """Creates a mesh-file-based object."""
        ...
