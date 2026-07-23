from abc import ABC, abstractmethod

import torch as th

from .base_objects import ObjectProperties, ObjectType, BaseBox, BaseObject


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

    @abstractmethod
    def create_box(
        self,
        properties: ObjectProperties,
    ) -> BaseBox:
        """Creates a box object."""
        ...
