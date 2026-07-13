import torch as th

from .base_objects import ObjectProperties, ObjectType, BaseBox, BaseObject
from .real.ros_grpc_objects import RosGrpcBox
from .sim.genesis_objects import GenesisBox

from config.types import Control


class ObjectsFactory:
    @classmethod
    def create_object(
        cls,
        ctrl_type: Control,
        obj_type: ObjectType,
        obj_properties: ObjectProperties,
        device: th.device,
    ) -> BaseObject:
        match obj_type:
            case ObjectType.BOX:
                return cls.create_box(
                    ctrl_type=ctrl_type,
                    properties=obj_properties,
                    device=device,
                )

    @classmethod
    def create_box(
        cls,
        ctrl_type: Control,
        properties: ObjectProperties,
        device: th.device,
    ) -> BaseBox:
        match ctrl_type:
            case Control.SIM:
                return GenesisBox(
                    properties=properties,
                    device=device,
                )
            case Control.ROS:
                return RosGrpcBox(
                    properties=properties,
                    device=device,
                )
