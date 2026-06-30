import torch as th
import genesis as gs

from .base_objects import ObjectType, BaseBox, BaseObject
from .real.ros_grpc_objects import RosGrpcBox
from .sim.genesis_objects import GenesisBox
from ..scene import BaseScene

from config.types import Control


class ObjectsFactory:
    @classmethod
    def create_object(
        cls,
        obj_type: ObjectType,
        scene: BaseScene | gs.Scene,  # TODO(issue@128) remove genesis from this api
        ctrl: Control,
        device: th.device,
    ) -> BaseObject:
        match obj_type:
            case ObjectType.BOX:
                return cls.create_box(
                    scene=scene,
                    ctrl=ctrl,
                    device=device,
                )

    @classmethod
    def create_box(
        cls,
        scene: BaseScene | gs.Scene,  # TODO(issue@128) remove genesis from this api
        ctrl: Control,
        device: th.device,
    ) -> BaseBox:
        match ctrl:
            case Control.SIM:
                return GenesisBox(
                    scene=scene,
                    device=device,
                )
            case Control.ROS:
                return RosGrpcBox(
                    scene=scene,
                    device=device,
                )
