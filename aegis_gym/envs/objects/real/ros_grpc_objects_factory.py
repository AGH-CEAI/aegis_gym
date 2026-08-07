import torch as th

from ..base_objects import BaseMesh, ObjectProperties
from ..base_objects_factory import BaseObjectsFactory
from .ros_grpc_objects import RosGrpcBox


class RosGrpcObjectsFactory(BaseObjectsFactory):
    def __init__(self, device: th.device):
        super().__init__(device=device)

    def create_box(self, properties: ObjectProperties) -> RosGrpcBox:
        return RosGrpcBox(properties=properties, device=self.device)

    def create_mesh(self, properties: ObjectProperties) -> BaseMesh:
        raise NotImplementedError(
            "Mesh objects are not supported on the real ROS/gRPC hardware bridge."
        )
