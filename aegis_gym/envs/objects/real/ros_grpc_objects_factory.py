import torch as th

from ..base_objects import ObjectProperties
from ..base_objects_factory import BaseObjectsFactory
from .ros_grpc_objects import RosGrpcBox


class RosGrpcObjectsFactory(BaseObjectsFactory):
    def __init__(self, device: th.device):
        super().__init__(device=device)

    def create_box(self, properties: ObjectProperties) -> RosGrpcBox:
        return RosGrpcBox(properties=properties, device=self.device)
