import torch as th

from ..base_objects import ObjectProperties
from ..base_objects_factory import BaseObjectsFactory
from .ros_grpc_objects import RosGrpcBox, RosGrpcMesh, RosGrpcURDF


class RosGrpcObjectsFactory(BaseObjectsFactory):
    def __init__(self, device: th.device):
        super().__init__(device=device)

    def create_box(self, properties: ObjectProperties) -> RosGrpcBox:
        return RosGrpcBox(properties=properties, device=self.device)

    # TODO(issue#151): Implement mesh/URDF object support on the real ROS/gRPC hardware bridge
    def create_mesh(self, properties: ObjectProperties) -> RosGrpcMesh:
        return RosGrpcMesh(properties=properties, device=self.device)
        raise NotImplementedError(
            "Mesh objects are not supported on the real ROS/gRPC hardware bridge."
        )

    def create_urdf(self, properties: ObjectProperties) -> RosGrpcURDF:
        return RosGrpcURDF(properties=properties, device=self.device)
        # raise NotImplementedError(
        #     "URDF objects are not supported on the real ROS/gRPC hardware bridge."
        # )
