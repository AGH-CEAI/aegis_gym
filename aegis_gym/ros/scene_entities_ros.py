import torch as th

from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from ..scene import EntityType, Target, Box


class TargetROS(Target):
    def __init__(self, node: Node, device: str = "cuda"):
        super().__init__(device)
        self._node = node
        self._pose: th.Tensor = th.zeros()

    def create(self, topic: str = "/target_marker") -> None:
        self._target_pub = self._node.create_publisher(Marker, topic, 10)

    def set_pose(self, pose: th.Tensor) -> None:
        self._pose = pose
        p = pose.clone().cpu().numpy()

        msg = Marker()
        msg.header.frame_id = "world"
        msg.header.stamp = Clock().now().to_msg()

        msg.ns = "target"
        msg.id = 0
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD

        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])
        msg.pose.orientation.w = 1.0
        if len(p) > 3:
            msg.pose.orientation.x = float(p[3])
            msg.pose.orientation.y = float(p[4])
            msg.pose.orientation.z = float(p[5])
            msg.pose.orientation.w = float(p[6])

        msg.scale.x = 0.04
        msg.scale.y = 0.04
        msg.scale.z = 0.04

        msg.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        self._target_pub.publish(msg)

    def get_pose(self) -> th.Tensor:
        return self._pose


class BoxROS(Box):
    def __init__(self, node: Node, device: str = "cuda"):
        super().__init__(device)
        self._node = node
        raise NotImplementedError(
            "The Box scene object is not implemented for ROS usage."
        )

    def create(self) -> None:
        pass

    def set_pose(self, pose: th.Tensor) -> None:
        pass

    def get_pose(self) -> th.Tensor:
        return th.zeros()


EntityTypeROS = {
    EntityType.TARGET: TargetROS,
    EntityType.BOX: BoxROS,
}
