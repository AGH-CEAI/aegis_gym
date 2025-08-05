import torch
from typing import Optional
import rclpy
from rclpy.clock import Clock
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from aegis_director.robot_director import RobotDirector


class ROSInterface:
    _instance: Optional["ROSInterface"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ROSInterface, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        rclpy.init()
        self.robot_director = RobotDirector(synchronous=True)
        joint_state = self.robot_director._get_joint_states()
        self.joint_names = list(joint_state.name)[1:]
        self.dof_home = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -2.09,
            "elbow_joint": 2.09,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            "robotiq_hande_left_finger_joint": 0.025,
        }
        self.marker_node = rclpy.create_node("marker_publisher")
        self.target_pub = self.marker_node.create_publisher(
            Marker, "/target_marker", 10
        )
        self._initialized = True

    def get_joint_positions(self) -> torch.Tensor:
        jp = self.robot_director.get_joint_positions()
        return torch.tensor(
            [jp[name] for name in self.joint_names], dtype=torch.float32
        )

    def get_joint_velocities(self) -> torch.Tensor:
        jv = self.robot_director.get_joint_velocities()
        return torch.tensor(
            [jv[name] for name in self.joint_names], dtype=torch.float32
        )

    def get_tcp_position(self) -> torch.Tensor:
        tcp = self.robot_director.get_tcp_pose()
        return torch.tensor(tcp["position"], dtype=torch.float32)

    def control_dofs_position(
        self, target_pos: torch.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ):
        joint_dict = {
            name: float(pos) for name, pos in zip(self.joint_names, target_pos)
        }
        self.robot_director.joint_move(
            joint_positions=joint_dict, max_vel=max_vel, max_accel=max_accel
        )

    def move_to_home(self):
        self.robot_director.joint_move(
            joint_positions=self.dof_home,
            max_vel=0.5,
            max_accel=0.5,
        )

    def publish_target_pos(self, pos):
        msg = Marker()
        msg.header.frame_id = "world"
        msg.header.stamp = Clock().now().to_msg()

        msg.ns = "target"
        msg.id = 0
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = 1.0

        msg.scale.x = 0.04
        msg.scale.y = 0.04
        msg.scale.z = 0.04

        msg.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        self.target_pub.publish(msg)

    def shutdown(self):
        rclpy.shutdown()

    def __del__(self):
        self.shutdown()
