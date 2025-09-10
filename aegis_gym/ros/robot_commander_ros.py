import numpy as np
from typing import Optional

try:
    from aegis_director.robot_director import RobotDirector
except ImportError:
    print(
        "Failed to import aegis_director. Double check if you have sourced the AGH-CEAI/aegis_ros project."
    )
    raise ImportError

from ..scene import RobotCommanderInterface


class RobotCommanderROS(RobotCommanderInterface):
    _instance: Optional["RobotCommanderROS"] = None

    def __new__(cls) -> "RobotCommanderROS":
        if cls._instance is None:
            cls._instance = super(RobotCommanderROS, cls).__new__(cls)
        return cls._instance

    def __init__(self, robot_director: RobotDirector) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.robot_director = robot_director
        joint_state = self.robot_director._get_joint_states()
        self.joint_names = list(joint_state.name)[1:]

        # TODO Take HOME position from SRDF file.
        self.dof_home = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -2.09,
            "elbow_joint": 2.09,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            "robotiq_hande_left_finger_joint": 0.025,
        }
        self._initialized = True

    def get_joint_positions(self) -> np.ndarray:
        jp = self.robot_director.get_joint_positions()
        return np.array([jp[name] for name in self.joint_names], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        jv = self.robot_director.get_joint_velocities()
        return np.array([jv[name] for name in self.joint_names], dtype=np.float32)

    def get_tcp_position(self) -> np.ndarray:
        tcp = self.robot_director.get_tcp_pose()
        return np.array(tcp["position"], dtype=np.float32)

    def control_dofs_position(
        self, target_pos: np.ndarray, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        joint_dict = {
            name: float(pos) for name, pos in zip(self.joint_names, target_pos)
        }
        self.robot_director.joint_move(
            joint_positions=joint_dict, max_vel=max_vel, max_accel=max_accel
        )

    def move_to_home(self) -> None:
        self.robot_director.joint_move(
            joint_positions=self.dof_home,
            max_vel=0.5,
            max_accel=0.5,
        )
