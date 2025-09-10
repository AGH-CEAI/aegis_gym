from .robot_commander_interface import RobotCommanderInterface
import numpy as np


class RobotCommanderMock(RobotCommanderInterface):
    def __init__(self) -> None:
        super().__init__()
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.dof_home = {name: 0.0 for name in self.joint_names}

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(len(self.joint_names), dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        return np.zeros(len(self.joint_names), dtype=np.float32)

    def get_tcp_position(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def control_dofs_position(
        self, target_pos: np.ndarray, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        pass

    def move_to_home(self) -> None:
        pass
