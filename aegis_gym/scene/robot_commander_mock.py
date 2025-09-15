from .robot_commander_interface import RobotCommanderInterface
import torch as th


class RobotCommanderMock(RobotCommanderInterface):
    def __init__(self, device: str) -> None:
        super().__init__(device)
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.dof_home = {name: 0.0 for name in self.joint_names}

    def get_joint_positions(self) -> th.Tensor:
        return th.zeros(len(self.joint_names), dtype=th.float32, device=self.device)

    def get_joint_velocities(self) -> th.Tensor:
        return th.zeros(len(self.joint_names), dtype=th.float32, device=self.device)

    def get_tcp_position(self) -> th.Tensor:
        return th.tensor([0.0, 0.0, 0.0], dtype=th.float32, device=self.device)

    def control_dofs_position(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        pass

    def move_to_home(self) -> None:
        pass
