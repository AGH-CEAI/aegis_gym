from typing import Optional
import torch as th
from tensordict import TensorDict
from .robot_commander_interface import RobotCommanderInterface


class RobotCommanderMock(RobotCommanderInterface):
    def __init__(self, device: str) -> None:
        super().__init__(device)

    def read_state(self) -> None:
        pass

    def get_state_tensordict(self) -> TensorDict:
        state = th.zeros(len(self.joint_names), dtype=th.float32, device=self.device)
        raise TensorDict({"state": state})

    def get_joint_positions(self) -> th.Tensor:
        return th.zeros(len(self.joint_names), dtype=th.float32, device=self.device)

    def get_joint_velocities(self) -> th.Tensor:
        return th.zeros(len(self.joint_names), dtype=th.float32, device=self.device)

    def get_joints_efforts(self) -> th.Tensor:
        return th.zeros(len(self.joint_names), dtype=th.float32, device=self.device)

    def get_tcp_position(self) -> th.Tensor:
        return th.tensor([0.0, 0.0, 0.0], dtype=th.float32, device=self.device)

    def get_tcp_orientation(self) -> th.Tensor:
        return th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32, device=self.device)

    def get_tcp_pose(self) -> th.Tensor:
        return th.tensor(
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=th.float32, device=self.device
        )

    def get_wrench(self) -> th.Tensor:
        return th.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=th.float32, device=self.device
        )

    def get_base_position(self) -> th.Tensor:
        return th.tensor([0.0, 0.0, 0.0], dtype=th.float32, device=self.device)

    def control_dofs_position(self, target_pos: th.Tensor) -> None:
        pass

    def control_dofs_position_servo(self, target_pos: Optional[th.Tensor]) -> None:
        pass

    def control_dofs_velocity_servo(self, target_vel: Optional[th.Tensor]) -> None:
        pass

    def control_tcp_position(
        self,
        target_pos: th.Tensor,
        target_ori: Optional[th.Tensor],
    ) -> None:
        pass

    def control_tcp_position_servo(
        self,
        target_pos: th.Tensor,
        target_ori_euler: Optional[th.Tensor],
    ) -> None:
        pass

    def control_tcp_velocity_servo(
        self,
        target_pos: th.Tensor,
        target_ori_euler: Optional[th.Tensor],
    ) -> None:
        pass

    def move_to_home(self) -> None:
        pass
