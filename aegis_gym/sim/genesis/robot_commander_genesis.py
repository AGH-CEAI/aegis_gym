from typing import Optional

import torch as th
from tensordict import TensorDict

from ...scene.robot_commander_interface import RobotCommanderInterface
from .manipulator import Manipulator


class RobotCommanderSimGenesis(RobotCommanderInterface):
    _instance: Optional["RobotCommanderInterface"] = None

    def __new__(cls, *args, **kwargs) -> "RobotCommanderInterface":
        if cls._instance is None:
            cls._instance = super(RobotCommanderInterface, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, manipulator: Manipulator, motor_dofs: tuple[str], device: str = "cuda"
    ):
        super().__init__(device)
        self.manipulator = manipulator
        self.gs_robot = manipulator.get_robot_entity()
        self.motor_dofs = motor_dofs
        self.tcp_link_name = "robotiq_hande_end"
        self.base_link_name = "world"

    def read_state(self) -> None:
        pass

    def get_state_tensordict(self) -> TensorDict:
        return TensorDict(
            {
                {
                    "pose": self.get_tcp_pose(),
                    "wrench": self.get_wrench(),
                    "joints_pos": self.get_joint_positions(),
                    "joints_vel": self.get_joint_velocities(),
                    "joints_eff": self.get_joints_efforts(),
                }
            }
        )

    def get_joint_positions(self) -> th.Tensor:
        return self.gs_robot.get_dofs_position(self.motor_dofs).clone().detach()

    def get_joint_velocities(self) -> th.Tensor:
        return self.gs_robot.get_dofs_velocity(self.motor_dofs).clone().detach()

    # TODO get F/T data from genesis
    def get_joints_efforts(self) -> th.Tensor:
        return th.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float32, device=self.device
        )

    def get_tcp_position(self) -> th.Tensor:
        return self.gs_robot.get_link(self.tcp_link_name).get_pos().clone().detach()

    def get_tcp_orientation(self) -> th.Tensor:
        return self.gs_robot.get_link(self.tcp_link_name).get_quat().clone().detach()

    def get_tcp_pose(self) -> th.Tensor:
        return self.manipulator.ee_pose

    # TODO get F/T data from genesis
    def get_wrench(self) -> th.Tensor:
        return th.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float32, device=self.device
        )

    def get_base_position(self) -> th.Tensor:
        return self.manipulator.base_pos

    # TODO(issue#9): Limit velocity and acceleration for Genesis robot commander

    # TODO(issue#23): Implement synchronous control for Genesis robot commander
    def control_dofs_position(
        self,
        target_pos: th.Tensor,
    ) -> None:
        raise NotImplementedError

    def control_dofs_position_servo(self, target_pos: Optional[th.Tensor]) -> None:
        self.gs_robot.control_dofs_position(target_pos, self.motor_dofs)

    def control_dofs_velocity_servo(
        self,
        target_vel: Optional[th.Tensor],
    ) -> None:
        raise NotImplementedError

    # TODO(issue#23): Implement synchronous control for Genesis robot commander
    def control_tcp_position(
        self,
        target_pos: th.Tensor,
        target_ori: Optional[th.Tensor],
    ) -> None:
        raise NotImplementedError

    def control_tcp_position_servo(
        self,
        target_pos: th.Tensor,
        target_ori_euler: Optional[th.Tensor],
    ) -> None:
        # TODO handle the manipulator API
        self.manipulator.apply_action(target_pos, open_gripper=True)

    def control_tcp_velocity_servo(
        self,
        target_pos: th.Tensor,
        target_ori_euler: Optional[th.Tensor],
    ) -> None:
        raise NotImplementedError

    def move_to_home(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        # TODO(issue#8) Do we need trajectory to home in simulation?
        # self.control_dofs_position(self.dof_home)
        self.manipulator.reset_home(envs_idx)

    def _teleport_to_home(self) -> None:
        self.gs_robot.set_dofs_position(
            position=self.dof_home,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        self.gs_robot.zero_all_dofs_velocity()
