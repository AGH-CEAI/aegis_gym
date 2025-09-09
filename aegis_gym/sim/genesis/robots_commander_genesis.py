import numpy as np
import torch as th

from ...robot.robot_commander_interface import RobotCommanderInterface
from .sim_manager_genesis import SimManagerGenesis


class RobotCommanderSimGenesis(RobotCommanderInterface):
    def __init__(self):
        super().__init__()

        self.sim_gs = SimManagerGenesis(show_viewer=True)
        self.sim_gs.build()

        self.robot = self.sim_gs.get_robot()

    def step(self) -> None:
        self.sim_gs.step()

    # def get_joint_positions(self) -> gs.Tensor:
    def get_joint_positions(self) -> np.ndarray:
        return self.robot.get_dofs_position(self.sim_gs.motor_dofs).cpu().numpy()

    # def get_joint_velocities(self) -> gs.Tensor:
    def get_joint_velocities(self) -> np.ndarray:
        return self.robot.get_dofs_velocity(self.sim_gs.motor_dofs).cpu().numpy()

    # def get_tcp_position(self) -> gs.Tensor:
    def get_tcp_position(self) -> np.ndarray:
        return self.robot.get_links_pos()[7, :].cpu().numpy()

    def control_dofs_position(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        self.robot.control_dofs_position(target_pos, self.sim_gs.motor_dofs)

    def teleport_to_home(self) -> None:
        self.robot.set_dofs_position(
            position=self.sim_gs.dof_home,
            dofs_idx_local=self.sim_gs.motor_dofs,
            zero_velocity=True,
        )
        self.robot.zero_all_dofs_velocity()

    def move_to_home(self) -> None:
        # self.control_dofs_position(self.sim_gs.dof_home)
        self.teleport_to_home()

    def publish_target_pos(self, pos: np.ndarray) -> None:
        # TODO remove ROS Reacher-only method
        # THis should be only visual
        pass

    def shutdown(self) -> None:
        pass
