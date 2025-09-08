import numpy as np
import torch as th
import genesis as gs

from aegis_gym.robot.robot_commander_interface import SimulationCommanderInterface
from aegis_gym.sim.genesis.sim_manager_genesis import SimManagerGenesis


class GenesisCommander(SimulationCommanderInterface):
    def __init__(self):
        super().__init__()

        self.sim_gs = SimManagerGenesis()
        self.sim_gs.build()

        self.robot = self.sim_gs.get_robot()

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

    def step(self) -> None:
        self.sim_gs.step()

    def get_joint_positions(self) -> gs.Tensor:
        return self.robot.get_dofs_position(self.motor_dofs)

    def get_joint_velocities(self) -> gs.Tensor:
        return self.robot.get_dofs_velocity(self.motor_dofs)

    def get_tcp_position(self) -> gs.Tensor:
        return self.robot.get_links_pos()[7, :]

    def control_dofs_position(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        self.robot.control_dofs_position(target_pos, self.motor_dofs)

    def teleport_to_home(self) -> None:
        self.robot.set_dofs_position(
            position=self.dof_home,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        self.robot.zero_all_dofs_velocity()

    def move_to_home(self) -> None:
        self.control_dofs_position(self.dof_home)

    def publish_target_pos(self, pos: np.ndarray) -> None:
        # TODO remove ROS Reacher-only method
        # THis should be only visual
        pass

    def shutdown(self) -> None:
        pass
