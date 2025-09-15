import torch as th
from genesis.engine.entities.rigid_entity import RigidEntity

from ...scene.robot_commander_interface import RobotCommanderInterface


class RobotCommanderSimGenesis(RobotCommanderInterface):
    def __init__(
        self, gs_robot: RigidEntity, motor_dofs: tuple[str], device: str = "cuda"
    ):
        super().__init__(device)
        self.robot = gs_robot
        self.motor_dofs = motor_dofs

    def get_joint_positions(self) -> th.Tensor:
        return self.robot.get_dofs_position(self.motor_dofs).clone().detach()

    def get_joint_velocities(self) -> th.Tensor:
        return self.robot.get_dofs_velocity(self.motor_dofs).clone().detach()

    def get_tcp_position(self) -> th.Tensor:
        return self.robot.get_links_pos()[7, :].clone().detach()

    def get_tcp_orientation(self) -> th.Tensor:
        return self.robot.get_links_quat()[7, :].clone().detach()

    def get_tcp_pose(self) -> th.Tensor:
        pos = self.get_tcp_position()
        ori = self.get_tcp_orientation()
        return th.cat([pos, ori])

    def control_dofs_position(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        # TODO(issue#9) use max cel and accel
        self.robot.control_dofs_position(target_pos, self.motor_dofs)

    def move_to_home(self) -> None:
        # TODO(issue#8) Do we need trajectory to home in simulation?
        # self.control_dofs_position(self.dof_home)
        self._teleport_to_home()

    def _teleport_to_home(self) -> None:
        self.robot.set_dofs_position(
            position=self.dof_home,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        self.robot.zero_all_dofs_velocity()
