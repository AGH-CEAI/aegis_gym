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

    # TODO(issue#9): Limit velocity and acceleration for Genesis robot commander
    # TODO(issue#23): Implement synchronous control for Genesis robot commander

    def control_dofs_position(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        raise NotImplementedError

    def control_dofs_position_servo(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        self.robot.control_dofs_position(target_pos, self.motor_dofs)

    def control_tcp_position(
        self,
        target_pos: th.Tensor,
        target_ori: th.Tensor,
        max_vel: float = 0.3,
        max_accel: float = 0.3,
    ) -> None:
        raise NotImplementedError

    def control_tcp_position_servo(
        self,
        target_pos: th.Tensor,
        target_ori: th.Tensor | None = None,
        max_vel: float = 0.3,
        max_accel: float = 0.3,
    ) -> None:
        if target_ori is None:
            target_ori = th.tensor(
                [0.0, 0.7071, 0.7071, 0.0],
                dtype=th.float32,
                device=self.device,
            )

        pos_np = target_pos.detach().cpu().numpy()
        ori_np = target_ori.detach().cpu().numpy()

        qpos = self.robot.inverse_kinematics(
            link=self.robot.get_link("robotiq_hande_end"),
            pos=pos_np,
            quat=ori_np,
        )[self.motor_dofs[0]:self.motor_dofs[-1]+1]

        self.robot.control_dofs_position(qpos, self.motor_dofs)

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
