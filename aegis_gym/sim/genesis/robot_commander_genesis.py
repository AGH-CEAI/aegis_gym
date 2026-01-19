from typing import Optional

import torch as th
from tensordict import TensorDict
from genesis.engine.entities.rigid_entity import RigidEntity

from ...scene.robot_commander_interface import RobotCommanderInterface


class RobotCommanderSimGenesis(RobotCommanderInterface):
    _instance: Optional["RobotCommanderInterface"] = None

    def __new__(cls, *args, **kwargs) -> "RobotCommanderInterface":
        if cls._instance is None:
            cls._instance = super(RobotCommanderInterface, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, gs_robot: RigidEntity, motor_dofs: tuple[str], device: str = "cuda"
    ):
        super().__init__(device)
        self.robot = gs_robot
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
        return self.robot.get_dofs_position(self.motor_dofs).clone().detach()

    def get_joint_velocities(self) -> th.Tensor:
        return self.robot.get_dofs_velocity(self.motor_dofs).clone().detach()

    # TODO get F/T data from genesis
    def get_joints_efforts(self) -> th.Tensor:
        return th.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float32, device=self.device
        )

    def get_tcp_position(self) -> th.Tensor:
        return self.robot.get_link(self.tcp_link_name).get_pos().clone().detach()

    def get_tcp_orientation(self) -> th.Tensor:
        return self.robot.get_link(self.tcp_link_name).get_quat().clone().detach()

    def get_tcp_pose(self) -> th.Tensor:
        pos = self.get_tcp_position()
        ori = self.get_tcp_orientation()
        return th.cat([pos, ori])

    # TODO get F/T data from genesis
    def get_wrench(self) -> th.Tensor:
        return th.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float32, device=self.device
        )

    def get_base_position(self) -> th.Tensor:
        return self.robot.get_link(self.base_link_name).get_pos().clone().detach()

    # TODO(issue#9): Limit velocity and acceleration for Genesis robot commander

    # TODO(issue#23): Implement synchronous control for Genesis robot commander
    def control_dofs_position(
        self,
        target_pos: th.Tensor,
    ) -> None:
        raise NotImplementedError

    def control_dofs_position_servo(self, target_pos: Optional[th.Tensor]) -> None:
        self.robot.control_dofs_position(target_pos, self.motor_dofs)

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
        if target_ori_euler is None:
            target_ori_euler = th.tensor(
                # TODO(issue#25): Investigate TCP frame reference discrepancy
                [0.0, 0.7071, 0.0, 0.7071],
                # [1.0, 0.0, 0.0, 0.0],
                dtype=th.float32,
                device=self.device,
            )

        pos_np = target_pos.detach().cpu().numpy()
        ori_np = target_ori_euler.detach().cpu().numpy()

        qpos = self.robot.inverse_kinematics(
            link=self.robot.get_link(self.tcp_link_name),
            pos=pos_np,
            quat=ori_np,
        )[self.motor_dofs[0] : self.motor_dofs[-1] + 1]

        self.robot.control_dofs_position(qpos, self.motor_dofs)

    def control_tcp_velocity_servo(
        self,
        target_pos: th.Tensor,
        target_ori_euler: Optional[th.Tensor],
    ) -> None:
        raise NotImplementedError

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
