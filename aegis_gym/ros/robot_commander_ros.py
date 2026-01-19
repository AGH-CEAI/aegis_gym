import asyncio
from typing import Optional

import numpy as np
import torch as th

from tensordict import TensorDict

try:
    from aegis_grpc_client import AegisRobotClient
except ImportError:
    print(
        "Failed to import aegis_grpc_client. "
        "Double check if you have installed the `aegis_grpc_client` and `proto_aegis_grpc` packages."
    )
    raise

from ..scene import RobotCommanderInterface


class RobotCommanderROS(RobotCommanderInterface):
    _instance: Optional["RobotCommanderROS"] = None

    def __new__(cls, *args, **kwargs) -> "RobotCommanderROS":
        if cls._instance is None:
            cls._instance = super(RobotCommanderROS, cls).__new__(cls)
        return cls._instance

    def __init__(self, robot_client: AegisRobotClient, device: str) -> None:
        super().__init__(device=device)
        self._robot_client = robot_client

        # Dynamically obtain joints names
        self.joint_names = asyncio.run(self._robot_client.get_joint_names())[1:]
        self.default_orientaton = np.array([0, 0, 0, 1], dtype=np.float32)

        # Prepare initial observation
        self._state: Optional[TensorDict] = None
        self.read_state()

    def read_state(self) -> None:
        state = asyncio.run(self._robot_client.get_all())
        self._state = TensorDict(
            {
                "pose": th.from_numpy(state["pose"]),
                "wrench": th.from_numpy(state["wrench"]),
                "joints_pos": th.from_numpy(state["joints_pos"]),
                "joints_vel": th.from_numpy(state["joints_vel"]),
                "joints_eff": th.from_numpy(state["joints_eff"]),
            }
        )

    def get_state_tensordict(self) -> TensorDict:
        return self._state

    def get_joints_positions(self) -> th.Tensor:
        return self._state["joints_pos"].to(device=self.device, dtype=th.float32)

    def get_joints_velocities(self) -> th.Tensor:
        return self._state["joints_vel"].to(device=self.device, dtype=th.float32)

    def get_joints_efforts(self) -> th.Tensor:
        return self._state["joints_eff"].to(device=self.device, dtype=th.float32)

    def get_tcp_position(self) -> th.Tensor:
        return self._state["pose"].to(device=self.device, dtype=th.float32)[:3]

    def get_tcp_orientation(self) -> th.Tensor:
        return self._state["pose"].to(device=self.device, dtype=th.float32)[3:]

    def get_tcp_pose(self) -> th.Tensor:
        return self._state["pose"].to(device=self.device, dtype=th.float32)

    def get_wrench(self) -> th.Tensor:
        return self._state["wrench"].to(device=self.device, dtype=th.float32)

    def get_base_position(self) -> th.Tensor:
        return th.zeros(3, dtype=th.float32, device=self.device)

    def control_dofs_position(self, target_pos: th.Tensor) -> None:
        # TODO re-enable servo func
        # if self.robot_director.servo_enabled:
        #     self.robot_director.servo_disable()

        target_pos_np = target_pos.detach().cpu().numpy()
        asyncio.run(
            self._robot_client.goto_joints(
                names=self.joint_names,
                positions=target_pos_np,
            )
        )

    # TODO(issue#35) provide APIs for both servoing
    # implement in grpc_server handling of
    # https://github.com/moveit/moveit2/blob/fec72cb44c71d226254a886c07bdc7a3737daedc/moveit_ros/moveit_servo/src/servo_calcs.cpp#L1043
    def control_dofs_position_servo(
        self, target_pos: Optional[th.Tensor] = None
    ) -> None:
        # TODO enable servo control
        # if not self.robot_director.servo_enabled:
        #     self.robot_director.servo_enable()

        if target_pos is None:
            target_pos_np = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            target_pos_np = target_pos.detach().cpu().numpy()

        asyncio.run(
            self._robot_client.servo_joint(
                joint_names=self.joint_names,
                displacements=target_pos_np,
                velocities=None,
            )
        )

    def control_dofs_velocity_servo(
        self,
        target_vel: Optional[th.Tensor] = None,
    ) -> None:
        # TODO enable servo control
        # if not self.robot_director.servo_enabled:
        #     self.robot_director.servo_enable()

        if target_vel is None:
            target_vel_np = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            target_vel_np = target_vel.detach().cpu().numpy()

        asyncio.run(
            self._robot_client.servo_joint(
                joint_names=self.joint_names,
                displacements=None,
                velocities=target_vel_np,
            )
        )

    def control_tcp_position(
        self,
        target_pos: th.Tensor,
        target_ori: Optional[th.Tensor] = None,
    ) -> None:
        # TODO re-enable servo func
        # if self.robot_director.servo_enabled:
        #     self.robot_director.servo_disable()

        target_pos_np = target_pos.detach().cpu().numpy()
        if target_ori is None:
            target_ori_np = self.default_orientaton
        else:
            target_ori_np = target_ori.detach().cpu().numpy()

        asyncio.run(
            self._robot_client.goto_pose(
                position=target_pos_np,
                orientation=target_ori_np,
            )
        )

    # TODO(issue#33): Change servo API to Euler angles
    def control_tcp_position_servo(
        self,
        target_pos: th.Tensor,
        target_ori_euler: Optional[th.Tensor] = None,
    ) -> None:
        # TODO re-enable servo func
        # if not self.robot_director.servo_enabled:
        #     self.robot_director.servo_enable()

        # TODO(issue#35) provide APIs for both servoing
        # implement in grpc_server handling of
        # https://github.com/moveit/moveit2/blob/fec72cb44c71d226254a886c07bdc7a3737daedc/moveit_ros/moveit_servo/src/servo_calcs.cpp#L1043
        raise NotImplementedError

    # TODO(issue#33): Change servo API to Euler angles
    def control_tcp_velocity_servo(
        self,
        target_pos: th.Tensor,
        target_ori_euler: Optional[th.Tensor] = None,
    ) -> None:
        # TODO re-enable servo func
        # if not self.robot_director.servo_enabled:
        #     self.robot_director.servo_enable()

        target_pos_np = target_pos.detach().cpu().numpy()
        if target_ori_euler is None:
            target_ori_euler_np = self.default_orientaton
        else:
            target_ori_euler_np = target_ori_euler.detach().cpu().numpy()

        asyncio.run(
            self._robot_client.servo_tcp(
                linear=target_pos_np,
                angular=target_ori_euler_np,
            )
        )

    def move_to_home(self) -> None:
        # # TODO(issue#34) There are edge positions where the Moveit2 planner can't plan the trajectory to home position. This issue breaks the training.
        asyncio.run(
            self._robot_client.goto_joints(
                names=tuple(self.dof_home_dict.keys()),
                positions=tuple(self.dof_home_dict.values()),
            )
        )
