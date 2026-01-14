from typing import Optional
import torch as th

try:
    from aegis_director.robot_director import RobotDirector
    from aegis_director.utils import quaternion_to_euler
except ImportError:
    print(
        "Failed to import aegis_director. Double check if you have sourced the AGH-CEAI/aegis_ros project."
    )
    raise ImportError

try:
    from aegis_grpc_client import AegisRobotClient
except ImportError:
    print(
        "Failed to import aegis_grpc_client. "
        "Double check if you have installed the `aegis_grpc_client` and `proto_aegis_grpc` packages."
    )
    raise ImportError

from ..scene import RobotCommanderInterface


class RobotCommanderROS(RobotCommanderInterface):
    _instance: Optional["RobotCommanderROS"] = None

    def __new__(cls, *args, **kwargs) -> "RobotCommanderROS":
        if cls._instance is None:
            cls._instance = super(RobotCommanderROS, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, robot_director: RobotDirector, robot_client: AegisRobotClient, device: str
    ) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        super().__init__(device=device)
        self.robot_director = robot_director
        self.robot_client = robot_client

        # TODO rewrite all getters to use robot_client
        joint_state = self.robot_director._get_joint_states()
        self.joint_names = list(joint_state.name)[1:]
        self._initialized = True

    def get_joint_positions(self) -> th.Tensor:
        jp = self.robot_director.get_joint_positions()
        return th.tensor(
            [jp[name] for name in self.joint_names],
            dtype=th.float32,
            device=self.device,
        )

    def get_joint_velocities(self) -> th.Tensor:
        jv = self.robot_director.get_joint_velocities()
        return th.tensor(
            [jv[name] for name in self.joint_names],
            dtype=th.float32,
            device=self.device,
        )

    def get_tcp_position(self) -> th.Tensor:
        return self.get_tcp_pose()[:3]

    def get_tcp_orientation(self) -> th.Tensor:
        return self.get_tcp_pose()[3:]

    def get_tcp_pose(self) -> th.Tensor:
        tcp = self.robot_director.get_tcp_pose()
        return th.cat(
            [
                th.tensor(tcp["position"], dtype=th.float32, device=self.device),
                th.tensor(tcp["orientation"], dtype=th.float32, device=self.device),
            ],
        )

    def get_base_position(self) -> th.Tensor:
        return th.zeros(3, dtype=th.float32, device=self.device)

    def control_dofs_position(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        if self.robot_director.servo_enabled:
            self.robot_director.servo_disable()

        target_pos_np = target_pos.detach().cpu().numpy()
        target_pos_dict = {
            name: float(pos) for name, pos in zip(self.joint_names, target_pos_np)
        }
        self.robot_director.joint_move(
            joint_positions=target_pos_dict, max_vel=max_vel, max_accel=max_accel
        )

    def control_dofs_position_servo(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        # TODO(issue#35) - Unify servoing into position or velocities commands
        self.control_dofs_velocity_servo(target_vel=target_pos)

    def control_dofs_velocity_servo(
        self,
        target_vel: th.Tensor | None = None,
    ) -> None:
        if not self.robot_director.servo_enabled:
            self.robot_director.servo_enable()

        if target_vel is None:
            target_vel_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            target_vel_tuple = tuple(target_vel.detach().cpu().numpy())

        self.robot_director.servo_jog(
            joint_names=tuple(self.joint_names),
            velocities=target_vel_tuple,
        )

    def control_tcp_position(
        self,
        target_pos: th.Tensor,
        target_ori: th.Tensor | None = None,
        max_vel: float = 0.3,
        max_accel: float = 0.3,
    ) -> None:
        if self.robot_director.servo_enabled:
            self.robot_director.servo_disable()

        if target_ori is None:
            target_ori = th.tensor(
                [1.0, 0.0, 0.0, 0.0],
                dtype=th.float32,
                device=self.device,
            )

        target_pos_np = target_pos.detach().cpu().numpy()
        target_ori_np = target_ori.detach().cpu().numpy()

        target_pos_tuple = tuple(float(v) for v in target_pos_np)
        target_ori_tuple = tuple(float(v) for v in target_ori_np)

        self.robot_director.pose_move(
            position=target_pos_tuple,
            quat_xyzw=target_ori_tuple,
            max_vel=max_vel,
            max_accel=max_accel,
        )

    # TODO(issue#33): Change servo API to Euler angles
    def control_tcp_position_servo(
        self,
        target_pos: th.Tensor,
        target_ori: th.Tensor | None = None,  # HACK interface mismatch for default None
        max_vel: float = 0.3,
        max_accel: float = 0.3,
    ) -> None:
        # TODO(issue#35) - Unify servoing into position or velocities commands
        self.control_tcp_velocity_servo(target_pos, target_ori)

    # TODO(issue#33): Change servo API to Euler angles
    def control_tcp_velocity_servo(
        self,
        target_pos: th.Tensor,
        target_ori: th.Tensor | None = None,
    ) -> None:
        if not self.robot_director.servo_enabled:
            self.robot_director.servo_enable()

        if target_ori is None:
            target_ori_tuple = tuple([0.0, 0.0, 0.0])
        else:
            target_ori_np = target_ori.detach().cpu().numpy()
            target_ori_tuple = tuple(quaternion_to_euler(q_xyzw=target_ori_np))

        target_pos_np = target_pos.detach().cpu().numpy()
        target_pos_tuple = tuple(float(v) for v in target_pos_np)

        self.robot_director.servo_move(
            linear=target_pos_tuple,
            angular=target_ori_tuple,
        )

    def move_to_home(self) -> None:
        if self.robot_director.servo_enabled:
            self.robot_director.servo_disable()
        # TODO(issue#34) There are edge positions where the Moveit2 planner can't plan the trajectory to home position. This issue breaks the training.
        self.robot_director.joint_move(
            joint_positions=self.dof_home_dict,
            max_vel=0.5,
            max_accel=0.5,
        )
