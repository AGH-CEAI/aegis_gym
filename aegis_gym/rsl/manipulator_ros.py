import asyncio
from typing import Literal, Optional

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


class ManipulatorROS:
    _instance: Optional["ManipulatorROS"] = None
    
    def __new__(cls, *args, **kwargs) -> "ManipulatorROS":
        if cls._instance is None:
            cls._instance = super(ManipulatorROS, cls).__new__(cls)
        return cls._instance
    
    def __del__(self) -> None:
        if self.robot_client.is_connected:
            asyncio.run(self.robot_client.disconnect())
    
    def __init__(
        self,
        num_envs: int,
        args: dict,
        device: str = "cpu",
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self.num_envs = num_envs
        if self.num_envs > 1:
            raise ValueError(
                "The `num_envs` is greater than 1 for controlling just 1 robot station!!!"
            )
        
        self._num_envs = num_envs
        self._args = args
        self._device = device

        # TODO get from cfg
        self.ctrl_dt = 1 / 250.0 # 1/servo_f
        
        self.dof_home_dict = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -2.09,
            "elbow_joint": 2.09,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            # "robotiq_hande_left_finger_joint": 0.025,
            # "robotiq_hande_right_finger_joint": 0.025,
        }
                
        self.robot_client = AegisRobotClient(server_address="127.0.0.1:50051")
        asyncio.run(self.robot_client.connect())
        
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

    def reset(self, envs_idx: th.IntTensor) -> None:
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        self._move_to_home()
        
    def _move_to_home(self) -> None:
        asyncio.run(
            self._robot_client.goto_joints(
                names=tuple(self.dof_home_dict.keys()),
                positions=tuple(self.dof_home_dict.values()),
            )
        )

    def apply_action(self, action: th.Tensor, open_gripper: Optional[bool] = None) -> None:
        
        # Change position commands into velocities for the MoveIt2 Servo
        delta_velocity = action[:, :3] / self.ctrl_dt
        delta_angular = action[:, 3:6] / self.ctrl_dt
        
        asyncio.run(
            self._robot_client.servo_tcp(
                linear=delta_velocity,
                angular=delta_angular,
            )
        )
        
        if open_gripper is None:
            return
        
        if open_gripper:
            asyncio.run(self._robot_client.gripper_open())
        else:
            asyncio.run(self._robot_client.gripper_close())

    def go_to_goal(self, goal_pose: th.Tensor, open_gripper: Optional[bool] = None) -> None:
        target_pos_np = goal_pose[:, :3].detach().cpu().numpy()
        target_ori_np = goal_pose[:, 3:7].detach().cpu().numpy()

        asyncio.run(
            self._robot_client.goto_pose(
                position=target_pos_np,
                orientation=target_ori_np,
            )
        )
        
        if open_gripper is None:
            return
        
        if open_gripper:
            asyncio.run(self._robot_client.gripper_open())
        else:
            asyncio.run(self._robot_client.gripper_close())

    @property
    def base_pos(self) -> th.Tensor:
        return self.get_base_position()

    @property
    def ee_pose(self) -> th.Tensor:
        # TODO validate if its the actual gripper's EE
        return self.get_tcp_pose()
