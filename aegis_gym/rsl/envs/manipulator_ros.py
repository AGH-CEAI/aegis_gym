import asyncio
import threading
from typing import Optional
from concurrent.futures import Future

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
        if hasattr(self, "_robot_client") and self._robot_client.is_connected:
            self._run_coro(self._robot_client.disconnect())
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2.0)

    def __init__(
        self,
        num_envs: int,
        args: dict,
        device: th.device = th.device("cpu"),
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        if num_envs > 1:
            raise ValueError("num_envs > 1 not supported for single robot station")

        self._num_envs = num_envs
        self._args = args
        self.device = device

        # TODO get from cfg / dynamically from rsl_rl
        self.ctrl_dt = 1 / 10.0  # 1 / poliicy_f

        self.dof_home_dict = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -2.09,
            "elbow_joint": 2.09,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
        }

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        self._robot_client = AegisRobotClient(server_address="127.0.0.1:50051")
        self._run_coro(self._robot_client.connect())

        self._run_coro(self._robot_client.gripper_open())
        self._gripper_last_action = True

        # Prepare initial observation
        self._state: Optional[TensorDict] = None
        self.read_state()

    def _run_loop(self) -> None:
        """Run the event loop forever in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro) -> any:
        """
        Schedule a coroutine on the persistent loop and block until done.
        Safe to call from the main (sync) thread.
        """
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # blocks until complete

    def read_state(self) -> None:
        state = self._run_coro(self._robot_client.get_all())
        self._state = TensorDict(
            {k: th.from_numpy(v).to(self.device) for k, v in state.items()},
            device=self.device,
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
        self._run_coro(
            self._robot_client.goto_joints(
                names=tuple(self.dof_home_dict.keys()),
                positions=tuple(self.dof_home_dict.values()),
            )
        )

    def apply_action(
        self, action: th.Tensor, open_gripper: Optional[bool] = None
    ) -> None:
        # Control only one real robot
        action = action.squeeze(dim=0)

        # Change position commands into velocities for the MoveIt2 Servo
        delta_velocity = action[:3] / self.ctrl_dt
        delta_angular = action[3:6] / self.ctrl_dt

        self._run_coro(
            self._robot_client.servo_tcp(
                linear=delta_velocity,
                angular=delta_angular,
            )
        )

        if open_gripper is None:
            return

        if open_gripper and not self._gripper_last_action:
            self._run_coro(self._robot_client.gripper_open())
        if not open_gripper and self._gripper_last_action:
            self._run_coro(self._robot_client.gripper_close())
        self._gripper_last_action = open_gripper

    def go_to_goal(
        self, goal_pose: th.Tensor, open_gripper: Optional[bool] = None
    ) -> None:
        target_pos_np = goal_pose[:, :3].detach().cpu().numpy()
        target_ori_np = goal_pose[:, 3:7].detach().cpu().numpy()

        self._run_coro(
            self._robot_client.goto_pose(
                position=target_pos_np,
                orientation=target_ori_np,
            )
        )

        if open_gripper is None:
            return

        if open_gripper and not self._gripper_last_action:
            self._run_coro(self._robot_client.gripper_open())
        if not open_gripper and self._gripper_last_action:
            self._run_coro(self._robot_client.gripper_close())
        self._gripper_last_action = open_gripper

    @property
    def base_pos(self) -> th.Tensor:
        return self.get_base_position().unsqueeze(dim=0)

    @property
    def ee_pose(self) -> th.Tensor:
        # TODO validate if its the actual gripper's EE
        return self.get_tcp_pose().unsqueeze(dim=0)
