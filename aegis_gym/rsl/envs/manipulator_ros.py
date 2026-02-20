import asyncio
import atexit
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


class PoseTransformUtils:
    def __init__(self, device: th.device):
        self.device = device

    def quat_xyzw_to_wxyz(self, quat: th.Tensor) -> th.Tensor:
        return th.roll(quat, 1, dims=-1)

    def quat_wxyz_to_xyzw(self, quat: th.Tensor) -> th.Tensor:
        return th.roll(quat, -1, dims=-1)


class ManipulatorROS:
    _instance: Optional["ManipulatorROS"] = None

    def __new__(cls, *args, **kwargs) -> "ManipulatorROS":
        if cls._instance is None:
            cls._instance = super(ManipulatorROS, cls).__new__(cls)
        return cls._instance

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

        self.pt = PoseTransformUtils(device=device)
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

        self._gripper_last_action = False  # Forcing first opening
        self.gripper_open()

        try:
            self._run_coro(self._robot_client.servo_disable())
        except RuntimeError:
            pass
        self._servo_enabled = False

        # Prepare initial observation
        self._state: Optional[TensorDict] = None
        self._vision: Optional[TensorDict] = None
        self.read_state()

        # shutdown() will be called at interpreter exit
        atexit.register(self.shutdown)
        self._initialized = True
        print("[GraspEnvROS][ManipulatorROS] Finalized initialization")

    def shutdown(self) -> None:
        """
        Explicitly clean up gRPC connection and event loop.
        Should be called before program exit or when done with the robot.
        """
        # Only clean up once
        if hasattr(self, "_cleaned_up") and self._cleaned_up:
            return

        try:
            self._run_coro(self._robot_client.servo_disable())
        except RuntimeError:
            pass

        try:
            # Disconnect gRPC client
            if hasattr(self, "_robot_client") and self._robot_client.is_connected:
                self._run_coro(self._robot_client.disconnect())
        except Exception as e:
            print(f"Error disconnecting robot client: {e}")

        try:
            # Stop the event loop
            if hasattr(self, "_loop") and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

            # Wait for thread to finish
            if hasattr(self, "_loop_thread") and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5.0)
                if self._loop_thread.is_alive():
                    print("Warning: Event loop thread did not stop within timeout")
        except Exception as e:
            print(f"Error stopping event loop: {e}")
        finally:
            # Close the loop
            if hasattr(self, "_loop") and not self._loop.is_closed():
                self._loop.close()

        self._cleaned_up = True

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
        # TOD(issue#62) add option to disable image processing
        states = self._run_coro(self._robot_client.get_all())
        self._state = TensorDict(
            {k: th.from_numpy(v).to(self.device) for k, v in states["state"].items()},
            device=self.device,
        )
        # We assume RGB channels instead of default BGR
        self._vision = TensorDict(
            {
                k: th.from_numpy(v).to(self.device).roll(1, dims=-1)
                for k, v in states["vision"].items()
            },
            device=self.device,
        )
        # In Genesis project, every quaterion is assumed to be in WXYZ, where in ROS it is XYZW
        self._state["pose"][3:] = self.pt.quat_xyzw_to_wxyz(self._state["pose"][3:])

    def get_state_tensordict(self) -> TensorDict:
        return self._state

    def get_vision_tensordict(self) -> TensorDict:
        return self._vision

    def get_joints_positions(self) -> th.Tensor:
        return self._state["joints_pos"].to(device=self.device, dtype=th.float32)

    def get_joints_velocities(self) -> th.Tensor:
        return self._state["joints_vel"].to(device=self.device, dtype=th.float32)

    def get_joints_efforts(self) -> th.Tensor:
        return self._state["joints_eff"].to(device=self.device, dtype=th.float32)

    def get_tcp_position(self) -> th.Tensor:
        return self.get_tcp_pose()[:3]

    def get_tcp_orientation(self) -> th.Tensor:
        return self.get_tcp_pose()[3:]

    def get_tcp_pose(self) -> th.Tensor:
        return self._state["pose"].to(device=self.device, dtype=th.float32)

    def get_wrench(self) -> th.Tensor:
        return self._state["wrench"].to(device=self.device, dtype=th.float32)

    def get_base_position(self) -> th.Tensor:
        return th.zeros(3, dtype=th.float32, device=self.device)

    def get_camera_frame(self, camera_name: str) -> th.Tensor:
        return self._vision[camera_name].unsqueeze(dim=0)

    def get_camera_scene_frame(self) -> th.Tensor:
        return self._vision["scene"].unsqueeze(dim=0)

    def get_camera_tool_right_frame(self) -> th.Tensor:
        return self._vision["right"].unsqueeze(dim=0)

    def get_camera_tool_left_frame(self) -> th.Tensor:
        return self._vision["left"].unsqueeze(dim=0)

    def _servo_enable(self) -> None:
        if self._servo_enabled:
            return
        self._run_coro(self._robot_client.servo_enable())
        self._servo_enabled = True
        print("[GraspEnvROS][ManipulatorROS] Servo enabled")

    def _servo_disable(self) -> None:
        if not self._servo_enabled:
            return
        self._run_coro(self._robot_client.servo_disable())
        self._servo_enabled = False
        print("[GraspEnvROS][ManipulatorROS] Servo disabled")

    def reset(self, envs_idx: th.IntTensor) -> None:
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        print("[GraspEnvROS][ManipulatorROS] Moving to home")
        self.gripper_open()
        self._move_to_home()

    def _move_to_home(self) -> None:
        self._servo_disable()
        self._run_coro(
            self._robot_client.goto_joints(
                names=tuple(self.dof_home_dict.keys()),
                positions=tuple(self.dof_home_dict.values()),
            )
        )

    def apply_action(
        self, action: th.Tensor, open_gripper: Optional[bool] = None
    ) -> None:
        self._servo_enable()
        # Control only one real robot
        action = action.squeeze(dim=0)

        self._run_coro(
            self._robot_client.servo_tcp(
                linear=action[:3],
                angular=action[3:6],
            )
        )

        if open_gripper is None:
            return
        elif open_gripper:
            self.gripper_open()
        else:
            self.gripper_close()

    def apply_dof_rel_action(self, joints_diff: th.Tensor) -> None:
        raise NotImplementedError

    def go_to_goal(
        self, goal_pose: th.Tensor, open_gripper: Optional[bool] = None
    ) -> None:
        self._servo_disable()

        goal_pose = goal_pose.squeeze(dim=0)
        goal_pose[3:] = self.pt.quat_wxyz_to_xyzw(goal_pose[3:])

        target_pos_np = goal_pose[:3].detach().cpu().numpy()
        target_ori_np = goal_pose[3:7].detach().cpu().numpy()

        self._run_coro(
            self._robot_client.goto_pose(
                position=target_pos_np,
                orientation=target_ori_np,
            )
        )

        if open_gripper is None:
            return
        elif open_gripper:
            self.gripper_open()
        else:
            self.gripper_close()

    def gripper_open(self) -> None:
        if self._gripper_last_action:
            return
        self._run_coro(self._robot_client.gripper_open())
        self._gripper_last_action = True

    def gripper_close(self) -> None:
        if not self._gripper_last_action:
            return
        self._run_coro(self._robot_client.gripper_close())
        self._gripper_last_action = False

    @property
    def base_pos(self) -> th.Tensor:
        return self.get_base_position().unsqueeze(dim=0)

    @property
    def ee_pose(self) -> th.Tensor:
        return self.get_tcp_pose().unsqueeze(dim=0)

    @property
    def camera_scene(self) -> th.Tensor:
        return self.get_camera_scene_frame()

    @property
    def camera_tool_right(self) -> th.Tensor:
        return self.get_camera_tool_right_frame()

    @property
    def camera_tool_left(self) -> th.Tensor:
        return self.get_camera_tool_left_frame()
