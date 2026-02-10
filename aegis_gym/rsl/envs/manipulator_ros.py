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

        # Rotation matrix for transforming actions
        self.rotate_z180_mat = th.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=self.device
        )
        # Z-180° quat: [w=0, x=0, y=-1, z=0] for WXYZ
        self.rotate_z180_quat = th.tensor([0.0, 0.0, -1.0, 0.0], device=self.device)

    def transform_to_robot_frame(self, pose: th.tensor) -> th.tensor:
        # Transform position: only x,y affected
        pos = pose[..., :3]
        pos_robot = th.einsum("ij,...j->...i", self.rotate_z180_mat, pos)

        # Rotate quaternion by Z-180°
        quat_robot = self.rotate_z_180(pose[..., 3:])

        return th.cat([pos_robot, quat_robot], dim=-1)

    def transform_to_world_frame(self, pose: th.Tensor) -> th.Tensor:
        # Un-Transform position: apply inverse Z-rotation (transpose for orthogonal matrix)
        pos = pose[..., :3]
        pos_world = th.einsum("ij,...j->...i", self.rotate_z180_mat.T, pos)

        # Undo: multiply by inverse (Z+180° = Z-180° since 180°=-180°)
        quat_world = self.rotate_z_180(pose[..., 3:])

        return th.cat([pos_world, quat_world], dim=-1)

    def quat_xyzw_to_wxyz(self, quat: th.Tensor) -> th.Tensor:
        return th.roll(quat, -1, dims=-1)

    def quat_wxyz_to_xyzw(self, quat: th.Tensor) -> th.Tensor:
        return th.roll(quat, 1, dims=-1)

    def rotate_z_180(self, quat: th.Tensor) -> th.Tensor:
        # Broadcast to batch dims
        rot_z180 = self.rotate_z180_quat.to(quat.device).expand_as(quat)

        # Quaternion multiply: quat @ rot_z180 (WXYZ matrix order)
        w1, x1, y1, z1 = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        w2, x2, y2, z2 = (
            rot_z180[..., 0],
            rot_z180[..., 1],
            rot_z180[..., 2],
            rot_z180[..., 3],
        )

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return th.stack([w, x, y, z], dim=-1)


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
        policy_dt: float,
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
        self.ctrl_dt = policy_dt  # 1 / poliicy_f

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

        try:
            self._run_coro(self._robot_client.servo_disable())
        except RuntimeError:
            pass
        self._servo_enabled = False

        # Prepare initial observation
        self._state: Optional[TensorDict] = None
        self.read_state()

        self.pt = PoseTransformUtils(device=device)

        # TODO read it from the config
        self.max_lin_speed = 0.0098  # m/s
        self.max_ang_speed = 0.302  # rad/s

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
        # TODO enable state and vision
        # state = self._run_coro(self._robot_client.get_all())["state"]
        state = self._run_coro(self._robot_client.get_robot_state())
        self._state = TensorDict(
            {k: th.from_numpy(v).to(self.device) for k, v in state.items()},
            device=self.device,
        )
        # In Genesis project, every quaterion is assumed to be in WXYZ, where in ROS it is XYZW
        self._state["pose"][3:] = self.pt.quat_xyzw_to_wxyz(self._state["pose"][3:])

    def get_state_tensordict(self) -> TensorDict:
        return self._state

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
        robot_pose = self._state["pose"].to(device=self.device, dtype=th.float32)
        return self.pt.transform_to_world_frame(robot_pose)

    def get_wrench(self) -> th.Tensor:
        return self._state["wrench"].to(device=self.device, dtype=th.float32)

    def get_base_position(self) -> th.Tensor:
        return th.zeros(3, dtype=th.float32, device=self.device)

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
        """
        The action is assumed to be a EE velocity [m/s] and euler angular speed [rad/s]
        i.e. v_x, v_y, v_z, w_x, w_y, w_z
        """
        self._servo_enable()
        # Control only one real robot
        action = action.squeeze(dim=0)

        # Since the action is in the EE frame, the rotation should not be needed
        # action = self._transform_to_rotated_base(action).squeeze(dim=0)

        print(f"[GraspEnvROS][ManipulatorROS] Action before scaling: {action}")

        # Scaling the action to the unitless servo input
        action[:3] /= self.max_lin_speed
        action[3:6] /= self.max_ang_speed
        action = th.clamp(action, min=-1.0, max=1.0)

        print(f"[GraspEnvROS][ManipulatorROS] Action after scaling: {action}")

        self._run_coro(
            self._robot_client.servo_tcp(
                linear=action[:3],
                angular=action[3:6],
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
        """
        The goal is assumed to be a pose with position and WXYZ quaterion in the world frame of reference.
        """
        self._servo_disable()

        # Prepare data for ROS control, i.e. in robot frame: position + quat_xyzw
        print(f">> DEBUG, goal_pose before transform {goal_pose}")

        goal_pose = goal_pose.squeeze(dim=0)
        goal_pose = self.pt.transform_to_robot_frame(goal_pose)
        goal_pose[3:] = self.pt.quat_wxyz_to_xyzw(goal_pose[3:])

        target_pos_np = goal_pose[:3].detach().cpu().numpy()
        target_ori_np = goal_pose[3:7].detach().cpu().numpy()

        print(f">> DEBUG, target_pos_np {target_pos_np}")
        print(f">> DEBUG, target_ori_np {target_ori_np}")

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
        return self.get_tcp_pose().unsqueeze(dim=0)
