import asyncio
import atexit
import threading
from typing import Optional, Any
from concurrent.futures import Future

import torch as th
from tensordict import TensorDict

try:
    from aegis_grpc_client import AegisJointIndex, AegisRobotClient
except ImportError:
    print(
        "Failed to import aegis_grpc_client. "
        "Double check if you have installed the `aegis_grpc_client` and `proto_aegis_grpc` packages."
    )
    raise

from ..base_manipulator import BaseManipulator, CameraID, CameraModality

# from ...scene import BaseScene
from config.types import RobotCfg


class PoseTransformUtils:
    @staticmethod
    def quat_xyzw_to_wxyz(quat: th.Tensor) -> th.Tensor:
        return th.roll(quat, 1, dims=-1)

    @staticmethod
    def quat_wxyz_to_xyzw(quat: th.Tensor) -> th.Tensor:
        return th.roll(quat, -1, dims=-1)


class RosGrpcManipulator(BaseManipulator):
    _instance: Optional["RosGrpcManipulator"] = None

    def __new__(cls, *args, **kwargs) -> "RosGrpcManipulator":
        if cls._instance is None:
            cls._instance = super(RosGrpcManipulator, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        num_envs: int,
        scene,  # TODO(issue#128) introduce proper typing
        robot_cfg: RobotCfg,
        policy_dt: float,
        disable_vision: bool = False,
        device: Optional[th.device] = None,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        super().__init__(device=device)

        self._num_envs = num_envs
        self._scene = scene
        self._cfg_robot = robot_cfg
        self._policy_dt = policy_dt
        self._disable_vision = disable_vision

        self.pt = PoseTransformUtils()

        if num_envs > 1:
            raise ValueError("num_envs > 1 not supported for single robot station")

        def_dofs = robot_cfg.default_arm_dof
        self.dof_home_dict = {
            "shoulder_pan_joint": def_dofs[0],
            "shoulder_lift_joint": def_dofs[1],
            "elbow_joint": def_dofs[2],
            "wrist_1_joint": def_dofs[3],
            "wrist_2_joint": def_dofs[4],
            "wrist_3_joint": def_dofs[5],
        }

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        self._robot_client = AegisRobotClient(server_address="127.0.0.1:50051")
        self._run_coro(self._robot_client.connect())

        self._gripper_last_action = False  # Forcing first opening
        self.ctrl_gripper_open()

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

    def _run_loop(self) -> None:
        """Run the event loop forever in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro) -> Any:
        """
        Schedule a coroutine on the persistent loop and block until done.
        Safe to call from the main (sync) thread.
        """
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # blocks until complete

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

    def read_state(self) -> None:
        states = self._run_coro(self._robot_client.get_all())
        self._state = TensorDict(
            {
                k: th.from_numpy(v)
                .to(device=self.device, dtype=th.float32)
                .unsqueeze(dim=0)
                for k, v in states["state"].items()
            },
            device=self.device,
        )
        # Convert BGR into RGB and np.ndarray into th.Tensor
        if not self._disable_vision:
            self._vision = TensorDict(
                {
                    k: th.from_numpy(v[[2, 1, 0], :, :])
                    .to(self.device)
                    .roll(1, dims=-1)
                    .unsqueeze(dim=0)
                    for k, v in states["vision"].items()
                },
                device=self.device,
            )
        else:
            self._vision = None
        # In Genesis project, every quaterion is assumed to be in WXYZ, where in ROS it is XYZW
        self._state["pose"][3:] = self.pt.quat_xyzw_to_wxyz(self._state["pose"][3:])

    def set_joints_pd_gains(
        self,
        kp_gain: Optional[th.Tensor] = None,
        kv_gain: Optional[th.Tensor] = None,
    ) -> None:
        raise NotImplementedError(
            "Setting PD gains is not supported for the ROS<->gRPC bridge."
        )

    def ctrl_apply_vel_action(
        self,
        action: th.Tensor,
        open_gripper: Optional[bool] = None,
        envs_idx: Optional[th.Tensor] = None,
    ) -> None:
        self._servo_enable()
        # Control only one real robot
        action_target = action.squeeze(dim=0).cpu().numpy()

        self._run_coro(
            self._robot_client.servo_tcp(
                linear=action_target[:3],
                angular=action_target[3:6],
            )
        )

        if open_gripper is None:
            return
        elif open_gripper:
            self.ctrl_gripper_open()
        else:
            self.ctrl_gripper_close()

    def ctrl_apply_joints_diff_action(
        self, joints_diff: th.Tensor, envs_idx: Optional[th.Tensor] = None
    ) -> None:
        raise NotImplementedError

    def ctrl_go_to_goal(
        self,
        goal_pose: th.Tensor,
        open_gripper: Optional[bool] = None,
        envs_idx: Optional[th.Tensor] = None,
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
            self.ctrl_gripper_open()
        else:
            self.ctrl_gripper_close()

    def ctrl_go_to_home(self, envs_idx: Optional[th.Tensor] = None) -> None:
        self._servo_disable()
        self._run_coro(
            self._robot_client.goto_joints(
                names=tuple(self.dof_home_dict.keys()),
                positions=tuple(self.dof_home_dict.values()),
            )
        )

    def ctrl_gripper_open(self, envs_idx: Optional[th.Tensor] = None) -> None:
        if self._gripper_last_action:
            return
        self._run_coro(self._robot_client.gripper_open())
        self._gripper_last_action = True

    def ctrl_gripper_close(self, envs_idx: Optional[th.Tensor] = None) -> None:
        if not self._gripper_last_action:
            return
        self._run_coro(self._robot_client.gripper_close())
        self._gripper_last_action = False

    def get_n_dofs(self) -> int:
        # TODO(issue#111) get it dynamcilly (for instance, from the shape of the joints)
        return 7

    def get_joints_positions(self) -> th.Tensor:
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state["joints"][:, 0]

    def get_joints_velocities(self) -> th.Tensor:
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state["joints"][:, 1]

    def get_joints_efforts(self) -> th.Tensor:
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state["joints"][:, 2]

    def get_ft_wrench(self) -> th.Tensor:
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state["wrench"]

    def get_tcp_pose(self) -> th.Tensor:
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state["pose"]

    def get_base_pose(self) -> th.Tensor:
        return th.tensor([0, 0, 0, 1, 0, 0, 0], dtype=th.float32, device=self.device)

    def get_gripper_width(self) -> th.Tensor:
        idx = AegisJointIndex.ROBOTIQ_HANDE_LEFT_FINGER_JOINT.value
        result = self.get_joints_positions()[idx] * 2
        return result.unsqueeze(dim=0)

    def get_camera_image(
        self, camera_id: CameraID, modality: CameraModality = CameraModality.RGB
    ) -> th.Tensor:
        """
        Returns image tensor for the given camera and modality:
            - RGB:   [num_envs, H, W, 3], dtype uint8
            - DEPTH: [num_envs, H, W, 1], dtype float32, values in meters
        """
        if self._vision is None:
            raise ValueError("Vision disabled.")

        # TODO(issue#125) rewrite the gRPC TensorDict output to match the camera id convention
        cam_name = {
            CameraID.SCENE_CAMERA: "scene",
            CameraID.TOOL_LEFT: "left",
            CameraID.TOOL_RIGHT: "right",
        }[camera_id]
        match modality:
            case CameraModality.RGB:
                return self._vision[cam_name]
            case _:
                raise ValueError(
                    f"Not supported modality: {modality} ({modality.name})."
                )

    def get_all_cameras_images(
        self, modality: CameraModality = CameraModality.RGB
    ) -> TensorDict:
        return self._vision
