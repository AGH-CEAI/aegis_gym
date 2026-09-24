import asyncio
import atexit
import threading
from concurrent.futures import Future
from typing import Any, Optional

import torch as th
from tensordict import TensorDict

from aegis_gym.aux.logging import get_logger

logger = get_logger(__name__)

try:
    from aegis_grpc_client import (
        AegisJointIndex,
        AegisJointName,
        AegisRobotClient,
        ModalityGroup,
        StateModality,
    )
    from aegis_grpc_client import CameraName as RosGrpcCameraName
except ImportError:
    logger.error(
        "Failed to import aegis_grpc_client. "
        "Double check if you have installed the `aegis_grpc_client` and `proto_aegis_grpc` packages."
    )
    raise

from typing_extensions import Self

from aegis_gym.config.types import CameraName, RobotCfg

from ..base_manipulator import BaseManipulator, CameraModality

# Kinematic order, which is what this project uses everywhere for arm joints
# (`RobotCfg.default_arm_dof`, the Genesis DOF order). Neither `AegisJointName` nor the
# /joint_states topic is in this order, so anything crossing the bridge is paired by
# name rather than by position.
_ARM_JOINT_NAMES = (
    AegisJointName.SHOULDER_PAN_JOINT,
    AegisJointName.SHOULDER_LIFT_JOINT,
    AegisJointName.ELBOW_JOINT,
    AegisJointName.WRIST_1_JOINT,
    AegisJointName.WRIST_2_JOINT,
    AegisJointName.WRIST_3_JOINT,
)


class PoseTransformUtils:
    @staticmethod
    def quat_xyzw_to_wxyz(quat: th.Tensor) -> th.Tensor:
        return th.roll(quat, 1, dims=-1)

    @staticmethod
    def quat_wxyz_to_xyzw(quat: th.Tensor) -> th.Tensor:
        return th.roll(quat, -1, dims=-1)


class RosGrpcManipulator(BaseManipulator):
    _instance: Optional["RosGrpcManipulator"] = None

    def __new__(cls, *args, **kwargs) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        num_envs: int,
        robot_cfg: RobotCfg,
        policy_dt: float,
        disable_cameras: bool = False,
        device: th.device | None = None,
        server_address: str = "127.0.0.1:50051",
    ):
        self.logger = get_logger("GraspEnvROS/ManipulatorROS")
        if hasattr(self, "_initialized") and self._initialized:
            return

        super().__init__(device=device)

        self._num_envs = num_envs
        self._cfg_robot = robot_cfg
        self._policy_dt = policy_dt
        self._disable_cameras = disable_cameras

        self.pt = PoseTransformUtils()

        self._cam_map = {
            CameraName.CAMERA_SCENE: RosGrpcCameraName.CAMERA_SCENE,
            CameraName.CAMERA_TOOL_LEFT: RosGrpcCameraName.CAMERA_TOOL_LEFT,
            CameraName.CAMERA_TOOL_RIGHT: RosGrpcCameraName.CAMERA_TOOL_RIGHT,
        }

        def_dofs = robot_cfg.default_arm_dof
        self.dof_home_dict = {
            AegisJointName.SHOULDER_PAN_JOINT: def_dofs[0],
            AegisJointName.SHOULDER_LIFT_JOINT: def_dofs[1],
            AegisJointName.ELBOW_JOINT: def_dofs[2],
            AegisJointName.WRIST_1_JOINT: def_dofs[3],
            AegisJointName.WRIST_2_JOINT: def_dofs[4],
            AegisJointName.WRIST_3_JOINT: def_dofs[5],
        }

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        self._robot_client = AegisRobotClient(server_address=server_address)
        self._run_coro(self._robot_client.connect())

        self._ft_bias_active = False
        self.set_ft_payload(robot_cfg.fts_payload_mass, robot_cfg.fts_payload_com)

        self._gripper_last_action = False  # Forcing first opening
        self.ctrl_gripper_open()

        try:
            self._run_coro(self._robot_client.servo_disable())
        except RuntimeError:
            pass
        self._servo_enabled = False

        # Prepare initial observation
        self._state: TensorDict | None = None
        self._vision: TensorDict | None = None
        self.read_state()

        # shutdown() will be called at interpreter exit
        atexit.register(self.shutdown)
        self._initialized = True
        self.logger.info("Finalized initialization")

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
        self.logger.info("Servo enabled")

    def _servo_disable(self) -> None:
        if not self._servo_enabled:
            return
        self._run_coro(self._robot_client.servo_disable())
        self._servo_enabled = False
        self.logger.info("Servo disabled")

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
        except RuntimeError as e:
            self.logger.error(f"Error disconnecting robot client: {e}")

        try:
            # Stop the event loop
            if hasattr(self, "_loop") and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

            # Wait for thread to finish
            if hasattr(self, "_loop_thread") and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5.0)
                if self._loop_thread.is_alive():
                    self.logger.warning("Event loop thread did not stop within timeout")
        except RuntimeError as e:
            self.logger.error(f"Error stopping event loop: {e}")
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
                for k, v in states[ModalityGroup.STATE].items()
            },
            device=self.device,
        )
        # Convert BGR into RGB and np.ndarray into th.Tensor
        if not self._disable_cameras:
            self._vision = TensorDict(
                {
                    k: th.from_numpy(v[[2, 1, 0], :, :])
                    .to(self.device)
                    .roll(1, dims=-1)
                    .unsqueeze(dim=0)
                    for k, v in states[ModalityGroup.VISION].items()
                },
                device=self.device,
            )
        else:
            self._vision = None
        # In Genesis project, every quaterion is assumed to be in WXYZ, where in ROS it is XYZW.
        # Indexed on the last axis: the state is [1, 7] after the unsqueeze above, so a bare
        # `[3:]` slices the batch dimension instead, selects nothing, and silently leaves the
        # quaternion in XYZW -- which then reads as a 180 deg rotation about X.
        self._state[StateModality.POSE][..., 3:] = self.pt.quat_xyzw_to_wxyz(
            self._state[StateModality.POSE][..., 3:]
        )

    def set_joints_pd_gains(
        self,
        kp_gain: th.Tensor | None = None,
        kv_gain: th.Tensor | None = None,
    ) -> None:
        raise NotImplementedError(
            "Setting PD gains is not supported for the ROS<->gRPC bridge."
        )

    def ctrl_apply_vel_action(
        self,
        action: th.Tensor,
        open_gripper: bool | None = None,
        envs_idx: th.Tensor | None = None,
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
        self, joints_diff: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        raise NotImplementedError

    def ctrl_go_to_goal(
        self,
        goal_pose: th.Tensor,
        open_gripper: bool | None = None,
        envs_idx: th.Tensor | None = None,
    ) -> None:
        self._servo_disable()

        # `squeeze` returns a view, so the conversion below would otherwise rewrite the
        # caller's own tensor in place and leave it holding an XYZW quaternion.
        goal_pose = goal_pose.squeeze(dim=0).clone()
        goal_pose[3:] = self.pt.quat_wxyz_to_xyzw(goal_pose[3:])

        target_pos_np = goal_pose[:3].detach().cpu().numpy()
        target_ori_np = goal_pose[3:7].detach().cpu().numpy()

        success, msg = self._run_coro(
            self._robot_client.goto_pose(
                position=target_pos_np,
                orientation=target_ori_np,
            )
        )
        # A rejected plan used to pass silently, leaving the arm where it was while the
        # caller carried on as if it had moved.
        if not success:
            raise RuntimeError(
                f"The planner did not accept the TCP goal {goal_pose.tolist()}: {msg}"
            )

        if open_gripper is None:
            return
        elif open_gripper:
            self.ctrl_gripper_open()
        else:
            self.ctrl_gripper_close()

    def ctrl_go_to_home(self, envs_idx: th.Tensor | None = None) -> None:
        self._servo_disable()
        self._run_coro(
            self._robot_client.goto_joints(
                names=tuple(self.dof_home_dict.keys()),
                positions=tuple(self.dof_home_dict.values()),
            )
        )

    def ctrl_go_to_joints(
        self, joints: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        joints = joints.reshape(-1)
        if joints.numel() != len(_ARM_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(_ARM_JOINT_NAMES)} arm joint positions, "
                f"got {joints.numel()}"
            )

        self._servo_disable()
        # Paired by name rather than by position: the bridge's joint enum is not in
        # kinematic order, and neither is the /joint_states topic.
        success, msg = self._run_coro(
            self._robot_client.goto_joints(
                names=_ARM_JOINT_NAMES,
                positions=tuple(joints.detach().cpu().tolist()),
            )
        )
        if not success:
            raise RuntimeError(
                f"The planner did not accept the joint goal {joints.tolist()}: {msg}"
            )

    def ctrl_gripper_open(self, envs_idx: th.Tensor | None = None) -> None:
        if self._gripper_last_action:
            return
        self._run_coro(self._robot_client.gripper_open())
        self._gripper_last_action = True

    def ctrl_gripper_close(self, envs_idx: th.Tensor | None = None) -> None:
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
        return self._state[StateModality.JOINTS][:, :, 0]

    def get_joints_velocities(self) -> th.Tensor:
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state[StateModality.JOINTS][:, :, 1]

    def get_joints_efforts(self) -> th.Tensor:
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state[StateModality.JOINTS][:, :, 2]

    def get_ft_wrench(self) -> th.Tensor:
        """The F/T sensor measurement as the bridge reports it.

        Nothing is computed here: the robot owns the sensor, applies its own tare when
        one is set, and this only hands the value on. `read_state()` refreshes it.
        """
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state[StateModality.WRENCH]

    def set_ft_bias(self) -> None:
        """Tares the sensor in hardware, through the bridge (`wrench_bias_set`)."""
        success, msg = self._run_coro(self._robot_client.wrench_bias_set())
        if not success:
            raise RuntimeError(f"Failed to set the F/T sensor bias: {msg}")
        self._ft_bias_active = True
        self.read_state()  # so the next getter sees the tared measurement
        self.note_ft_bias_pose()
        self.logger.info("F/T sensor tared in hardware")

    def clear_ft_bias(self) -> None:
        """Clears the tare in hardware, through the bridge (`wrench_bias_clear`)."""
        success, msg = self._run_coro(self._robot_client.wrench_bias_clear())
        if not success:
            raise RuntimeError(f"Failed to clear the F/T sensor bias: {msg}")
        self._ft_bias_active = False
        self._ft_gravity_at_bias = None
        self.read_state()
        self.logger.info("F/T sensor bias cleared in hardware")

    def is_ft_biased(self) -> bool:
        return self._ft_bias_active

    def get_tcp_pose(self) -> th.Tensor:
        if self._state is None:
            raise ValueError("Call read_state() to initialize values")
        return self._state[StateModality.POSE]

    def get_base_pose(self) -> th.Tensor:
        return th.tensor([0, 0, 0, 1, 0, 0, 0], dtype=th.float32, device=self.device)

    def get_gripper_width(self) -> th.Tensor:
        idx = AegisJointIndex.ROBOTIQ_HANDE_LEFT_FINGER_JOINT.value
        result = self.get_joints_positions()[idx] * 2
        return result.unsqueeze(dim=0)

    def get_camera_image(
        self, camera: CameraName, modality: CameraModality = CameraModality.RGB
    ) -> th.Tensor:
        """
        Returns image tensor for the given camera and modality:
            - RGB:   [num_envs, H, W, 3], dtype uint8
            - DEPTH: [num_envs, H, W, 1], dtype float32, values in meters
        """
        if self._vision is None:
            raise ValueError("Vision disabled.")

        match modality:
            case CameraModality.RGB:
                return self._vision[self._cam_map[camera]]
            case _:
                raise ValueError(
                    f"Not supported modality: {modality} ({modality.name})."
                )

    def get_all_cameras_images(
        self, modality: CameraModality = CameraModality.RGB
    ) -> TensorDict:
        return self._vision
