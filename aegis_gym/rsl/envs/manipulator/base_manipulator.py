from abc import ABC, abstractmethod
from enum import auto
from strenum import StrEnum
from typing import Optional

import torch as th
from tensordict import TensorDict


class CameraID(StrEnum):
    SCENE_CAMERA = auto()
    TOOL_LEFT = auto()
    TOOL_RIGHT = auto()

    @classmethod
    def from_str(cls, x: str) -> "CameraID":
        if "left" in x.lower():
            return CameraID.TOOL_LEFT
        if "right" in x.lower():
            return CameraID.TOOL_RIGHT
        if "scene" in x.lower():
            return CameraID.SCENE_CAMERA
        raise ValueError(f"Could not resolve the CameraID from given string: {x}")


class CameraModality(StrEnum):
    RGB = auto()
    RGBD = auto()
    DEPTH = auto()


class BaseManipulator(ABC):
    """
    Interface for interacting with the robotic arm, both in real and simulated worlds.
    Note:
        The `envs_idx` parameter is meaningful only in simulation (multi-environment).
        Real robot implementations should accept it for interface compatibility but may
        ignore it or assert it is None.
    """

    @abstractmethod
    def shutdown(self) -> None:
        """
        Terminate the connection with the Manipulator(s).
        """
        ...

    @abstractmethod
    def read_state(self) -> None:
        """
        Update the internal state.
        This should be called after each step, before collecting a new observation for the policy.
        Without the call, the data provided by the getters will be old.
        """
        ...

    @abstractmethod
    def set_joints_pd_gains(
        self,
        kp_gain: Optional[th.FloatTensor] = None,
        kv_gain: Optional[th.FloatTensor] = None,
    ) -> None:
        """Simulation only: Sets the gains from [0..1]. Accepts sizes [n_dofs]."""
        ...

    @abstractmethod
    def ctrl_apply_vel_action(
        self,
        action: th.Tensor,
        open_gripper: Optional[bool] = None,
        envs_idx: Optional[th.IntTensor] = None,
    ) -> None:
        """
        Apply the action (velocity servoing) to the robot.

        Args:
            action: [num_envs, 6] tensor containing target end-effector velocities
                    [vx, vy, vz, wx, wy, wz] where v is linear and w is angular velocity
            open_gripper: Optional bool to control gripper state
        """
        ...

    @abstractmethod
    def ctrl_apply_joints_diff_action(
        self, joints_diff: th.Tensor, envs_idx: Optional[th.IntTensor] = None
    ) -> None:
        """
        Apply the action (joints difference) to the robot.

        Args:
            joints_diff: [num_envs, n_dofs] tensor containing relative joints positions w.r.t. current joints positions.
        """
        ...

    @abstractmethod
    def ctrl_go_to_goal(
        self,
        goal_pose: th.Tensor,
        open_gripper: Optional[bool] = None,
        envs_idx: Optional[th.IntTensor] = None,
    ) -> None:
        """
        Apply the goal_pose (position target) to the robot.

        Args:
            goal_pose:  [num_envs, 7] tensor containing target end-effector pose
                        [x, y, z, qw, qx, qy, qz] where x,y,z represent position and q prefix represent orientation quaterion.
            open_gripper: Optional bool to control gripper state
        """
        ...

    @abstractmethod
    def ctrl_go_to_home(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        """Move to the home joint configuration."""
        ...

    @abstractmethod
    def ctrl_gripper_open(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        """Open the gripper to its maximum width."""
        ...

    @abstractmethod
    def ctrl_gripper_close(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        """Close the gripper to its minimum width."""
        ...

    @abstractmethod
    def get_n_dofs(self) -> int:
        """Returns the number of controllable degrees of freedom."""
        ...

    @abstractmethod
    def get_joints_positions(self) -> th.Tensor:
        """Returns [num_envs, n_dofs] joint positions in radians."""
        ...

    @abstractmethod
    def get_joints_velocities(self) -> th.Tensor:
        """Returns [num_envs, n_dofs] joint velocities in rad/s."""
        ...

    @abstractmethod
    def get_joints_efforts(self) -> th.Tensor:
        """Returns [num_envs, n_dofs] joint efforts in Nm."""
        ...

    @abstractmethod
    def get_ft_wrench(self) -> th.Tensor:
        """Returns [num_envs, 6] wrench in the F/T sensor link as [fx, fy, fz, tx, ty, tz] in Newtons and Newton-metre."""
        ...

    @abstractmethod
    def get_tcp_pose(self) -> th.Tensor:
        """
        Get TCP pose as [n_envs, 7], where 7 values: [x, y, z, qw, qx, qy, qz].
        """
        ...

    def get_tcp_position(self) -> th.Tensor:
        return self.get_tcp_pose()[:, :3]

    def get_tcp_orientation(self) -> th.Tensor:
        return self.get_tcp_pose()[:, 3:]

    @abstractmethod
    def get_base_pose(self) -> th.Tensor:
        """Returns [num_envs, 7] robot base pose as [x, y, z, qw, qx, qy, qz]."""
        ...

    @abstractmethod
    def get_gripper_width(self) -> th.Tensor:
        """Returns [num_envs, 1] gripper width in meters."""
        ...

    @abstractmethod
    def get_camera_image(
        self, camera_id: CameraID, modality: CameraModality = CameraModality.RGB
    ) -> th.Tensor:
        """
        Returns image tensor for the given camera and modality:
            - RGB:   [num_envs, H, W, 3], dtype uint8
            - RGBD:  [num_envs, H, W, 4], dtype float32, depth in meters
            - DEPTH: [num_envs, H, W, 1], dtype float32, values in meters
        """
        ...

    @abstractmethod
    def get_all_cameras_images(
        self, modality: CameraModality = CameraModality.RGB
    ) -> TensorDict:
        """Returns TensorDict (with `CameraID` keys) of all images (with specified `modality`)."""
        ...
