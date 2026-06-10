import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import torch as th
import genesis as gs
from clearml import Dataset
from tensordict import TensorDict

from ..base_manipulator import BaseManipulator, CameraID, CameraModality


class GenesisManipulator(BaseManipulator):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: dict,
        show_cell: bool,
        device: Optional[th.device] = None,
    ):
        # == set members ==
        self._device = device or th.device("cpu")
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # TODO(issue#99): Implement URDF model with cell collision handling
        if show_cell:
            self._urdf_model_id = args["urdf_model_id"]["cell"]
        else:
            self._urdf_model_id = args["urdf_model_id"]["no_cell"]

        if self._urdf_model_id:
            print(
                f"[GraspEnv::Manipulator] URDF ClearML dataset ID: {self._urdf_model_id}"
            )
        self._urdf_path = self._resolve_aegis_urdf()
        print(f"[GraspEnv::Manipulator] URDF path: {self._urdf_path}")

        # == Genesis configurations ==
        material = gs.materials.Rigid(gravity_compensation=1.0)
        morph = gs.morphs.URDF(
            file=self._urdf_path,
            fixed=True,
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            links_to_keep=[
                "ur_base",
                "robotiq_hande_end",
                "cam_tool_right",
                "cam_tool_left",
                "cam_scene_rgb_camera_frame",
            ],
        )
        self._robot_entity = scene.add_entity(material=material, morph=morph)

        self._gripper_open_dof = 0.025
        self._gripper_close_dof = 0.0

        self._ik_method: Literal["gs_ikv", "dls_ikv"] = args["ik_method"]

        self._setup_config()
        self._init_pd_tensors()

    def _resolve_aegis_urdf(self) -> Path:
        default_path = Path("~/ceai_ws/aegis_urdf/aegis.urdf").expanduser().resolve()

        if self._urdf_model_id is not None:
            try:
                dataset = Dataset.get(
                    dataset_id=self._urdf_model_id, alias="urdf_model"
                )
                local_path = Path(dataset.get_local_copy())
            except ValueError:
                warnings.warn(
                    "Failed to obtain the dataset: `{e}`. Fallbacking to the default path..."
                )
                return default_path

            urdf_files = list(local_path.rglob("*.urdf"))
            if not urdf_files:
                raise FileNotFoundError(
                    f"No URDF file in dataset {self._urdf_model_id}"
                )
            if len(urdf_files) > 1:
                raise RuntimeError(
                    f"Found {len(urdf_files)} URDF files in dataset {self._urdf_model_id}, expected just one"
                )
            return Path(urdf_files[0])

        warnings.warn(
            "There is no given ClearML dataset ID for the URDF assets! Trying to read the default directory in 5s.."
        )
        time.sleep(5.0)

        if not default_path.exists():
            raise FileNotFoundError(
                f"Couldn't resolve the path to the URDF file: Default file '{default_path}' doesn't exist!"
            )
        return default_path

    def _setup_config(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2  # total number of arm joints
        self._gripper_dim = 2  # number of gripper joints

        self._arm_dof_idx = th.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = th.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        # self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        # self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = self._args["default_arm_dof"]
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += self._args["default_gripper_dof"]

    def _init_pd_tensors(self) -> None:
        """Cache default PD tensors; call once after the entity is ready."""
        # TODO(issue#98) Move the robot calibration data into the URDF-dataset
        KP_GAINS = [4500.0, 4500.0, 3500.0, 3500.0, 3500.0, 3500.0, 100.0, 100.0]
        KV_GAINS = [350.0, 350.0, 250.0, 250.0, 250.0, 250.0, 10.0, 10.0]
        FORCE_LOWER = [-87.0, -87.0, -87.0, -87.0, -87.0, -87.0, -100.0, -100.0]
        FORCE_UPPER = [87.0, 87.0, 87.0, 87.0, 87.0, 87.0, 100.0, 100.0]

        # Sanity-check against the actual robot
        assert self._robot_entity.n_dofs == len(KP_GAINS)

        self._default_kp = self._build_gain_tensor(KP_GAINS)
        self._default_kv = self._build_gain_tensor(KV_GAINS)
        self._force_lower = self._build_gain_tensor(FORCE_LOWER)
        self._force_upper = self._build_gain_tensor(FORCE_UPPER)

    def _build_gain_tensor(self, values: list[float]) -> th.Tensor:
        return th.tensor(values, dtype=th.float32)

    def shutdown(self) -> None:
        pass

    def read_state(self) -> None:
        pass

    def set_joints_pd_gains(
        self,
        kp_gain: Optional[th.Tensor] = None,
        kv_gain: Optional[th.Tensor] = None,
    ) -> None:
        """
        Sets joints gains. Must be called after the build of the Genesis scene.
        """
        kp_g = kp_gain if kp_gain is not None else 1.0
        kv_g = kv_gain if kv_gain is not None else 1.0

        self._robot_entity.set_dofs_kp(self._default_kp * kp_g)
        self._robot_entity.set_dofs_kv(self._default_kv * kv_g)

        self._robot_entity.set_dofs_force_range(
            self._force_lower,
            self._force_upper,
        )
        # TODO(issue#57) configure armature, damping and stiffness
        # self._robot_entity.set_dofs_armature(
        #     th.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # )
        # self._robot_entity.set_dofs_stiffness(
        #     th.tensor([1.0, -30, -87, -87, -12, -12, -100, -100]),
        # )

    def ctrl_apply_vel_action(
        self,
        action: th.Tensor,
        open_gripper: Optional[bool] = None,
        envs_idx: Optional[th.IntTensor] = None,
    ) -> None:
        # Compute joint velocities using inverse velocity kinematics
        match self._ik_method:
            case "gs_ikv":
                q_vel = self._pseudoinverse_velocity_ik(action)
            case "dls_ikv":
                q_vel = self._dls_velocity_ik(action)
            case _:
                raise ValueError(f"Invalid IK method: {self._ik_method}")

        # Set gripper position if specified
        if open_gripper is not None:
            q_pos = self._robot_entity.get_qpos()
            if open_gripper:
                q_pos[:, self._fingers_dof] = self._gripper_open_dof
            else:
                q_pos[:, self._fingers_dof] = self._gripper_close_dof
            # Control gripper with position control
            if q_vel is not None:
                self._robot_entity.control_dofs_position(
                    position=q_pos[:, self._fingers_dof],
                    dofs_idx_local=self._fingers_dof,
                )
            self._robot_entity.control_dofs_position(position=q_pos)

        self._robot_entity.control_dofs_velocity(velocity=q_vel)

    def _pseudoinverse_velocity_ik(self, ee_velocity: th.Tensor) -> th.Tensor:
        """
        Pseudoinverse method for inverse velocity kinematics.

        Args:
            ee_velocity: [num_envs, 6] end-effector velocities [vx, vy, vz, wx, wy, wz]

        Returns:
            [num_envs, arm_dof_dim] joint velocities
        """
        # Get Jacobian matrix [num_envs, 6, n_dofs]
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)

        # Extract Jacobian for arm joints only
        jacobian_arm = jacobian[:, :, self._arm_dof_idx]  # [num_envs, 6, arm_dof_dim]

        # Use torch.linalg.pinv for numerical stability: J+ = J^T (J J^T)^-1
        jacobian_pinv = th.linalg.pinv(jacobian_arm)  # [num_envs, arm_dof_dim, 6]

        # Compute joint velocities: q_dot = J+ * ee_velocity
        q_vel = (jacobian_pinv @ ee_velocity.unsqueeze(-1)).squeeze(-1)

        # Append zeros for finger joints [num_envs, 2]
        finger_zeros = th.zeros(
            q_vel.shape[0], 2, device=q_vel.device, dtype=q_vel.dtype
        )
        return th.cat([q_vel, finger_zeros], dim=-1)

    def _dls_velocity_ik(self, ee_velocity: th.Tensor) -> th.Tensor:
        """
        Damped least squares method for inverse velocity kinematics.
        More stable near singularities than pseudoinverse.

        Args:
            ee_velocity: [num_envs, 6] end-effector velocities [vx, vy, vz, wx, wy, wz]

        Returns:
            [num_envs, arm_dof_dim] joint velocities
        """
        # Damping factor (tune this based on your application)
        lambda_val = 0.01

        # Get Jacobian matrix
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_arm = jacobian[:, :, self._arm_dof_idx]  # [num_envs, 6, arm_dof_dim]

        jacobian_T = jacobian_arm.transpose(1, 2)  # [num_envs, arm_dof_dim, 6]

        # Damping matrix
        lambda_matrix = (lambda_val**2) * th.eye(
            n=jacobian_arm.shape[1], device=self._device
        )  # [6, 6]

        # Damped least squares: q_dot = J^T (J J^T + λ^2 I)^-1 * ee_velocity
        q_vel = (
            jacobian_T
            @ th.inverse(jacobian_arm @ jacobian_T + lambda_matrix)
            @ ee_velocity.unsqueeze(-1)
        ).squeeze(-1)

        # Append zeros for finger joints [num_envs, 2]
        finger_zeros = th.zeros(
            q_vel.shape[0], 2, device=q_vel.device, dtype=q_vel.dtype
        )
        return th.cat([q_vel, finger_zeros], dim=-1)

    def ctrl_apply_joints_diff_action(
        self, joints_diff: th.Tensor, envs_idx: Optional[th.IntTensor] = None
    ) -> None:
        q_pos = self._robot_entity.get_qpos() + joints_diff
        self._robot_entity.control_dofs_position(position=q_pos)

    def ctrl_go_to_goal(
        self,
        goal_pose: th.Tensor,
        open_gripper: Optional[bool] = None,
        envs_idx: Optional[th.IntTensor] = None,
    ) -> None:
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        if open_gripper is not None:
            if open_gripper:
                q_pos[:, self._fingers_dof] = self._gripper_open_dof
            else:
                q_pos[:, self._fingers_dof] = self._gripper_close_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    def ctrl_go_to_home(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self._device)
        )

        default_joint_angles = th.tensor(
            self._default_joint_angles, dtype=th.float32, device=self._device
        ).repeat(len(idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=idx)
        self._robot_entity.control_dofs_position(
            position=default_joint_angles, envs_idx=idx
        )

    def ctrl_gripper_open(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self._device)
        )
        q_pos = self._robot_entity.get_qpos()
        q_pos[idx, self._fingers_dof] = self._gripper_open_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    def ctrl_gripper_close(self, envs_idx: Optional[th.IntTensor] = None) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self._device)
        )
        q_pos = self._robot_entity.get_qpos()
        q_pos[idx, self._fingers_dof] = self._gripper_close_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    def get_num_envs(self) -> int:
        return self._num_envs

    def get_n_dofs(self) -> int:
        return self._robot_entity.n_dofs

    def get_joints_positions(self) -> th.Tensor:
        return self._robot_entity.get_qpos()

    def get_joints_velocities(self) -> th.Tensor:
        # TODO get the joints vel from genesis
        raise NotImplementedError()

    def get_joints_efforts(self) -> th.Tensor:
        # TODO get the joints eff from genesis
        raise NotImplementedError()

    def get_ft_wrench(self) -> th.Tensor:
        # TODO get the F\T sensing from genesis
        raise NotImplementedError()

    def get_tcp_pose(self) -> th.Tensor:
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return th.cat([pos, quat], dim=-1).float()

    def get_tcp_position(self) -> th.Tensor:
        return self._ee_link.get_pos()

    def get_tcp_orientation(self) -> th.Tensor:
        return self._ee_link.get_quat()

    def get_base_pose(self) -> th.Tensor:
        pos, quat = self._robot_entity.get_pos(), self._robot_entity.get_quat()
        return th.cat([pos, quat], dim=-1).float()

    def get_gripper_width(self) -> th.Tensor:
        fingers = self._robot_entity.get_qpos()[:, self._fingers_dof]
        return fingers.sum(dim=1)

    def get_camera_image(
        self, camera_id: CameraID, modality: CameraModality = CameraModality.RGB
    ) -> th.Tensor:
        # TODO pass cameras reference to have  the same API for accessing images.
        raise NotImplementedError(
            "Currently in Genesis Sim, the manipulator doesn't have access to the cameras observation."
        )

    def get_all_cameras_images(
        self, modality: CameraModality = CameraModality.RGB
    ) -> TensorDict:
        # TODO pass cameras reference to have  the same API for accessing images.
        raise NotImplementedError(
            "Currently in Genesis Sim, the manipulator doesn't have access to the cameras observation."
        )
