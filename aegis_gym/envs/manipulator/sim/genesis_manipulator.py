import math
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import genesis as gs
import torch as th
from clearml import Dataset
from tensordict import TensorDict

from aegis_gym.aux.geom import (
    quat_to_rotvec_error,
    rotvec_to_quat,
    transform_by_quat,
    transform_quat_by_quat,
)
from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import CameraName, DomainRandomizationCfg, RobotCfg
from aegis_gym.envs.manipulator.base_manipulator import BaseManipulator, CameraModality

from .sim_admittance import SimAdmittance

RigidLink = TypeVar

# The ATI Axia80 output frame is `tool_mount_link` rotated by this angle about its Z.
# Measured by matching a simulated and a real 90 deg sweep (direction cosine 0.9998).
# Re-measure if the sensor or its adapter is remounted.
_FT_OUTPUT_YAW_DEG = 90.0


class GenesisManipulator(BaseManipulator):
    # Fraction of the policy step the position target leads the joints; larger values overshoot
    # the calibrated speed on the action changes (0.3 measured at ~96% of the per-step limit)
    _SERVO_TARGET_LEAD = 0.3
    # The pose target of the A1 mode trails the reference along its velocity by this much, so the
    # observed TCP lags the reference like on the robot (~17 ms at the observation, delta_pose_test
    # 2026-11-04); without it the stiff PD with velocity feedforward ends each step slightly ahead.
    _A1_TRACKING_LAG_S = 0.060
    # Fraction of a step's motion the real JTC has not yet commanded when the policy observes (its
    # segment is interpolated over the step); not integrated away by the Humble spring, because it
    # is part of the reference the controller follows.
    _A1_SEGMENT_REMAINDER = 0.37
    # Adds the inertial load of the links past the sensor to the simulated F/T, as the real
    # sensor reads it (a1_square*.csv: stop impulse = 1.25 kg * dv).
    _FT_TOOL_INERTIA = True

    def __init__(
        self,
        num_envs: int,
        gs_scene: gs.Scene,
        cameras_obs_getter: Callable[[CameraName, CameraModality], th.Tensor],
        available_cameras: dict[CameraName, tuple[CameraModality]],
        cfg_robot: RobotCfg,
        cfg_dr: DomainRandomizationCfg,
        show_cell: bool,
        policy_dt: float,
        max_linear_speed: float,
        max_angular_speed: float,
        device: th.device | None = None,
    ):
        super().__init__(device=device)

        logger = get_logger("Genesis::Manipulator")
        self._num_envs = num_envs
        self._gs_scene = gs_scene
        self._observe_camera_fn = cameras_obs_getter
        self._available_cameras = available_cameras
        self._cfg_robot = cfg_robot
        self._policy_dt = policy_dt
        self._ft_residual_bias = th.zeros(
            (num_envs, 6), dtype=th.float32, device=self.device
        )
        # Software tare offset, [num_envs, 6]; None until `set_ft_bias()`.
        self._ft_bias: th.Tensor | None = None
        self._ft_drift = th.zeros((num_envs, 6), dtype=th.float32, device=self.device)
        self._ft_drift_rate = th.zeros_like(self._ft_drift)
        self._ft_drift_walk = th.zeros(6, dtype=th.float32, device=self.device)
        self._ft_drift_step = -1

        cfg_noise = cfg_dr.ft_sensor_noise if cfg_dr.enabled else None
        self._ft_noise_std: th.Tensor | None = None
        self._ft_noise_limit: th.Tensor | None = None
        if cfg_noise is not None and cfg_noise.enabled:
            self._ft_noise_std = th.tensor(
                [[*cfg_noise.force_std, *cfg_noise.torque_std]],
                dtype=th.float32,
                device=self.device,
            )
            self._ft_noise_limit = th.tensor(
                [[*cfg_noise.force_limit, *cfg_noise.torque_limit]],
                dtype=th.float32,
                device=self.device,
            )
            self._ft_noise_corr = th.tensor(
                [cfg_noise.correlation], dtype=th.float32, device=self.device
            )
        self._ft_noise: th.Tensor | None = None
        self._ft_noise_step = -1

        # TODO(issue#99): Implement URDF model with cell collision handling
        if show_cell:
            self._urdf_model_id = cfg_robot.urdf_id_cell
        else:
            self._urdf_model_id = cfg_robot.urdf_id_no_cell

        if self._urdf_model_id:
            logger.info(f"URDF ClearML dataset ID: {self._urdf_model_id}")
        self._urdf_path = self._resolve_aegis_urdf()
        logger.info(f"URDF path: {self._urdf_path}")

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
                "tool_mount_link",
                "adapter_from_sensor",
                "robotiq_hande_coupler",
                "robotiq_hande_link",
            ],
        )
        self._robot_entity = gs_scene.add_entity(material=material, morph=morph)

        half = math.radians(_FT_OUTPUT_YAW_DEG) / 2.0
        self._ft_output_offset = th.tensor(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]],
            dtype=th.float32,
            device=self.device,
        )

        self._gripper_open_dof = 0.025
        self._gripper_close_dof = 0.0
        # calibrated on the real robot, may be scaled by the domain randomization
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

        self.set_ft_payload(cfg_robot.ft_payload_mass, cfg_robot.ft_payload_com)

        self._ik_method = cfg_robot.ik_method
        self._admittance = SimAdmittance(num_envs=num_envs, device=self.device)
        self._a1_commanded = th.zeros(num_envs, 3, device=self.device)
        self._tool_com_vel_prev: th.Tensor | None = None
        self._tool_com_acc: th.Tensor | None = None
        self._tool_acc_step = -1

        self._setup_config()
        self._add_joint_torque_sensor()
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

        self._arm_dof_idx = th.arange(self._arm_dof_dim, device=self.device)
        self._fingers_dof = th.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self.device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._cfg_robot.ee_link_name)
        self._ft_link = self._robot_entity.get_link("tool_mount_link")
        self._gravity_links = self._get_downstream_links(self._ft_link)
        self._gravity_link_idx = th.tensor(
            [link.idx for link in self._gravity_links],
            dtype=th.int32,
            device=self.device,
        )
        self._gravity_link_masses: th.Tensor | None = None
        self._gravity_link_local_coms: th.Tensor | None = None
        # self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        # self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = self._cfg_robot.default_arm_dof
        if self._cfg_robot.default_gripper_dof is not None:
            self._default_joint_angles += self._cfg_robot.default_gripper_dof

    def _get_downstream_links(self, root_link: RigidLink) -> list[RigidLink]:
        """Links mounted past `root_link` in the kinematic chain (its descendants)."""
        links = self._robot_entity.links
        link_start = self._robot_entity.link_start
        downstream = []
        for link in links:
            parent_idx = link.parent_idx
            while parent_idx != -1:
                if parent_idx == root_link.idx:
                    downstream.append(link)
                    break
                parent_idx = links[parent_idx - link_start].parent_idx
        return downstream

    def _add_joint_torque_sensor(self) -> None:
        self._joint_torque_sensor = self._gs_scene.add_sensor(
            gs.sensors.JointTorque(
                entity_idx=self._robot_entity.idx,
                dofs_idx_local=tuple(range(self._arm_dof_dim)),
            )
        )

    def _init_pd_tensors(self) -> None:
        """Cache default PD tensors; call once after the entity is ready."""
        # TODO(issue#98) Move the robot calibration data into the URDF-dataset
        KP_GAINS = [4500.0, 4500.0, 3500.0, 3500.0, 3500.0, 3500.0, 500.0, 500.0]
        KV_GAINS = [350.0, 350.0, 250.0, 250.0, 250.0, 250.0, 100.0, 100.0]
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
        kp_gain: th.Tensor | None = None,
        kv_gain: th.Tensor | None = None,
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
        open_gripper: bool | None = None,
        envs_idx: th.Tensor | None = None,
    ) -> None:
        self._admittance.deactivate()
        action[:, :3] *= self.max_linear_speed
        action[:, 3:] *= self.max_angular_speed

        q_vel = self._velocity_ik(action)

        # Emulates the real servo, which streams joint positions: the stiff PD tracks the commanded
        # velocity with a position target slightly ahead of the current joints, so the contacts can't
        # deflect the arm. Anchoring at the current joints (not the last target) avoids a windup.
        q_pos = self._robot_entity.get_qpos()
        arm_q_vel = q_vel[:, self._arm_dof_idx]
        self._robot_entity.control_dofs_position_velocity(
            position=q_pos[:, self._arm_dof_idx]
            + arm_q_vel * self._policy_dt * self._SERVO_TARGET_LEAD,
            velocity=arm_q_vel,
            dofs_idx_local=self._arm_dof_idx,
        )

        if open_gripper is None:
            return
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(
            position=q_pos[:, self._fingers_dof],
            dofs_idx_local=self._fingers_dof,
        )

    def _velocity_ik(self, ee_velocity: th.Tensor) -> th.Tensor:
        """[num_envs, 6] end-effector twist -> [num_envs, n_dofs] joint velocities (fingers 0)."""
        match self._ik_method:
            case "gs_ikv":
                return self._pseudoinverse_velocity_ik(ee_velocity)
            case "dls_ikv":
                return self._dls_velocity_ik(ee_velocity)
            case _:
                raise ValueError(f"Invalid IK method: {self._ik_method}")

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
            n=jacobian_arm.shape[1], device=self.device
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

    def ctrl_apply_delta_pose_action(
        self,
        delta: th.Tensor,
        open_gripper: bool | None = None,
        envs_idx: th.Tensor | None = None,
    ) -> th.Tensor:
        """Mirrors the real chain (see `SimAdmittance`): limited reference plus the spring offset,
        tracked by the stiff joint PD like the UR tracks servoj. The joint target is the measured
        joints plus the IK step to the target pose, so the PD lag does not wind up."""
        pos, quat = self.get_tcp_position(), self.get_tcp_orientation()
        force = transform_by_quat(self.get_ft_wrench()[:, :3], self.get_ft_frame_quat())
        adm = self._admittance
        starting = ~adm.active
        adm.activate(pos, quat, force)
        self._a1_commanded[starting] = pos[starting]
        tracking_error = pos - self._a1_commanded
        clipped, applied_dp, applied_dr = adm.step_reference(
            delta[:, :3], delta[:, 3:6], pos, quat
        )
        adm.step_spring(force, self._policy_dt, tracking_error)

        lin_vel = applied_dp / self._policy_dt
        rot_vel = applied_dr / self._policy_dt
        self._a1_commanded = adm.target_pos() - self._A1_SEGMENT_REMAINDER * applied_dp
        target_pos = (
            self._a1_commanded
            - lin_vel * self._A1_TRACKING_LAG_S
            + adm.step_velocity_memory(lin_vel, self._policy_dt)
        )
        target_quat = transform_quat_by_quat(
            rotvec_to_quat(-rot_vel * self._A1_TRACKING_LAG_S), adm.ref_quat
        )
        pose_error = th.cat(
            [target_pos - pos, quat_to_rotvec_error(target_quat, quat)], dim=-1
        )
        feedforward = th.cat([lin_vel + adm.offset_vel, rot_vel], dim=-1)
        q_pos = self._robot_entity.get_qpos()
        arm = self._arm_dof_idx
        self._robot_entity.control_dofs_position_velocity(
            position=q_pos[:, arm] + self._velocity_ik(pose_error)[:, arm],
            velocity=self._velocity_ik(feedforward)[:, arm],
            dofs_idx_local=arm,
        )

        if open_gripper is not None:
            q_pos[:, self._fingers_dof] = (
                self._gripper_open_dof if open_gripper else self._gripper_close_dof
            )
            self._robot_entity.control_dofs_position(
                position=q_pos[:, self._fingers_dof], dofs_idx_local=self._fingers_dof
            )
        return clipped

    def get_tcp_reference(self) -> th.Tensor:
        return th.where(
            self._admittance.active.unsqueeze(-1),
            th.cat([self._admittance.ref_pos, self._admittance.ref_quat], dim=-1),
            self.get_tcp_pose(),
        )

    def ctrl_apply_joints_diff_action(
        self, joints_diff: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        self._admittance.deactivate()
        q_pos = self._robot_entity.get_qpos() + joints_diff
        self._robot_entity.control_dofs_position(position=q_pos)

    def ctrl_go_to_goal(
        self,
        goal_pose: th.Tensor,
        open_gripper: bool | None = None,
        envs_idx: th.Tensor | None = None,
    ) -> None:
        self._admittance.deactivate()
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

    def ctrl_reset_to_pose(
        self,
        pose: th.Tensor,
        open_gripper: bool | None = None,
        envs_idx: th.Tensor | None = None,
    ) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )
        self._admittance.deactivate(idx)
        # IK is seeded with the current joints, so reset to home beforehand to keep the arm configuration
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=pose[:, :3],
            quat=pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
            envs_idx=idx,
        )
        if open_gripper is not None:
            if open_gripper:
                q_pos[:, self._fingers_dof] = self._gripper_open_dof
            else:
                q_pos[:, self._fingers_dof] = self._gripper_close_dof

        self._robot_entity.set_qpos(q_pos, envs_idx=idx)
        self._tool_com_vel_prev = None
        self._robot_entity.control_dofs_position(position=q_pos, envs_idx=idx)

    def ctrl_go_to_home(self, envs_idx: th.Tensor | None = None) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )

        self._admittance.deactivate(idx)
        default_joint_angles = th.tensor(
            self._default_joint_angles, dtype=th.float32, device=self.device
        ).repeat(len(idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=idx)
        self._tool_com_vel_prev = None
        self._robot_entity.control_dofs_position(
            position=default_joint_angles, envs_idx=idx
        )

    def ctrl_go_to_joints(
        self, joints: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )

        self._admittance.deactivate(idx)
        joints = joints.reshape(-1, self._arm_dof_dim).to(
            dtype=th.float32, device=self.device
        )
        if joints.shape[0] == 1:
            joints = joints.expand(len(idx), self._arm_dof_dim)

        # Only the arm DOFs are driven; the fingers keep whatever target they hold, so
        self._robot_entity.control_dofs_position(
            position=joints, dofs_idx_local=self._arm_dof_idx, envs_idx=idx
        )

    def ctrl_gripper_open(self, envs_idx: th.Tensor | None = None) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )
        q_pos = self._robot_entity.get_qpos()
        q_pos[idx[:, None], self._fingers_dof] = self._gripper_open_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    def ctrl_gripper_close(self, envs_idx: th.Tensor | None = None) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )
        q_pos = self._robot_entity.get_qpos()
        q_pos[idx[:, None], self._fingers_dof] = self._gripper_close_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    def get_n_dofs(self) -> int:
        return self._robot_entity.n_dofs

    def get_joints_positions(self) -> th.Tensor:
        return self._robot_entity.get_qpos()

    def get_joints_velocities(self) -> th.Tensor:
        return self._robot_entity.get_dofs_velocity()

    def get_joints_efforts(self) -> th.Tensor:
        return self._joint_torque_sensor.read()

    def _gravity_wrench_world(self) -> th.Tensor:
        """Wrench at the sensor (world frame) from the weight of the links past it.
        The robot uses `gravity_compensation=1.0`, so the simulation itself never
        applies this weight and it is added here.
        """
        if self._gravity_link_masses is None:
            # Per env ([num_envs, n_links] and [num_envs, n_links, 3]), as
            # `batch_links_info` is on.
            links_idx = [link.idx for link in self._gravity_links]
            solver = self._ft_link.solver
            self._gravity_link_masses = solver.get_links_mass(links_idx=links_idx).to(
                dtype=th.float32, device=self.device
            )
            self._gravity_link_local_coms = solver.get_links_COM(
                links_idx=links_idx
            ).to(dtype=th.float32, device=self.device)

        sensor_pos = self._ft_link.get_pos()  # [num_envs, 3]
        gravity = self._ft_link.solver.get_gravity().expand_as(
            sensor_pos
        )  # [num_envs, 3]

        if self._ft_payload_mass is not None:
            weight = self._ft_payload_mass * gravity  # [num_envs, 3]
            lever = transform_by_quat(
                self._ft_payload_com.expand_as(sensor_pos), self.get_ft_frame_quat()
            )
            return th.cat([weight, th.linalg.cross(lever, weight)], dim=-1)

        force = th.zeros_like(sensor_pos)
        torque = th.zeros_like(sensor_pos)
        for i, link in enumerate(self._gravity_links):
            local_com = self._gravity_link_local_coms[:, i]  # [num_envs, 3]
            mass = self._gravity_link_masses[:, i, None]  # [num_envs, 1]
            com_world = link.get_pos() + transform_by_quat(local_com, link.get_quat())
            weight = mass * gravity  # [num_envs, 3]
            force = force + weight
            torque = torque + th.linalg.cross(com_world - sensor_pos, weight)

        return th.cat([force, torque], dim=-1)

    def get_ft_wrench(self) -> th.Tensor:
        """The simulated F/T measurement, tared when `set_ft_bias()` has been called."""
        wrench = self.get_ft_wrench_raw()
        if self._ft_bias is not None:
            wrench = wrench - self._ft_bias
        return wrench + self._ft_residual_bias + self._advance_ft_drift()

    def set_ft_drift(
        self, rate: th.Tensor, walk_std: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        """Simulation only: restarts the zero drift of `envs_idx` from zero with a linear `rate`
        [len(envs_idx), 6] (N/s, Nm/s) plus a random walk of `walk_std` [6] (per sqrt(s))."""
        idx = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )
        self._ft_drift[idx] = 0.0
        self._ft_drift_rate[idx] = rate.to(dtype=th.float32, device=self.device)
        self._ft_drift_walk = walk_std.to(dtype=th.float32, device=self.device)

    def _advance_ft_drift(self) -> th.Tensor:
        """The drift since the last tare, advanced once per `scene.step()`."""
        step = self._gs_scene.sim.cur_step_global
        if step != self._ft_drift_step:
            if self._ft_drift_step >= 0:
                steps = step - self._ft_drift_step
                dt = steps * self._policy_dt
                self._ft_drift = (
                    self._ft_drift
                    + self._ft_drift_rate * dt
                    + th.randn_like(self._ft_drift)
                    * self._ft_drift_walk
                    * math.sqrt(dt)
                )
            self._ft_drift_step = step
        return self._ft_drift

    def set_ft_bias(self) -> None:
        self._admittance.deactivate()
        self._ft_drift.zero_()
        self._ft_bias = self.get_ft_wrench_raw().detach().clone()
        self.note_ft_bias_pose()

    def clear_ft_bias(self) -> None:
        self._admittance.deactivate()
        self._ft_bias = None
        self._ft_gravity_at_bias = None

    def is_ft_biased(self) -> bool:
        return self._ft_bias is not None

    def get_ft_bias(self) -> th.Tensor | None:
        """Simulation only: the stored offset, [num_envs, 6], or None when untared."""
        return None if self._ft_bias is None else self._ft_bias.clone()

    def _tool_links_com(self) -> tuple[th.Tensor, th.Tensor]:
        """World COM positions and velocities [num_envs, n_links, 3] of the links past the sensor."""
        self._gravity_wrench_world()  # caches the link masses and local COMs
        pos, vel = [], []
        for i, link in enumerate(self._gravity_links):
            lever = transform_by_quat(
                self._gravity_link_local_coms[:, i], link.get_quat()
            )
            pos.append(link.get_pos() + lever)
            vel.append(link.get_vel() + th.linalg.cross(link.get_ang(), lever))
        return th.stack(pos, dim=1), th.stack(vel, dim=1)

    def _inertial_wrench_world(self) -> th.Tensor:
        """Load at the sensor (world frame) from accelerating the links past it: -sum m_i a_i.

        `a_i` is the mean COM acceleration over the last policy step (velocity difference), not an
        instantaneous sample: the admittance reads the wrench once per step, and what deflects the
        real arm is the impulse m * dv of a velocity change. Updated once per `scene.step()`; zero
        for the first step after a teleport (`set_qpos`).
        """
        com_pos, com_vel = self._tool_links_com()
        step = self._gs_scene.sim.cur_step_global
        if step != self._tool_acc_step:
            if self._tool_com_vel_prev is None:
                self._tool_com_acc = th.zeros_like(com_vel)
            else:
                self._tool_com_acc = (
                    com_vel - self._tool_com_vel_prev
                ) / self._policy_dt
            self._tool_com_vel_prev = com_vel.clone()
            self._tool_acc_step = step
        load = -self._gravity_link_masses.unsqueeze(-1) * self._tool_com_acc
        lever = com_pos - self._ft_link.get_pos().unsqueeze(1)
        return th.cat(
            [load.sum(dim=1), th.linalg.cross(lever, load, dim=-1).sum(dim=1)], dim=-1
        )

    def get_ft_wrench_raw(self) -> th.Tensor:
        """Simulation only: the modelled wrench before the software tare."""
        wrench_world = self._contact_wrench_world() + self._gravity_wrench_world()
        if self._FT_TOOL_INERTIA:
            wrench_world = wrench_world + self._inertial_wrench_world()

        quat = self.get_ft_frame_quat()  # [num_envs, 4], WXYZ
        quat_conj = quat * th.tensor(
            [1.0, -1.0, -1.0, -1.0], device=quat.device, dtype=quat.dtype
        )
        force_local = transform_by_quat(wrench_world[:, :3], quat_conj)
        torque_local = transform_by_quat(wrench_world[:, 3:], quat_conj)

        return self._add_ft_noise(th.cat([force_local, torque_local], dim=-1))

    def set_ft_residual_bias(
        self, bias: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        """Sets the post-tare residual offset, [len(envs_idx), 6] or [6].

        Simulation only: on the real robot this offset is a physical property of the
        sensor, not something that can be dialled in.
        """
        idx = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )
        self._ft_residual_bias[idx] = bias.to(
            dtype=th.float32, device=self.device
        ).reshape(len(idx), 6)

    def get_ft_residual_bias(self) -> th.Tensor:
        """Simulation only: the current post-tare residual offset, [num_envs, 6]."""
        return self._ft_residual_bias.clone()

    def _add_ft_noise(self, wrench: th.Tensor) -> th.Tensor:
        """Gaussian measurement noise, truncated per axis, advanced once per `scene.step()` as a
        first-order process: rho = `correlation` between consecutive steps (0 is white), the
        stationary std stays `force_std` / `torque_std`."""
        if self._ft_noise_std is None:
            return wrench
        step = self._gs_scene.sim.cur_step_global
        if step != self._ft_noise_step:
            fresh = th.randn_like(wrench) * self._ft_noise_std
            if self._ft_noise is None or self._ft_noise.shape != wrench.shape:
                self._ft_noise = fresh
            else:
                rho = self._ft_noise_corr
                self._ft_noise = rho * self._ft_noise + th.sqrt(1.0 - rho * rho) * fresh
            self._ft_noise_step = step
        return wrench + self._ft_noise.clamp(
            -self._ft_noise_limit, self._ft_noise_limit
        )

    def _contact_wrench_world(self) -> th.Tensor:
        """Wrench at the sensor (world frame) from the contacts on the links past it."""
        sensor_pos = self._ft_link.get_pos()  # [num_envs, 3]
        contacts = self._robot_entity.get_contacts(
            exclude_self_contact=True, is_padded=True
        )

        valid = contacts["valid_mask"]  # [num_envs, n_contacts]
        if valid.numel() == 0:
            return th.zeros((sensor_pos.shape[0], 6), device=self.device)

        # `force_b` is the force on `link_b` and `force_a` the force on `link_a`; take
        # whichever side of the pair is ours, and drop contacts upstream of the sensor.
        is_a = th.isin(contacts["link_a"], self._gravity_link_idx)
        is_b = th.isin(contacts["link_b"], self._gravity_link_idx)
        force = th.where(is_b.unsqueeze(-1), contacts["force_b"], contacts["force_a"])
        keep = (valid & (is_a | is_b)).unsqueeze(-1)
        force = th.where(keep, force, th.zeros_like(force))

        lever = contacts["position"] - sensor_pos.unsqueeze(1)
        return th.cat(
            [force.sum(dim=1), th.linalg.cross(lever, force, dim=-1).sum(dim=1)],
            dim=-1,
        )

    def get_ft_frame_quat(self) -> th.Tensor:
        """World orientation [num_envs, 4] (WXYZ) of the frame `get_ft_wrench()` reports
        in: `tool_mount_link` rotated by `_FT_OUTPUT_YAW_DEG` about its Z."""
        quat = self._ft_link.get_quat()
        return transform_quat_by_quat(quat, self._ft_output_offset.expand_as(quat))

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
        self, camera: CameraName, modality: CameraModality = CameraModality.RGB
    ) -> th.Tensor:
        return self._observe_camera_fn(camera, modality)

    def get_all_cameras_images(
        self, modality: CameraModality = CameraModality.RGB
    ) -> TensorDict:
        res = {}
        for cam, cam_modalities in self._available_cameras.items():
            if modality not in cam_modalities:
                raise ValueError(f"Camera {cam} doesn't support modality {modality}")
            res[cam] = self._observe_camera_fn(cam, modality)
        return TensorDict(res)

    def get_robot_link(self, link: str) -> RigidLink:
        return self._robot_entity.get_link(link)
