import math
import time
import warnings
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import genesis as gs
import torch as th
from clearml import Dataset
from tensordict import TensorDict

from aegis_gym.aux.geom import transform_by_quat, transform_quat_by_quat
from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import CameraName, DomainRandomizationCfg, RobotCfg
from aegis_gym.envs.manipulator import BaseManipulator, CameraModality

RigidLink = TypeVar

# The ATI Axia80 does not report in `tool_mount_link`'s axes: its output frame is that
# link turned by this much about Z. Measured, not assumed -- comparing a simulated and a
# real 90 deg sweep, the rotation taking the simulated wrench onto the real one came out
# as Rz(-90 deg) with a direction cosine of +0.9998 on force and +0.9999 on torque, both
# channels agreeing. Expressing the simulated wrench in the sensor's own frame therefore
# means rotating that frame by +90 deg about Z.
# Re-measure this if the sensor is remounted or the adapter changes.
_FTS_OUTPUT_YAW_DEG = 90.0


class GenesisManipulator(BaseManipulator):
    def __init__(
        self,
        num_envs: int,
        gs_scene: gs.Scene,
        cameras_obs_getter: Callable[[CameraName, CameraModality], th.Tensor],
        available_cameras: dict[CameraName, tuple[CameraModality]],
        cfg_robot: RobotCfg,
        cfg_dr: DomainRandomizationCfg,
        show_cell: bool,
        device: th.device | None = None,
    ):
        super().__init__(device=device)

        logger = get_logger("Genesis::Manipulator")
        self._num_envs = num_envs
        self._gs_scene = gs_scene
        self._observe_camera_fn = cameras_obs_getter
        self._available_cameras = available_cameras
        self._cfg_robot = cfg_robot
        self._ft_residual_bias = th.zeros(
            (num_envs, 6), dtype=th.float32, device=self.device
        )

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
                # Kept explicitly so their mass isn't folded into `tool_mount_link` by
                # the fixed-joint reduction: they carry the bulk of the gripper's
                # weight and must stay separate, downstream links for
                # `_gravity_wrench_world()` to account for it.
                "adapter_from_sensor",
                "robotiq_hande_coupler",
                "robotiq_hande_link",
            ],
        )
        self._robot_entity = gs_scene.add_entity(material=material, morph=morph)

        # Software tare offset, [num_envs, 6]; None until `set_ft_bias()`. The real
        # robot has no counterpart -- its sensor tares itself.
        self._ft_bias: th.Tensor | None = None

        self._fts_payload_mass: th.Tensor | None = None
        self._fts_payload_com: th.Tensor | None = None
        if cfg_robot.fts_payload_mass is not None:
            if cfg_robot.fts_payload_com is None:
                raise ValueError(
                    "fts_payload_mass is set but fts_payload_com is not; both are "
                    "needed to place the measured payload."
                )
            self._fts_payload_mass = th.tensor(
                cfg_robot.fts_payload_mass, dtype=th.float32, device=self.device
            )
            self._fts_payload_com = th.tensor(
                [cfg_robot.fts_payload_com], dtype=th.float32, device=self.device
            )
            logger.info(
                f"F/T payload calibration: {cfg_robot.fts_payload_mass:.4f} kg at "
                f"{cfg_robot.fts_payload_com} m (sensor frame)"
            )

        half = math.radians(_FTS_OUTPUT_YAW_DEG) / 2.0
        self._fts_output_offset = th.tensor(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]],
            dtype=th.float32,
            device=self.device,
        )

        self._gripper_open_dof = 0.025
        self._gripper_close_dof = 0.0
        self.max_linear_speed = 1.0
        self.max_angular_speed = 1.0

        # The servo integrates its own setpoint (see `_servo_arm`), so it needs to know
        # how long each command is held for. The scene overwrites this with `policy_dt`
        # when it builds the manipulator; 25 Hz is the rate the real cell servos at.
        self.servo_dt = 0.04
        self._q_servo_target: th.Tensor | None = None
        self._servo_max_error = float(cfg_robot.servo_follow_error_max_rad)
        self._servo_delay_s = float(cfg_robot.servo_command_delay_s)
        self._servo_queue: deque[th.Tensor] = deque()

        self.set_ft_payload(cfg_robot.fts_payload_mass, cfg_robot.fts_payload_com)

        self._ik_method = cfg_robot.ik_method
        self._fts_use_contact = cfg_robot.fts_wrench_source == "contact"
        logger.info(f"F/T contact model: {cfg_robot.fts_wrench_source}")

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
        self._fts_link = self._robot_entity.get_link("tool_mount_link")
        self._gravity_links = self._get_downstream_links(self._fts_link)
        # Global link indices past the sensor, for matching contacts against.
        self._gravity_link_idx = th.tensor(
            [link.idx for link in self._gravity_links],
            dtype=th.int32,
            device=self.device,
        )
        # Mass/COM are only available once the Genesis scene is built, so these are
        # filled in lazily on first use (see `_gravity_wrench_world`).
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
        KP_GAINS = [4500.0, 4500.0, 3500.0, 3500.0, 3500.0, 3500.0, 5000.0, 5000.0]
        KV_GAINS = [350.0, 350.0, 250.0, 250.0, 250.0, 250.0, 30.0, 30.0]
        FORCE_LOWER = [-87.0, -87.0, -87.0, -87.0, -87.0, -87.0, -100.0, -100.0]
        FORCE_UPPER = [87.0, 87.0, 87.0, 87.0, 87.0, 87.0, 100.0, 100.0]

        # Sanity-check against the actual robot
        assert self._robot_entity.n_dofs == len(KP_GAINS)

        # Joint stiffness is what sets how fast contact force builds: with an
        # integrating setpoint the force is K_eff times the following error, so scaling
        # kp scales the entire force-against-lag curve. Compared against the robot over
        # the same 1.5 mm/s press, the simulated arm needed ~3.0 mm of command to reach
        # 5 N where the robot needed ~2.8 mm -- about 6x softer -- and this is the knob
        # that closes it. kv follows as sqrt(scale) so the damping ratio is preserved;
        # raising kp on its own leaves the joint underdamped and it rings on contact.
        scale = float(self._cfg_robot.servo_stiffness_scale)
        if scale != 1.0:
            root = math.sqrt(scale)
            for i in range(self._arm_dof_dim):
                KP_GAINS[i] *= scale
                KV_GAINS[i] *= root
            get_logger("Genesis::Manipulator").info(
                f"Servo stiffness scaled x{scale:g} (kv x{root:.2f}); arm kp now "
                f"{KP_GAINS[: self._arm_dof_dim]}"
            )

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
        action = self._delay_command(action)
        action[:, :3] *= self.max_linear_speed
        action[:, 3:] *= self.max_angular_speed

        # Compute joint velocities using inverse velocity kinematics
        match self._ik_method:
            case "gs_ikv":
                q_vel = self._pseudoinverse_velocity_ik(action)
            case "dls_ikv":
                q_vel = self._dls_velocity_ik(action)
            case _:
                raise ValueError(f"Invalid IK method: {self._ik_method}")

        # Set gripper position if specified. Scoped to the fingers: a whole-DOF
        # position command here would reset the arm's setpoint to wherever the arm
        # currently is, which is precisely the accumulation `_servo_arm` depends on.
        if open_gripper is not None:
            q_pos = self._robot_entity.get_qpos()
            if open_gripper:
                q_pos[:, self._fingers_dof] = self._gripper_open_dof
            else:
                q_pos[:, self._fingers_dof] = self._gripper_close_dof
            self._robot_entity.control_dofs_position(
                position=q_pos[:, self._fingers_dof],
                dofs_idx_local=self._fingers_dof,
            )

        self._servo_arm(q_vel[:, self._arm_dof_idx])

    def _delay_command(self, action: th.Tensor) -> th.Tensor:
        """Holds a command back by `servo_command_delay_s` before it reaches the arm."""
        steps = round(self._servo_delay_s / max(self.servo_dt, 1e-9))
        if steps <= 0:
            return action
        if not self._servo_queue:
            self._servo_queue.extend(action.clone() for _ in range(steps))
        self._servo_queue.append(action.clone())
        return self._servo_queue.popleft()

    def resync_servo_target(self) -> None:
        """Drops the integrated setpoint, so the next velocity action starts from
        wherever the arm actually is.

        Anything that moves the arm by other means has to call this, or the servo will
        carry a following error left over from a pose that no longer exists.
        """
        self._q_servo_target = None
        self._servo_queue.clear()

    def _servo_arm(self, q_vel_arm: th.Tensor) -> None:
        """Advances the arm's joint setpoint by the commanded velocity, the way the
        robot's own `speedj` does, and drives the joints to it.

        Handing `q_vel_arm` straight to `control_dofs_velocity` looks equivalent and is
        not. That controller is proportional on velocity alone, so once contact blocks
        the joint it settles at `kv * q_vel` and stays there for good: the achievable
        contact force becomes a fixed multiple of the commanded *speed*, about 1.1 N per
        mm/s here, and pressing for longer adds nothing. Measured at 1.5 mm/s it tops
        out at 1.66 N after 10 s and is still 1.65 N after 40 s -- which is why a probe
        waiting on 5 N waited forever. Re-deriving a position target from the measured
        pose each step has the same defect for the same reason: the error is reset
        before it can grow.

        Integrating instead means the setpoint keeps descending while the tool is held
        up by the surface. Following error accumulates, and with it the torque, exactly
        as on the real arm -- where that accumulation is visible as the 51 -> 103 mm of
        commanded travel the arm reports while the force builds.

        Unbounded, that reaches any force at all (611 N in the same 40 s test), which is
        not the real robot either: the cell bounds it with a force limit and a
        protective stop. `_servo_max_error` is that bound, expressed as the largest
        following error a joint may hold.
        """
        q_now = self._robot_entity.get_qpos()[:, self._arm_dof_idx]
        if self._q_servo_target is None or self._q_servo_target.shape != q_now.shape:
            self._q_servo_target = q_now.clone()

        target = self._q_servo_target + q_vel_arm * self.servo_dt
        self._q_servo_target = th.clamp(
            target, q_now - self._servo_max_error, q_now + self._servo_max_error
        )
        self._robot_entity.control_dofs_position(
            position=self._q_servo_target, dofs_idx_local=self._arm_dof_idx
        )

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

    def ctrl_apply_joints_diff_action(
        self, joints_diff: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        q_pos = self._robot_entity.get_qpos() + joints_diff
        self._robot_entity.control_dofs_position(position=q_pos)
        self.resync_servo_target()

    def ctrl_go_to_goal(
        self,
        goal_pose: th.Tensor,
        open_gripper: bool | None = None,
        envs_idx: th.Tensor | None = None,
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
        self.resync_servo_target()

    def ctrl_go_to_home(self, envs_idx: th.Tensor | None = None) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )

        default_joint_angles = th.tensor(
            self._default_joint_angles, dtype=th.float32, device=self.device
        ).repeat(len(idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=idx)
        self._robot_entity.control_dofs_position(
            position=default_joint_angles, envs_idx=idx
        )
        self.resync_servo_target()

    def ctrl_go_to_joints(
        self, joints: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        idx: th.Tensor = (
            envs_idx
            if envs_idx is not None
            else th.arange(self._num_envs, device=self.device)
        )

        joints = joints.reshape(-1, self._arm_dof_dim).to(
            dtype=th.float32, device=self.device
        )
        if joints.shape[0] == 1:
            joints = joints.expand(len(idx), self._arm_dof_dim)

        # Only the arm DOFs are driven; the fingers keep whatever target they hold, so
        # this cannot disturb the gripper mid-test.
        self._robot_entity.control_dofs_position(
            position=joints, dofs_idx_local=self._arm_dof_idx, envs_idx=idx
        )
        self.resync_servo_target()

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
        # TODO(issue#126) get the joints eff from genesis
        return self._joint_torque_sensor.read()

    def _gravity_wrench_world(self) -> th.Tensor:
        """Wrench felt at the F/T sensor from the weight of the links mounted past it
        (gripper and its attachments). The robot's material sets
        `gravity_compensation=1.0`, which cancels true gravity in the simulated
        dynamics, so `_joint_torque_sensor` never sees this weight and it has to be
        added back by hand to match what a real F/T sensor would read.
        """
        if self._gravity_link_masses is None:
            # genesis-world 1.x removed `RigidLink.inertial_pos` and made
            # `RigidLink.get_mass()` return a per-environment tensor instead of a
            # float. Both values now come from the solver in one batched read:
            # `RigidOptions.batch_links_info` is on, so these are shaped
            # [num_envs, n_links] and [num_envs, n_links, 3] and per-environment
            # inertial properties stay distinct rather than collapsing to env 0.
            links_idx = [link.idx for link in self._gravity_links]
            solver = self._fts_link.solver
            self._gravity_link_masses = solver.get_links_mass(links_idx=links_idx).to(
                dtype=th.float32, device=self.device
            )
            self._gravity_link_local_coms = solver.get_links_COM(
                links_idx=links_idx
            ).to(dtype=th.float32, device=self.device)

        sensor_pos = self._fts_link.get_pos()  # [num_envs, 3]
        gravity = self._fts_link.solver.get_gravity().expand_as(
            sensor_pos
        )  # [num_envs, 3]

        if self._fts_payload_mass is not None:
            weight = self._fts_payload_mass * gravity  # [num_envs, 3]
            lever = transform_by_quat(
                self._fts_payload_com.expand_as(sensor_pos), self.get_ft_frame_quat()
            )
            return th.cat([weight, th.linalg.cross(lever, weight)], dim=-1)

        force = th.zeros_like(sensor_pos)
        torque = th.zeros_like(sensor_pos)
        # Indexed per link rather than zipped: the batched reads above put the
        # environment on dim 0, so iterating them directly would walk envs.
        for i, link in enumerate(self._gravity_links):
            local_com = self._gravity_link_local_coms[:, i]  # [num_envs, 3]
            mass = self._gravity_link_masses[:, i, None]  # [num_envs, 1]
            com_world = link.get_pos() + transform_by_quat(local_com, link.get_quat())
            weight = mass * gravity  # [num_envs, 3]
            force = force + weight
            torque = torque + th.linalg.cross(com_world - sensor_pos, weight)

        return th.cat([force, torque], dim=-1)

    def get_ft_wrench(self) -> th.Tensor:
        """The simulated F/T measurement, tared when `set_ft_bias()` has been called.

        Counterpart of the real sensor's reading. Unlike the real robot, which tares in
        hardware, the offset is subtracted here.
        """
        wrench = self.get_ft_wrench_raw()
        if self._ft_bias is not None:
            wrench = wrench - self._ft_bias
        return wrench + self._ft_residual_bias

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

    def set_ft_bias(self) -> None:
        self._ft_bias = self.get_ft_wrench_raw().detach().clone()
        self.note_ft_bias_pose()

    def clear_ft_bias(self) -> None:
        self._ft_bias = None
        self._ft_gravity_at_bias = None

    def is_ft_biased(self) -> bool:
        return self._ft_bias is not None

    def get_ft_bias(self) -> th.Tensor | None:
        """Simulation only: the stored offset, [num_envs, 6], or None when untared.
        The real robot keeps its tare inside the sensor and cannot answer this."""
        return None if self._ft_bias is None else self._ft_bias.clone()

    def get_ft_wrench_raw(self) -> th.Tensor:
        """Simulation only: the modelled wrench before the software tare.

        There is no real-robot counterpart -- the hardware reports one measurement and
        the tare is applied inside it. This exists to check the simulated model against
        real data, where the untared term is what a sign or frame error shows up in.
        """
        wrench_world = (
            self._contact_wrench_world()
            if self._fts_use_contact
            else self._jacobian_wrench_world()
        )
        wrench_world = wrench_world + self._gravity_wrench_world()

        quat = self.get_ft_frame_quat()  # [num_envs, 4], WXYZ
        quat_conj = quat * th.tensor(
            [1.0, -1.0, -1.0, -1.0], device=quat.device, dtype=quat.dtype
        )
        force_local = transform_by_quat(wrench_world[:, :3], quat_conj)
        torque_local = transform_by_quat(wrench_world[:, 3:], quat_conj)

        return self._add_ft_noise(th.cat([force_local, torque_local], dim=-1))

    def _contact_wrench_world(self) -> th.Tensor:
        sensor_pos = self._fts_link.get_pos()  # [num_envs, 3]
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

    def _jacobian_wrench_world(self) -> th.Tensor:
        """Wrench inferred from actuator torque. See `_contact_wrench_world()` for why
        this is kept only for comparison."""
        tau = self.get_joints_efforts()  # [num_envs, 6]

        jacobian = self._robot_entity.get_jacobian(link=self._fts_link)
        jacobian_arm = jacobian[:, :, self._arm_dof_idx]  # [num_envs, 6, 6]
        jacobian_arm_T = jacobian_arm.transpose(1, 2)  # [num_envs, 6, 6]

        # `tau = J^T @ F` solves for the wrench the arm delivers to hold/push the tip.
        # A real F/T sensor reports the opposite: the load's reaction acting on the sensor
        # (e.g. a hanging weight reads as pulling down, not as the arm holding it up).
        return -(th.linalg.pinv(jacobian_arm_T) @ tau.unsqueeze(-1)).squeeze(-1)

    def _add_ft_noise(self, wrench: th.Tensor) -> th.Tensor:
        """Gaussian measurement noise, drawn fresh per read and truncated per axis."""
        if self._ft_noise_std is None:
            return wrench
        noise = th.randn_like(wrench) * self._ft_noise_std
        return wrench + noise.clamp(-self._ft_noise_limit, self._ft_noise_limit)

    def get_ft_frame_quat(self) -> th.Tensor:
        """World orientation, [num_envs, 4] WXYZ, of the frame `get_ft_wrench()` reports in.

        That is the physical sensor's output frame, which is `tool_mount_link` turned by
        `_FTS_OUTPUT_YAW_DEG` about its Z (see the constant). Use this, not
        `_fts_link.get_quat()`, whenever a wrench component has to be tied to an axis.
        """
        quat = self._fts_link.get_quat()
        return transform_quat_by_quat(quat, self._fts_output_offset.expand_as(quat))

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
