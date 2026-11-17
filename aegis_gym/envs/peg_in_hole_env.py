import math
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import torch as th
from tensordict import TensorDict

from aegis_gym.aux.geom import (
    quat_to_rotvec_error,
    transform_by_quat,
    transform_quat_by_quat,
)
from aegis_gym.aux.logging import get_logger
from aegis_gym.aux.rand import rand_uniform
from aegis_gym.config.types import CameraName, ExpConfig
from aegis_gym.envs.base_env import BaseEnv, Modality, ResetReturn, StepReturn
from aegis_gym.envs.manipulator import BaseManipulator
from aegis_gym.envs.objects import BaseURDF, ObjectProperties, ObjectType

from .peg_in_hole_geometry import (
    PegInHoleGeometry,
    quat_to_yaw,
    wrap_symmetric,
    yaw_to_quat,
)
from .registry import register_env
from .scene import BaseScene


@register_env("peg_in_hole")
class PegInHoleEnv(BaseEnv):
    DEFAULT_MODALITIES = frozenset(
        {Modality.TCP_POSE, Modality.OBJECT_POSE, Modality.TCP_WRENCH}
    )
    DEFAULT_EPISODE_LENGTH_S = 10.0
    PREVIEW_CAMERA_POSE = ((0.85, 0.0, 0.35), (0.55, 0.0, 0.05))
    PRIVILEGED_OBS_GROUP = "privileged"

    _ACTION_DIM = 4
    _TCP_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)
    _NUM_KEYPOINTS = 4
    _KEYPOINT_SQUASH: ClassVar[dict[str, tuple[float, float]]] = {
        "keypoints_baseline": (5.0, 4.0),
        "keypoints_coarse": (50.0, 2.0),
        "keypoints_fine": (100.0, 0.0),
    }
    _ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "peg_in_hole"

    def __init__(self, scene: BaseScene, cfg: ExpConfig):
        super().__init__(scene=scene, cfg=cfg)
        self._extract_config()
        self._observation_fns = {
            Modality.TCP_POSE: self._observe_tcp_pose,
            Modality.OBJECT_POSE: self._observe_peg_pose,
            Modality.TCP_WRENCH: self._observe_tcp_wrench,
            Modality.CAMERA_SCENE_RGB: self._observe_camera_scene,
            Modality.CAMERA_TOOL_LEFT_RGB: self._observe_camera_tool_left,
            Modality.CAMERA_TOOL_RIGHT_RGB: self._observe_camera_tool_right,
        }

        self._setup_scene(cfg=cfg)
        self._scene.build()
        self.manipulator: BaseManipulator = self._scene.get_manipulator()

        self._init_geometry_buffers()
        self._init_reward_functions()
        self._init_buffers()
        self.reset_idx(th.arange(self.num_envs, device=self.device))
        self._scene.attach_object_to_tcp(self.peg)
        self.reset()

    def _extract_config(self) -> None:
        # TODO(issue##117) redesign the whole camera preview system
        self.show_cameras_gui = self._cfg_env.visualize_camera
        self.show_cell = self._cfg_env.visualize_cell

        self.num_obs = self._cfg_env.num_obs
        self.num_privileged_obs = None
        self.num_actions = self._ACTION_DIM
        if self._cfg_env.num_actions != self._ACTION_DIM:
            get_logger("PegInHoleEnv").warning(
                f"Ignoring `num_actions={self._cfg_env.num_actions}`: the peg_in_hole env uses "
                f"the [vx, vy, vz, wz] actions (num_actions={self._ACTION_DIM})."
            )
        self.image_width = self._cfg_env.image_resolution[0]
        self.image_height = self._cfg_env.image_resolution[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.cameras_setup = self._cfg_env.cameras_setup
        self.table_size = self._cfg_env.table_size
        self.workbench_size = self._cfg_env.workbench_size
        self.table_top_z = self.table_size[2] - self.workbench_size[2]

        self.ctrl_dt = self._cfg_env.ctrl_dt
        self.policy_dt = self._cfg_env.policy_dt
        self.sim_substeps = math.ceil(self._cfg_env.policy_dt / self._cfg_env.ctrl_dt)
        self.max_episode_length = math.ceil(
            self._cfg_env.episode_length_s / self.policy_dt
        )

        self.max_linear_speed = self._cfg_env.action_max_linear_speed
        self.max_angular_speed = self._cfg_env.action_max_angular_speed

        env_dict = self._cfg_env.env_specific_dict

        self.reward_scales = env_dict["reward_scales"]
        self.geometry = PegInHoleGeometry.from_dict(env_dict["geometry"])
        self.assets = env_dict["assets"]
        self.board_xy = list(env_dict["board_xy"])
        self.board_yaw = math.radians(env_dict["board_yaw_deg"])
        self.yaw_symmetry = env_dict["yaw_symmetry"]
        self.peg_friction = env_dict["friction"]["peg"]
        self.board_friction = env_dict["friction"]["board"]
        self.grasp_offset = list(env_dict["grasp_offset"])

        self.active_slots = list(env_dict["active_slots"])
        if not self.active_slots or not all(
            0 <= s < self.geometry.num_slots for s in self.active_slots
        ):
            raise ValueError(
                f"The `active_slots` {self.active_slots} must index the "
                f"{self.geometry.num_slots} slots of the board."
            )

        start = env_dict["start"]
        self.start_xy_range = start["xy_range"]
        self.start_height_range = list(start["height_range"])
        self.start_yaw_range = math.radians(start["yaw_range_deg"])

        noise = env_dict["hole_pose_noise"]
        self.hole_noise_enabled = noise["enabled"]
        self.hole_noise_xy = noise["xy"]
        self.hole_noise_yaw = math.radians(noise["yaw_deg"])

        self.ema_factor = env_dict["ema_factor"]
        self.tcp_hold_rot_gain = env_dict["tcp_hold_rot_gain"]
        workspace = env_dict["workspace"]
        self.workspace_xy = workspace["xy"]
        self.workspace_z = list(workspace["z"])
        self.workspace_exit_margin = workspace["exit_margin"]

        self.success_depth_ratio = env_dict["success_depth_ratio"]
        self.engage_depth = env_dict["engage_depth"]
        self.gate_xy_scale = env_dict["gate_xy_scale"]
        self.gate_yaw = math.radians(env_dict["gate_yaw_deg"])
        force = env_dict["force"]
        self.force_contact_threshold = force["contact_threshold"]
        self.force_limit = force["limit"]
        self.contact_penalty_ramp_steps = force["contact_penalty_ramp_steps"]
        self.tilt_limit = math.radians(env_dict["tilt_limit_deg"])
        self.terminate_on_success = env_dict["terminate_on_success"]

    @classmethod
    def get_default_env_specific_dict(cls) -> dict:
        return {
            "reward_scales": {
                # the keypoints of the insertion axis to their seated positions, in 3 scales
                "keypoints_baseline": 1.0,
                "keypoints_coarse": 1.0,
                "keypoints_fine": 1.0,
                # the depth, gated by the alignment, so descending next to the hole doesn't pay
                "depth_progress": 1.0,
                # the tips below the hole opening, aligned within the clearance
                "engaged": 1.0,
                # one-off bonus when the peg is inserted
                "success_bonus": 10.0,
                "contact_penalty": 0.05,
                # one-off penalty when exceeding the force limit, which ends the episode
                "force_limit_penalty": 50.0,
                "action_rate": 0.1,
            },
            "geometry": {
                "leg_size": [0.02, 0.02],
                "leg_length": 0.05,
                "leg_offsets": [[0.0, 0.0]],
                "handle_size": [0.02, 0.02, 0.04],
                "hole_depth": 0.05,
                # per side, one slot per value
                "clearances": [0.002, 0.0015, 0.001, 0.0005],
                "wall_thickness": 0.02,
                "slot_spacing": 0.08,
                "peg_mass": 0.05,
            },
            "assets": {
                "source": "procedural",
                "peg_urdf": None,
                "board_urdf": None,
            },
            # board origin on the table top
            "board_xy": [0.55, 0.0],
            "board_yaw_deg": 0.0,
            # the peg's yaw symmetry
            "yaw_symmetry": 4,
            "friction": {"peg": 0.5, "board": 0.5},
            "grasp_offset": [0.0, 0.0, 0.0],
            "active_slots": [1],
            # the peg tips' start pose w.r.t. the hole opening
            "start": {
                "xy_range": 0.015,
                "height_range": [0.03, 0.05],
                "yaw_range_deg": 45.0,
            },
            # constant offset of the observed hole pose per episode (the pose estimation error),
            # uniform in +-range, only with the domain randomization
            "hole_pose_noise": {"enabled": True, "xy": 0.002, "yaw_deg": 2.0},
            # weight of the new action in its exponential moving average, 1.0 disables it
            "ema_factor": 0.2,
            # P gain [1/s] of holding the roll and pitch of the TCP
            "tcp_hold_rot_gain": 5.0,
            # [m] the peg tips' box w.r.t. the hole opening the actions keep them in; leaving it
            # by more than `exit_margin` (pushed by the contacts) ends the episode
            "workspace": {"xy": 0.04, "z": [-0.06, 0.08], "exit_margin": 0.01},
            # inserted when deeper than this fraction of the hole depth, within the clearance
            "success_depth_ratio": 0.4,
            # [m] the tips below the opening that count as engaged
            "engage_depth": 0.005,
            # [m] XY gate scale exp(-d_xy / scale) and the yaw gate of the depth progress
            "gate_xy_scale": 0.002,
            "gate_yaw_deg": 10.0,
            # [N] read in the simulation, but not given to the actor: no compliant control yet
            "force": {
                "contact_threshold": 10.0,
                "limit": 60.0,
                # policy steps of ramping the contact penalty up from zero (~300 iterations),
                # or the policy learns to avoid the contact insertion needs
                "contact_penalty_ramp_steps": 14400,
            },
            "tilt_limit_deg": 7.0,
            "terminate_on_success": True,
        }

    def _observe_tcp_pose(self) -> th.Tensor:
        return self.manipulator.get_tcp_pose()

    def _observe_peg_pose(self) -> th.Tensor:
        return self.peg.get_pose()

    def _observe_tcp_wrench(self) -> th.Tensor:
        return self.manipulator.get_ft_wrench_compensated()

    def _observe_camera_scene(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_SCENE)

    def _observe_camera_tool_left(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_TOOL_LEFT)

    def _observe_camera_tool_right(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_TOOL_RIGHT)

    def _resolve_asset_paths(self) -> tuple[Path, Path]:
        match self.assets["source"]:
            case "procedural":
                return self.geometry.write_urdfs()
            case "files":
                paths = (
                    self._ASSETS_DIR / self.assets["peg_urdf"],
                    self._ASSETS_DIR / self.assets["board_urdf"],
                )
                for path in paths:
                    if not path.exists():
                        raise FileNotFoundError(f"Missing peg-in-hole asset: {path}")
                return paths
            case source:
                raise ValueError(f"Unknown peg-in-hole assets source: {source}")

    def _setup_scene(self, cfg: ExpConfig) -> None:
        self._scene.add_manipulator(cfg=cfg.robot_cfg)
        peg_urdf, board_urdf = self._resolve_asset_paths()

        board_quat = yaw_to_quat(th.tensor([self.board_yaw]))[0].tolist()
        p_board = ObjectProperties(
            dims=(),
            pose=(*self.board_xy, self.table_top_z, *board_quat),
            collision=True,
            fixed=True,
            color=(0.55, 0.55, 0.6),
            urdf_path=str(board_urdf),
            friction=self.board_friction,
        )
        self.board: BaseURDF = self._scene.add_entity(
            entity=ObjectType.URDF, properties=p_board
        )
        # placed in the gripper on the reset
        p_peg = ObjectProperties(
            dims=(),
            pose=(*self.board_xy, self.table_top_z + 0.3, 1.0, 0.0, 0.0, 0.0),
            collision=True,
            fixed=False,
            color=(0.85, 0.45, 0.05),
            urdf_path=str(peg_urdf),
            friction=self.peg_friction,
        )
        self.peg: BaseURDF = self._scene.add_entity(
            entity=ObjectType.URDF, properties=p_peg
        )
        self._scene.add_preview_camera(*self.PREVIEW_CAMERA_POSE)

    def _init_geometry_buffers(self) -> None:
        g, dev = self.geometry, self.device
        # hole openings (slot centers at the board top) in the world frame
        cos, sin = math.cos(self.board_yaw), math.sin(self.board_yaw)
        centers = g.slot_centers()
        slots_xy = th.stack(
            [
                self.board_xy[0] + cos * centers[:, 0] - sin * centers[:, 1],
                self.board_xy[1] + sin * centers[:, 0] + cos * centers[:, 1],
            ],
            dim=-1,
        )
        slots_z = th.full((g.num_slots, 1), self.table_top_z + g.hole_depth)
        self.slots_pos = th.cat([slots_xy, slots_z], dim=-1).to(dev)
        self.slots_clearance = th.tensor(g.clearances, device=dev)
        self.active_slots_t = th.tensor(self.active_slots, device=dev)

        # insertion point and keypoints of the peg in its frame
        self.peg_tips = g.leg_tips().to(dev)  # [L, 3]
        self.peg_insertion_point = g.tips_centroid().to(dev)  # [3]
        axis_z = th.linspace(0.0, g.leg_length, self._NUM_KEYPOINTS, device=dev)
        self.peg_keypoints = self.peg_insertion_point + th.stack(
            [th.zeros_like(axis_z), th.zeros_like(axis_z), axis_z], dim=-1
        )  # [K, 3]
        # their seated positions in the hole frame
        self.seated_depth = min(g.hole_depth, g.leg_length)
        self.hole_keypoints = self.peg_keypoints - self.peg_insertion_point
        self.hole_keypoints[:, 2] -= self.seated_depth
        self.success_depth = self.success_depth_ratio * g.hole_depth

        self.tcp_down_quat = th.tensor(self._TCP_DOWN_QUAT, device=dev)
        self.grasp_offset_t = th.tensor(self.grasp_offset, device=dev)
        self.workspace_low = th.tensor(
            [-self.workspace_xy, -self.workspace_xy, self.workspace_z[0]], device=dev
        )
        self.workspace_high = th.tensor(
            [self.workspace_xy, self.workspace_xy, self.workspace_z[1]], device=dev
        )

    def _init_reward_functions(self) -> None:
        # TODO(issue#141) simplify creation of the rewards_functions registry
        self.reward_functions, self.episode_sums = {}, {}
        dt = self.ctrl_dt * self.sim_substeps
        self.reward_scales = {
            name: scale * dt for name, scale in self.reward_scales.items()
        }
        for name in self.reward_scales:
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.num_envs,), device=self.device, dtype=th.float32
            )

    def _init_buffers(self) -> None:
        n, dev = self.num_envs, self.device

        def zeros(*shape: int) -> th.Tensor:
            return th.zeros((n, *shape), device=dev, dtype=th.float32)

        self.episode_length_buf = zeros()
        self.global_step = 0
        self.reset_buf = th.zeros(n, dtype=th.bool, device=dev)
        self.slot_idx = th.full((n,), self.active_slots[0], dtype=th.long, device=dev)
        self.hole_pos = self.slots_pos[self.slot_idx].clone()
        self.hole_yaw = th.full((n,), self.board_yaw, device=dev)
        self.clearance = self.slots_clearance[self.slot_idx].clone()
        # the observed hole pose error [dx, dy, dyaw], constant per episode
        self.hole_obs_offset = zeros(3)

        self.actions = zeros(self._ACTION_DIM)  # smoothed, applied
        self.last_raw_actions = zeros(self._ACTION_DIM)
        self.raw_actions = zeros(self._ACTION_DIM)
        self.last_tcp_pose = zeros(7)
        self.tcp_lin_vel = zeros(3)
        self.tcp_yaw_rate = zeros()

        # the true state of the peg w.r.t. its hole
        self.tip_rel = zeros(3)
        self.yaw_err = zeros()
        self.xy_err = zeros()
        self.depth = zeros()
        self.keypoint_dist = zeros()
        self.force_norm = zeros()
        self.tilt = zeros()

        self.success_buf = th.zeros(n, dtype=th.bool, device=dev)
        self.engaged_buf = th.zeros_like(self.success_buf)
        self.force_limit_buf = th.zeros_like(self.success_buf)
        self.out_of_workspace_buf = th.zeros_like(self.success_buf)
        self.tilted_buf = th.zeros_like(self.success_buf)
        self.max_force = zeros()

        self.extras = {}
        self.extras["observations"] = {}

    def _reset(self) -> ResetReturn:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=self.device))
        self._scene.update_state()
        return ResetReturn(self.get_observations(), self.extras)

    def _reset_idx_seeded(self, envs_idx: th.Tensor, seeds: Sequence[int]) -> None:
        self.reset_idx(envs_idx=envs_idx, seeds=seeds)

    def reset_idx(
        self, envs_idx: th.Tensor, seeds: Sequence[int] | None = None
    ) -> None:
        """
        Resets the `envs_idx` with the peg held above a random slot of the `active_slots`, or
        with the reproducible start of the `seeds` (then without the DR).
        """
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0
        randomize_domain = self._is_domain_randomized() and seeds is None

        u = rand_uniform(
            num=len(envs_idx), num_draws=5, device=self.device, seeds=seeds
        )
        slot_pick = th.clamp(
            (u[:, 4] * len(self.active_slots)).long(), max=len(self.active_slots) - 1
        )
        slot = self.active_slots_t[slot_pick]
        self.slot_idx[envs_idx] = slot
        self.hole_pos[envs_idx] = self.slots_pos[slot]
        self.hole_yaw[envs_idx] = self.board_yaw
        self.clearance[envs_idx] = self.slots_clearance[slot]

        peg_pose = self._start_peg_pose(envs_idx=envs_idx, u=u[:, :4])
        tcp_pose = self._peg_to_tcp_pose(peg_pose)
        self.manipulator.ctrl_gripper_close(envs_idx)
        self.manipulator.ctrl_go_to_home(envs_idx)
        self.manipulator.ctrl_reset_to_pose(
            pose=tcp_pose, open_gripper=False, envs_idx=envs_idx
        )
        # in the grasp of the reached TCP pose, as the IK may miss the target slightly and the
        # weld would pull the peg into the grasp it was attached in
        tcp_pose = self.manipulator.get_tcp_pose()[envs_idx]
        peg_pose = self._tcp_to_peg_pose(tcp_pose)
        self.peg.set_pose(pose=peg_pose, envs_idx=envs_idx)

        self.hole_obs_offset[envs_idx] = 0.0
        if randomize_domain and self.hole_noise_enabled:
            self._randomize_hole_obs_offset(envs_idx)

        # fill extras
        self.extras["episode"] = {
            "success_rate": self.success_buf[envs_idx].float(),
            "engaged_rate": self.engaged_buf[envs_idx].float(),
            "force_limit_rate": self.force_limit_buf[envs_idx].float(),
            "out_of_workspace_rate": self.out_of_workspace_buf[envs_idx].float(),
            "tilted_rate": self.tilted_buf[envs_idx].float(),
            "final_xy_err_mm": 1000 * self.xy_err[envs_idx],
            "final_depth_mm": 1000 * self.depth[envs_idx],
            "max_force_n": self.max_force[envs_idx],
        }
        self.success_buf[envs_idx] = False
        self.engaged_buf[envs_idx] = False
        self.force_limit_buf[envs_idx] = False
        self.out_of_workspace_buf[envs_idx] = False
        self.tilted_buf[envs_idx] = False
        self.max_force[envs_idx] = 0.0
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg_env.episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.actions[envs_idx] = 0.0
        self.raw_actions[envs_idx] = 0.0
        self.last_raw_actions[envs_idx] = 0.0
        self.tcp_lin_vel[envs_idx] = 0.0
        self.tcp_yaw_rate[envs_idx] = 0.0
        self.last_tcp_pose[envs_idx] = tcp_pose
        self._update_insertion_state(peg_pose=peg_pose, envs_idx=envs_idx)

        if not randomize_domain:
            return
        for rt in self._scene.get_available_randomizations():
            self._scene.randomize_domain(rand_type=rt, env_idx=envs_idx)

    def _start_peg_pose(self, envs_idx: th.Tensor, u: th.Tensor) -> th.Tensor:
        """[len(envs_idx), 7] peg poses with the tips above the hole, from the [len(envs_idx), 4] draws."""
        dx = (2 * u[:, 0] - 1) * self.start_xy_range
        dy = (2 * u[:, 1] - 1) * self.start_xy_range
        low, high = self.start_height_range
        height = low + u[:, 2] * (high - low)
        dyaw = (2 * u[:, 3] - 1) * self.start_yaw_range

        hole_yaw = self.hole_yaw[envs_idx]
        peg_quat = yaw_to_quat(hole_yaw + dyaw)
        tip = self.hole_pos[envs_idx] + th.stack(
            [
                th.cos(hole_yaw) * dx - th.sin(hole_yaw) * dy,
                th.sin(hole_yaw) * dx + th.cos(hole_yaw) * dy,
                height,
            ],
            dim=-1,
        )
        insertion_point = self.peg_insertion_point.expand(len(envs_idx), 3)
        peg_pos = tip - transform_by_quat(insertion_point, peg_quat)
        return th.cat([peg_pos, peg_quat], dim=-1)

    def _peg_to_tcp_pose(self, peg_pose: th.Tensor) -> th.Tensor:
        """The TCP pose holding the peg at the `peg_pose` in the nominal grasp."""
        n = peg_pose.shape[0]
        tcp_quat = transform_quat_by_quat(
            peg_pose[:, 3:], self.tcp_down_quat.expand(n, 4)
        )
        offset = transform_by_quat(self.grasp_offset_t.expand(n, 3), tcp_quat)
        return th.cat([peg_pose[:, :3] - offset, tcp_quat], dim=-1)

    def _tcp_to_peg_pose(self, tcp_pose: th.Tensor) -> th.Tensor:
        """The peg pose held by the TCP at the `tcp_pose` in the nominal grasp."""
        n = tcp_pose.shape[0]
        tcp_quat = tcp_pose[:, 3:]
        peg_pos = tcp_pose[:, :3] + transform_by_quat(
            self.grasp_offset_t.expand(n, 3), tcp_quat
        )
        # the down rotation is its own inverse (180 deg about X)
        peg_quat = transform_quat_by_quat(tcp_quat, self.tcp_down_quat.expand(n, 4))
        return th.cat([peg_pos, peg_quat], dim=-1)

    def _randomize_hole_obs_offset(self, envs_idx: th.Tensor) -> None:
        u = th.rand(len(envs_idx), 3, device=self.device) * 2 - 1
        scale = th.tensor(
            [self.hole_noise_xy, self.hole_noise_xy, self.hole_noise_yaw],
            device=self.device,
        )
        self.hole_obs_offset[envs_idx] = u * scale

    def _step(self, actions: th.Tensor) -> StepReturn:
        self.episode_length_buf += 1
        self.global_step += 1

        self.raw_actions = th.clamp(actions, min=-1.0, max=1.0)
        self.actions = (
            self.ema_factor * self.raw_actions + (1 - self.ema_factor) * self.actions
        )
        tcp_twist = self._action_to_tcp_twist(self.actions)

        self._scene.pre_step()
        self.manipulator.ctrl_apply_vel_action(tcp_twist, open_gripper=False)
        self._scene.step()

        self._update_tcp_velocity()
        self._update_insertion_state(peg_pose=self.peg.get_pose())
        self.success_buf = (self.depth >= self.success_depth) & (
            self.xy_err < self.clearance
        )
        self.engaged_buf |= (self.depth >= self.engage_depth) & (
            self.xy_err < self.clearance
        )
        self.force_limit_buf = self.force_norm > self.force_limit
        self.max_force = th.maximum(self.max_force, self.force_norm)
        # copies, the buffers are cleared in place by the reset below
        self.extras["success"] = self.success_buf.clone()

        # compute reward based on task, before the reset replaces the terminal state
        reward = th.zeros(self.num_envs, device=self.device, dtype=th.float32)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew
        self.last_raw_actions = self.raw_actions.clone()

        # check env termination
        env_reset_idx = self._is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)
            self._obs_cache_clear()

        obs = self.get_observations()
        dones = self.reset_buf
        return StepReturn(obs, reward, dones, self.extras)

    def _action_to_tcp_twist(self, actions: th.Tensor) -> th.Tensor:
        """
        Expands the [N, 4] action into the normalized [N, 6] TCP twist; the roll and pitch come
        from a P controller holding the TCP vertical at its current yaw.
        """
        tcp_pose = self.manipulator.get_tcp_pose()
        linear = self._limit_action_to_workspace(actions[:, :3], tcp_pose=tcp_pose)
        upright = yaw_to_quat(self._tcp_yaw(tcp_pose[:, 3:]))
        hold_quat = transform_quat_by_quat(
            upright, self.tcp_down_quat.expand_as(upright)
        )
        rot_err = quat_to_rotvec_error(hold_quat, tcp_pose[:, 3:])

        tcp_twist = th.zeros(self.num_envs, 6, device=self.device, dtype=th.float32)
        tcp_twist[:, :3] = linear
        tcp_twist[:, 3:5] = (
            self.tcp_hold_rot_gain * rot_err[:, :2] / self.max_angular_speed
        )
        tcp_twist[:, 5] = actions[:, 3]
        return th.clamp(tcp_twist, min=-1.0, max=1.0)

    def _limit_action_to_workspace(
        self, linear: th.Tensor, tcp_pose: th.Tensor
    ) -> th.Tensor:
        """Scales down the linear action, so the peg tips won't leave the workspace within the step."""
        max_step = self.max_linear_speed * self.policy_dt
        tip = self._tips_in_hole_frame(self._tcp_to_peg_pose(tcp_pose)).mean(dim=1)
        # the workspace box is of the hole frame and the action of the world, so the action is
        # limited in the hole frame and rotated back
        local = th.cat(
            [self._rotate_xy(linear[:, :2], -self.hole_yaw), linear[:, 2:]], dim=-1
        )
        # outside of the workspace only the moves back inside are allowed
        a_min = th.clamp((self.workspace_low - tip) / max_step, max=0.0)
        a_max = th.clamp((self.workspace_high - tip) / max_step, min=0.0)
        local = th.clamp(local, min=a_min, max=a_max)
        return th.cat(
            [self._rotate_xy(local[:, :2], self.hole_yaw), local[:, 2:]], dim=-1
        )

    def _update_tcp_velocity(self) -> None:
        tcp_pose = self.manipulator.get_tcp_pose()
        self.tcp_lin_vel = (
            tcp_pose[:, :3] - self.last_tcp_pose[:, :3]
        ) / self.policy_dt
        dyaw = self._tcp_yaw(tcp_pose[:, 3:]) - self._tcp_yaw(self.last_tcp_pose[:, 3:])
        self.tcp_yaw_rate = wrap_symmetric(dyaw, symmetry=1) / self.policy_dt
        self.last_tcp_pose = tcp_pose.clone()

    def _update_insertion_state(
        self, peg_pose: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> None:
        """The true state w.r.t. the hole: the rewards, the success and the critic read it."""
        idx = slice(None) if envs_idx is None else envs_idx
        tips = self._tips_in_hole_frame(peg_pose, envs_idx=envs_idx)  # [n, L, 3]
        self.tip_rel[idx] = tips.mean(dim=1)
        self.xy_err[idx] = th.linalg.vector_norm(self.tip_rel[idx, :2], dim=-1)
        # the depth of the shallowest leg below the opening
        self.depth[idx] = -tips[..., 2].max(dim=1).values
        self.yaw_err[idx] = wrap_symmetric(
            quat_to_yaw(peg_pose[:, 3:]) - self.hole_yaw[idx], self.yaw_symmetry
        )
        self.keypoint_dist[idx] = self._keypoint_distance(peg_pose, envs_idx=envs_idx)

        tcp_pose = self.manipulator.get_tcp_pose()[idx]
        tcp_z = transform_by_quat(
            th.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(tcp_pose), 3),
            tcp_pose[:, 3:],
        )
        self.tilt[idx] = th.acos(th.clamp(-tcp_z[:, 2], min=-1.0, max=1.0))
        if envs_idx is None:
            force = self.get_modality_observation(Modality.TCP_WRENCH)[:, :3]
            self.force_norm = th.linalg.vector_norm(force, dim=-1)
        else:
            self.force_norm[idx] = 0.0

    def _tips_in_hole_frame(
        self, peg_pose: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> th.Tensor:
        """[n, L, 3] leg tips of the `peg_pose` in the hole frame of the `envs_idx` (all if None)."""
        idx = slice(None) if envs_idx is None else envs_idx
        n, num_legs = peg_pose.shape[0], self.peg_tips.shape[0]
        quat = peg_pose[:, None, 3:].expand(n, num_legs, 4)
        tips = peg_pose[:, None, :3] + transform_by_quat(
            self.peg_tips.expand(n, num_legs, 3), quat
        )
        rel = tips - self.hole_pos[idx].unsqueeze(1)
        hole_yaw = self.hole_yaw[idx].unsqueeze(1).expand(n, num_legs)
        return th.cat([self._rotate_xy(rel[..., :2], -hole_yaw), rel[..., 2:]], dim=-1)

    def _keypoint_distance(
        self, peg_pose: th.Tensor, envs_idx: th.Tensor | None = None
    ) -> th.Tensor:
        """Mean distance of the insertion axis keypoints to their seated positions."""
        idx = slice(None) if envs_idx is None else envs_idx
        n, k = peg_pose.shape[0], self.peg_keypoints.shape[0]
        quat = peg_pose[:, None, 3:].expand(n, k, 4)
        points = peg_pose[:, None, :3] + transform_by_quat(
            self.peg_keypoints.expand(n, k, 3), quat
        )
        rel = points - self.hole_pos[idx].unsqueeze(1)
        hole_yaw = self.hole_yaw[idx].unsqueeze(1).expand(n, k)
        local = th.cat([self._rotate_xy(rel[..., :2], -hole_yaw), rel[..., 2:]], dim=-1)
        return th.linalg.vector_norm(local - self.hole_keypoints, dim=-1).mean(dim=-1)

    def _is_episode_complete(self) -> th.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length
        margin = self.workspace_exit_margin
        self.out_of_workspace_buf = (
            (self.tip_rel < self.workspace_low - margin)
            | (self.tip_rel > self.workspace_high + margin)
        ).any(dim=-1)
        self.tilted_buf = self.tilt > self.tilt_limit
        self.extras["force_limit"] = self.force_limit_buf.clone()

        self.reset_buf = (
            time_out_buf
            | self.force_limit_buf
            | self.out_of_workspace_buf
            | self.tilted_buf
        )
        if self.terminate_on_success:
            self.reset_buf |= self.success_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = th.zeros_like(
            self.reset_buf, device=self.device, dtype=th.float32
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> TensorDict:
        """The actor's observations, plus the critic's privileged state as its own group."""
        obs = super().get_observations()
        obs[self.PRIVILEGED_OBS_GROUP] = self._build_privileged_observations()
        return obs

    def _build_agent_observations(self, obs: TensorDict) -> th.Tensor:
        """
        What the real robot gives: the peg (from the TCP pose and the nominal grasp) w.r.t. the
        hole pose estimate, the TCP velocity and the previous action. No noise on the
        proprioception, only on the hole pose.
        """
        peg_pose = self._tcp_to_peg_pose(obs[Modality.TCP_POSE])
        tip = self._tips_in_hole_frame(peg_pose).mean(dim=1)
        # the hole is believed to be at the offset pose, so the tip seems shifted the other way
        tip_est = th.cat(
            [
                self._rotate_xy(
                    tip[:, :2] - self.hole_obs_offset[:, :2],
                    -self.hole_obs_offset[:, 2],
                ),
                tip[:, 2:],
            ],
            dim=-1,
        )
        yaw_est = wrap_symmetric(
            quat_to_yaw(peg_pose[:, 3:]) - self.hole_yaw - self.hole_obs_offset[:, 2],
            self.yaw_symmetry,
        )
        obs_components = [
            tip_est,  # insertion point w.r.t. the estimated hole opening
            self._yaw_features(yaw_est),  # yaw error w.r.t. the estimated hole
            self.tcp_lin_vel,
            self.tcp_yaw_rate[:, None],
            # with the velocity, the previous action also exposes the servo tracking error
            self.actions,
        ]
        return th.cat(obs_components, dim=-1)

    def _build_privileged_observations(self) -> th.Tensor:
        """The true state for the critic, which never runs on the robot."""
        obs_components = [
            self.tip_rel,
            self._yaw_features(self.yaw_err),
            self.depth[:, None],
            self.xy_err[:, None],
            self.clearance[:, None],
            self.hole_obs_offset,
            self.force_norm[:, None],
        ]
        return th.cat(obs_components, dim=-1)

    def _yaw_features(self, yaw_err: th.Tensor) -> th.Tensor:
        """The symmetric yaw error as a continuous [N, 2] feature."""
        k = self.yaw_symmetry
        return th.stack([th.sin(k * yaw_err), th.cos(k * yaw_err)], dim=-1)

    def _tcp_yaw(self, tcp_quat: th.Tensor) -> th.Tensor:
        """Yaw of the downward facing TCP, i.e. of the peg it holds."""
        return quat_to_yaw(
            transform_quat_by_quat(tcp_quat, self.tcp_down_quat.expand_as(tcp_quat))
        )

    @staticmethod
    def _rotate_xy(xy: th.Tensor, yaw: th.Tensor) -> th.Tensor:
        """Rotates the [..., 2] `xy` vectors by the [...] `yaw` angles."""
        cos, sin = th.cos(yaw), th.sin(yaw)
        x, y = xy[..., 0], xy[..., 1]
        return th.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)

    def scripted_action(self, gain: float = 5.0) -> th.Tensor:
        """
        [N, 4] actions of an oracle policy reading the true state, for the smoke runs: P
        controllers of `gain` [1/s] aligning the peg above the opening, then inserting it.
        """
        aligned = (self.xy_err < 0.5 * self.clearance) & (
            self.yaw_err.abs() < math.radians(2.0)
        )
        inside = self.depth > 0.0
        hover_z = th.full_like(self.depth, 0.005)
        target_z = th.where(aligned | inside, -self.seated_depth, hover_z)

        xy_world = self._rotate_xy(self.tip_rel[:, :2], self.hole_yaw)
        actions = th.zeros(self.num_envs, self._ACTION_DIM, device=self.device)
        actions[:, :2] = -gain * xy_world / self.max_linear_speed
        actions[:, 2] = gain * (target_z - self.tip_rel[:, 2]) / self.max_linear_speed
        actions[:, 3] = -gain * self.yaw_err / self.max_angular_speed
        return th.clamp(actions, min=-1.0, max=1.0)

    def _keypoint_squash(self, name: str) -> th.Tensor:
        a, b = self._KEYPOINT_SQUASH[name]
        d = self.keypoint_dist
        return 1 / (th.exp(a * d) + b + th.exp(-a * d))

    def _reward_keypoints_baseline(self) -> th.Tensor:
        return self._keypoint_squash("keypoints_baseline")

    def _reward_keypoints_coarse(self) -> th.Tensor:
        return self._keypoint_squash("keypoints_coarse")

    def _reward_keypoints_fine(self) -> th.Tensor:
        return self._keypoint_squash("keypoints_fine")

    def _reward_depth_progress(self) -> th.Tensor:
        gate_xy = th.exp(-self.xy_err / self.gate_xy_scale)
        gate_yaw = (self.yaw_err.abs() < self.gate_yaw).float()
        progress = th.clamp(self.depth / self.success_depth, min=0.0, max=1.0)
        return gate_xy * gate_yaw * progress

    def _reward_engaged(self) -> th.Tensor:
        return (
            (self.depth >= self.engage_depth) & (self.xy_err < self.clearance)
        ).float()

    def _reward_success_bonus(self) -> th.Tensor:
        # divided by dt, so the scale is the bonus value of a single successful step
        return self.success_buf.float() / self.policy_dt

    def _reward_contact_penalty(self) -> th.Tensor:
        ramp = (
            min(1.0, self.global_step / self.contact_penalty_ramp_steps)
            if self.contact_penalty_ramp_steps > 0
            else 1.0
        )
        return -ramp * th.relu(self.force_norm - self.force_contact_threshold)

    def _reward_force_limit_penalty(self) -> th.Tensor:
        # divided by dt, so the scale is the penalty of the episode ending step
        return -self.force_limit_buf.float() / self.policy_dt

    def _reward_action_rate(self) -> th.Tensor:
        return -(self.raw_actions - self.last_raw_actions).square().sum(dim=-1)
