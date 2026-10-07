import math
from pathlib import Path

import numpy as np
import torch as th
import trimesh
from tensordict import TensorDict

from aegis_gym.aux.geom import (
    check_points_in_polygon,
    quat_to_rotvec_error,
    quat_to_z_euler,
    quat_to_zrot,
)
from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import CameraName, ExpConfig
from aegis_gym.envs.base_env import BaseEnv, Modality, ResetReturn, StepReturn
from aegis_gym.envs.manipulator import BaseManipulator
from aegis_gym.envs.objects import BaseMesh, BaseURDF, ObjectProperties, ObjectType

from .registry import register_env
from .scene import BaseScene


@register_env("push_t")
class PushTEnv(BaseEnv):
    DEFAULT_MODALITIES = frozenset({Modality.TCP_POSE, Modality.OBJECT_POSE})
    DEFAULT_EPISODE_LENGTH_S = 30.0

    _ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "push_t"
    _TEE_STL_PATH = _ASSETS_DIR / "T_shape.stl"
    _TEE_URDF_PATH = _ASSETS_DIR / "T_shape.urdf"
    _SPAWN_CLEARANCE = 0.02
    _GOAL_MARKER_Z_SCALE = 0.0025
    _GOAL_MARKER_LIFT = 1e-3
    _TEE_DIMS = (0.2, 0.2, 0.04)
    # footprint of the tee in its frame as boxes (center xy, half size xy), see T_shape.urdf
    _TEE_FOOTPRINT_BOXES = (
        ((0.0, 0.0), (0.1, 0.025)),  # crossbar
        ((0.0, -0.1), (0.025, 0.075)),  # stem
    )
    # the tee origin dropped this much below the table top means it fell off
    _TEE_FALL_Z = 0.02
    # planar TCP velocity [vx, vy]; height and orientation are held by the env
    _ACTION_DIM = 2

    def __init__(self, scene: BaseScene, cfg: ExpConfig):
        super().__init__(scene=scene, cfg=cfg)
        self._extract_config()
        self._observation_fns = {
            Modality.TCP_POSE: self._observe_tcp_pose,
            Modality.OBJECT_POSE: self._observe_object_pose,
            Modality.CAMERA_SCENE_RGB: self._observe_camera_scene,
            Modality.CAMERA_TOOL_LEFT_RGB: self._observe_camera_tool_left,
            Modality.CAMERA_TOOL_RIGHT_RGB: self._observe_camera_tool_right,
        }

        self._setup_scene(cfg=cfg)
        self._scene.build()
        self.manipulator: BaseManipulator = self._scene.get_manipulator()

        self._build_tee_canonical_mask(mesh_path=str(self._TEE_STL_PATH))
        self._build_tee_footprint_moments()
        self._validate_tcp_workspace()

        self._init_reward_functions()
        self._init_buffers()
        self._build_goal_transform()
        self.reset()

    def _extract_config(self) -> None:
        # TODO(issue##117) redesign the whole camera preview system
        self.show_cameras_gui = self._cfg_env.visualize_camera
        self.show_cell = self._cfg_env.visualize_cell

        self.num_obs = self._cfg_env.num_obs
        self.num_privileged_obs = None
        self.num_actions = self._ACTION_DIM
        if self._cfg_env.num_actions != self._ACTION_DIM:
            get_logger("PushTEnv").warning(
                f"Ignoring `num_actions={self._cfg_env.num_actions}`: "
                f"the push_t env uses planar XY actions (num_actions={self._ACTION_DIM})."
            )
        self.image_width = self._cfg_env.image_resolution[0]
        self.image_height = self._cfg_env.image_resolution[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.cameras_setup = self._cfg_env.cameras_setup
        self.table_size = self._cfg_env.table_size
        self.workbench_size = self._cfg_env.workbench_size
        self.table_top_z = self.table_size[2] - self.workbench_size[2]
        # table top in XY, placed next to the workbench as in the scene
        self.table_x = (
            self.workbench_size[0] / 2,
            self.workbench_size[0] / 2 + self.table_size[0],
        )
        self.table_y = (-self.table_size[1] / 2, self.table_size[1] / 2)

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

        self.success_thresh = env_dict["success_intersection_thresh"]
        self.pose_dist_scale = env_dict["pose_dist_scale"]
        self.tee_friction = env_dict["friction"]
        self.goal_offset_xy = list(env_dict["goal_offset"])
        self.goal_z_rot = math.radians(env_dict["goal_z_rot_deg"])
        goal_rand = env_dict["goal_randomization"]
        self.goal_rand_enabled = goal_rand["enabled"]
        self.goal_rand_x = list(goal_rand["x_range"])
        self.goal_rand_y = list(goal_rand["y_range"])
        self.goal_rand_z_rot = [math.radians(a) for a in goal_rand["z_rot_range_deg"]]
        self.spawnbox_xlength = env_dict["spawnbox_xlength"]
        self.spawnbox_ylength = env_dict["spawnbox_ylength"]
        self.spawnbox_xoffset = env_dict["spawnbox_xoffset"]
        self.spawnbox_yoffset = env_dict["spawnbox_yoffset"]
        self.mask_resolution = env_dict["mask_resolution"]
        self.mask_half_width = env_dict["mask_half_width"]
        self.tcp_height = (
            self.table_top_z + env_dict["tcp_height_ratio"] * self._TEE_DIMS[2]
        )
        self.tcp_start_xy = list(env_dict["tcp_start_xy"])
        self.tcp_quat = list(env_dict["tcp_quat"])
        self.tcp_hold_z_gain = env_dict["tcp_hold_z_gain"]
        self.tcp_hold_rot_gain = env_dict["tcp_hold_rot_gain"]
        self.tcp_workspace_x = list(env_dict["tcp_workspace_x"])
        self.tcp_workspace_y = list(env_dict["tcp_workspace_y"])
        self.terminate_on_success = env_dict["terminate_on_success"]

    @classmethod
    def get_default_env_specific_dict(cls) -> dict:
        return {
            "reward_scales": {
                # rewards the decrease of the pose distance
                "pose_progress": 10.0,
                # squashed pose distance, rewards being close (also when idle)
                "pose_alignment": 0.0,
                # rewards the decrease of the TCP distance to the tee footprint
                "tcp_approach": 1.0,
                # one-off bonus when the tee reaches the goal
                "success_bonus": 10.0,
            },
            "success_intersection_thresh": 0.90,
            "terminate_on_success": True,
            # [m] scale of the distances in the progress rewards
            "pose_dist_scale": 0.05,
            "friction": 0.4,
            "goal_offset": [0.47, 0.0],
            "goal_z_rot_deg": 0.0,
            # per episode goal pose, sampled as [min, max] offsets from `goal_offset` and
            # `goal_z_rot_deg`; the tee spawn box stays w.r.t. the nominal goal
            "goal_randomization": {
                "enabled": False,
                "x_range": [-0.05, 0.05],
                "y_range": [-0.1, 0.1],
                "z_rot_range_deg": [-180.0, 180.0],
            },
            # tee origin spawn box w.r.t. the goal; with any yaw the tee stays within the TCP
            # workspace sideways and >= 4 cm away from the TCP start pose
            "spawnbox_xlength": 0.14,
            "spawnbox_ylength": 0.34,
            "spawnbox_xoffset": -0.05,
            "spawnbox_yoffset": -0.17,
            "mask_resolution": 64,
            "mask_half_width": 0.15,
            # TCP height above the table as a fraction of the tee height
            "tcp_height_ratio": 0.5,
            "tcp_start_xy": [0.2, 0.0],
            # downward facing TCP (w, x, y, z), kept fixed during the episode
            "tcp_quat": [0.0, 1.0, 0.0, 0.0],
            # P gains [1/s] of holding the TCP height and orientation
            "tcp_hold_z_gain": 5.0,
            "tcp_hold_rot_gain": 5.0,
            # [min, max] TCP position; the arm can't hold the height near its reach limit
            "tcp_workspace_x": [0.15, 0.7],
            "tcp_workspace_y": [-0.35, 0.35],
        }

    def _observe_tcp_pose(self) -> th.Tensor:
        return self.manipulator.get_tcp_pose()

    def _observe_object_pose(self) -> th.Tensor:
        return self.object.get_pose()

    def _observe_camera_scene(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_SCENE)

    def _observe_camera_tool_left(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_TOOL_LEFT)

    def _observe_camera_tool_right(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_TOOL_RIGHT)

    def _setup_scene(self, cfg: ExpConfig) -> None:
        self._scene.add_manipulator(cfg=cfg.robot_cfg)

        goal_quat = (
            math.cos(self.goal_z_rot / 2),
            0.0,
            0.0,
            math.sin(self.goal_z_rot / 2),
        )
        self.goal_pose_tuple = (
            self.goal_offset_xy[0],
            self.goal_offset_xy[1],
            self.table_top_z,
            *goal_quat,
        )

        p_tee = ObjectProperties(
            dims=self._TEE_DIMS,
            pose=self.goal_pose_tuple,
            collision=True,
            fixed=False,
            color=(0.02, 0.02, 0.02),
            urdf_path=str(self._TEE_URDF_PATH),
            friction=self.tee_friction,
        )
        self.object: BaseURDF = self._scene.add_entity(
            entity=ObjectType.URDF, properties=p_tee
        )

        goal_marker_pose = (
            self.goal_pose_tuple[0],
            self.goal_pose_tuple[1],
            self.goal_pose_tuple[2] + self._GOAL_MARKER_LIFT,
            *self.goal_pose_tuple[3:],
        )
        p_goal_marker = ObjectProperties(
            dims=self._TEE_DIMS,
            pose=goal_marker_pose,
            collision=False,
            fixed=True,
            color=(0.8, 0.0, 0.0),
            mesh_path=str(self._TEE_STL_PATH),
            scale=(1.0, 1.0, self._GOAL_MARKER_Z_SCALE),
        )
        self.goal_marker: BaseMesh = self._scene.add_entity(
            entity=ObjectType.MESH, properties=p_goal_marker
        )

    def _validate_tcp_workspace(self) -> None:
        """Warns if the goal or the tee spawn box are outside the TCP workspace (virtual fence)."""
        x0 = self.goal_offset_xy[0] + self.spawnbox_xoffset
        y0 = self.goal_offset_xy[1] + self.spawnbox_yoffset
        gx, gy = self.goal_offset_xy
        goals = [(gx, gy)]
        if self.goal_rand_enabled:
            goals = [
                (gx + dx, gy + dy) for dx in self.goal_rand_x for dy in self.goal_rand_y
            ]
        points = {
            "goal": goals,
            "tee spawn box": [
                (x0, y0),
                (x0 + self.spawnbox_xlength, y0 + self.spawnbox_ylength),
            ],
        }
        (x_min, x_max), (y_min, y_max) = self.tcp_workspace_x, self.tcp_workspace_y
        for name, pts in points.items():
            if any(not (x_min <= x <= x_max and y_min <= y <= y_max) for x, y in pts):
                get_logger("PushTEnv").warning(
                    f"The {name} lies outside of the TCP workspace "
                    f"x={self.tcp_workspace_x}, y={self.tcp_workspace_y}; it may be unreachable."
                )

    def _build_tee_canonical_mask(self, mesh_path: str) -> None:
        mesh = trimesh.load(mesh_path)
        z_mid = float(mesh.bounds[:, 2].mean())
        section = mesh.section(plane_origin=[0, 0, z_mid], plane_normal=[0, 0, 1])
        polygon = section.discrete[0][:, :2]

        radius = float(np.linalg.norm(polygon, axis=1).max())
        # widen the grid if the configured half width cannot cover the footprint
        self.mask_half_width = max(self.mask_half_width, radius * 1.15)
        res, half_width = self.mask_resolution, self.mask_half_width

        lin = (np.arange(res, dtype=np.float64) + 0.5) / res * (
            2 * half_width
        ) - half_width
        xx, yy = np.meshgrid(lin, lin, indexing="ij")
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        mask_np = check_points_in_polygon(grid_pts, polygon).reshape(res, res)

        self._px_per_meter = res / (2 * half_width)
        self._tee_mask = th.from_numpy(mask_np).to(device=self.device)
        self._tee_mask_flat = self._tee_mask.reshape(-1)
        self._goal_area = float(mask_np.sum())

        homo = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())], axis=0)
        self._homo_uv = th.tensor(homo, dtype=th.float32, device=self.device)

    def _build_tee_footprint_moments(self) -> None:
        """
        Helper function for calculating rewards.
        Centroid (in the tee frame) and squared radius of gyration of the tee footprint.
        """
        areas, centers, polar = [], [], []
        for (cx, cy), (hx, hy) in self._TEE_FOOTPRINT_BOXES:
            area = 4 * hx * hy
            areas.append(area)
            centers.append((cx, cy))
            # polar moment of the box around its own center
            polar.append(area * ((2 * hx) ** 2 + (2 * hy) ** 2) / 12)
        total_area = sum(areas)
        centroid = [
            sum(a * c[i] for a, c in zip(areas, centers, strict=True)) / total_area
            for i in range(2)
        ]
        # parallel axis theorem to move the moments into the footprint centroid
        polar_centroid = sum(
            j + a * ((c[0] - centroid[0]) ** 2 + (c[1] - centroid[1]) ** 2)
            for j, a, c in zip(polar, areas, centers, strict=True)
        )
        self._tee_centroid = th.tensor(centroid, device=self.device)
        self._tee_gyration_sq = polar_centroid / total_area

    def _build_goal_transform(self, envs_idx: th.Tensor | None = None) -> None:
        """Updates the per env world -> goal frame transforms from `goal_pose`."""
        if envs_idx is None:
            self._world_to_goal_trans = th.zeros(
                self.num_envs, 3, 3, device=self.device
            )
            envs_idx = th.arange(self.num_envs, device=self.device)
        goal_pose = self.goal_pose[envs_idx]
        goal_trans = quat_to_zrot(goal_pose[:, 3:], device=self.device)
        goal_trans[:, 0:2, 2] = goal_pose[:, 0:2]
        self._world_to_goal_trans[envs_idx] = th.linalg.inv(goal_trans)

    def _init_reward_functions(self) -> None:
        # TODO(issue#141) simplify creation of the rewards_functions registry
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.ctrl_dt * self.sim_substeps
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.num_envs,), device=self.device, dtype=th.float32
            )

    def _init_buffers(self) -> None:
        self.episode_length_buf = th.zeros(
            (self.num_envs,), device=self.device, dtype=th.float32
        )
        self.reset_buf = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        self.goal_pose = th.tensor(
            self.goal_pose_tuple, device=self.device, dtype=th.float32
        ).repeat(self.num_envs, 1)
        self.goal_yaw = th.full(
            (self.num_envs,), self.goal_z_rot, device=self.device, dtype=th.float32
        )
        self.tcp_start_pose = th.tensor(
            [*self.tcp_start_xy, self.tcp_height, *self.tcp_quat],
            device=self.device,
            dtype=th.float32,
        ).repeat(self.num_envs, 1)
        self.intersection_ratio = th.zeros(
            self.num_envs, device=self.device, dtype=th.float32
        )
        self.pose_dist = th.zeros(self.num_envs, device=self.device, dtype=th.float32)
        self.last_pose_dist = th.zeros_like(self.pose_dist)
        self.tcp_dist = th.zeros_like(self.pose_dist)
        self.last_tcp_dist = th.zeros_like(self.pose_dist)
        self.success_buf = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        self.tee_off_table_buf = th.zeros_like(self.success_buf)

        # tee [x, y, yaw] at the previous step, for its velocity observation
        self.last_tee_xy_yaw = th.zeros(self.num_envs, 3, device=self.device)
        self.extras = {}
        self.extras["observations"] = {}

    def _reset(self) -> ResetReturn:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=self.device))
        self._scene.update_state()
        return ResetReturn(self.get_observations(), self.extras)

    def reset_idx(self, envs_idx: th.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        self.manipulator.ctrl_gripper_close(envs_idx)
        self.manipulator.ctrl_go_to_home(envs_idx)
        self.manipulator.ctrl_reset_to_pose(
            pose=self.tcp_start_pose[envs_idx], open_gripper=False, envs_idx=envs_idx
        )

        if self.goal_rand_enabled:
            self._randomize_goal(envs_idx=envs_idx)

        tee_pose = self._get_random_tee_pose(envs_idx=envs_idx)
        self.object.set_pose(pose=tee_pose, envs_idx=envs_idx)
        self.last_pose_dist[envs_idx] = self._tee_to_goal_pose_distance()[envs_idx]
        self.last_tcp_dist[envs_idx] = self._tcp_to_tee_footprint_distance()[envs_idx]
        self.last_tee_xy_yaw[envs_idx] = self._get_tee_xy_yaw()[envs_idx]

        # fill extras
        self.extras["episode"] = {
            "success_rate": self.success_buf[envs_idx].float().mean().item(),
            "tee_off_table_rate": self.tee_off_table_buf[envs_idx]
            .float()
            .mean()
            .item(),
        }
        self.success_buf[envs_idx] = False
        self.tee_off_table_buf[envs_idx] = False
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg_env.episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0

        if not self._cfg_dr.enabled:
            return
        for rt in self._scene.get_available_randomizations():
            self._scene.randomize_domain(rand_type=rt, env_idx=envs_idx)

    def _randomize_goal(self, envs_idx: th.Tensor) -> None:
        num_reset = len(envs_idx)

        def uniform(low_high: list[float]) -> th.Tensor:
            low, high = low_high
            return th.rand(num_reset, device=self.device) * (high - low) + low

        goal_x = self.goal_offset_xy[0] + uniform(self.goal_rand_x)
        goal_y = self.goal_offset_xy[1] + uniform(self.goal_rand_y)
        goal_yaw = th.remainder(
            self.goal_z_rot + uniform(self.goal_rand_z_rot), 2 * math.pi
        )

        self.goal_yaw[envs_idx] = goal_yaw
        self.goal_pose[envs_idx, 0] = goal_x
        self.goal_pose[envs_idx, 1] = goal_y
        self.goal_pose[envs_idx, 3] = th.cos(goal_yaw / 2)
        self.goal_pose[envs_idx, 6] = th.sin(goal_yaw / 2)
        self._build_goal_transform(envs_idx=envs_idx)

        marker_pose = self.goal_pose[envs_idx].clone()
        marker_pose[:, 2] += self._GOAL_MARKER_LIFT
        self.goal_marker.set_pose(pose=marker_pose, envs_idx=envs_idx)

    def _get_random_tee_pose(self, envs_idx: th.Tensor) -> th.Tensor:
        num_reset = len(envs_idx)

        random_x = (
            th.rand(num_reset, device=self.device) * self.spawnbox_xlength
            + self.goal_offset_xy[0]
            + self.spawnbox_xoffset
        )
        random_y = (
            th.rand(num_reset, device=self.device) * self.spawnbox_ylength
            + self.goal_offset_xy[1]
            + self.spawnbox_yoffset
        )
        random_z = th.ones(num_reset, device=self.device) * (
            self.table_top_z + self._SPAWN_CLEARANCE
        )
        random_pos = th.stack([random_x, random_y, random_z], dim=-1)

        random_yaw = th.rand(num_reset, device=self.device) * 2 * math.pi
        random_quat = th.stack(
            [
                th.cos(random_yaw / 2),
                th.zeros(num_reset, device=self.device),
                th.zeros(num_reset, device=self.device),
                th.sin(random_yaw / 2),
            ],
            dim=-1,
        )

        return th.cat([random_pos, random_quat], dim=-1)

    def _step(self, actions: th.Tensor) -> StepReturn:
        self.episode_length_buf += 1

        actions = th.clamp(actions, min=-1.0, max=1.0)
        tcp_twist = self._planar_action_to_tcp_twist(actions)
        self.last_tee_xy_yaw = self._get_tee_xy_yaw()

        self._scene.pre_step()
        self.manipulator.ctrl_apply_vel_action(tcp_twist, open_gripper=False)
        self._scene.step()

        self.intersection_ratio = self._tee_projection_to_goal_intersection()
        self.success_buf = self.intersection_ratio >= self.success_thresh
        self.extras["success"] = self.success_buf
        self.pose_dist = self._tee_to_goal_pose_distance()
        self.tcp_dist = self._tcp_to_tee_footprint_distance()

        # compute reward based on task, before the reset replaces the terminal state
        reward = th.zeros(self.num_envs, device=self.device, dtype=th.float32)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew
        self.last_pose_dist = self.pose_dist.clone()
        self.last_tcp_dist = self.tcp_dist.clone()

        # check env termination (Sets the self.reset_buf)
        env_reset_idx = self._is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)
            self._obs_cache_clear()

        obs = self.get_observations()
        dones = self.reset_buf
        return StepReturn(obs, reward, dones, self.extras)

    def _planar_action_to_tcp_twist(self, actions: th.Tensor) -> th.Tensor:
        """
        Expands the planar [N, 2] action into the normalized [N, 6] TCP twist.
        The Z velocity and the angular velocity come from P controllers holding
        the TCP at `tcp_height` and `tcp_quat`.
        """
        tcp_pose = self.manipulator.get_tcp_pose()
        actions = self._limit_action_to_workspace(actions, tcp_xy=tcp_pose[:, :2])
        z_err = self.tcp_height - tcp_pose[:, 2]
        rot_err = quat_to_rotvec_error(self.tcp_start_pose[:, 3:], tcp_pose[:, 3:])

        tcp_twist = th.zeros(self.num_envs, 6, device=self.device, dtype=th.float32)
        tcp_twist[:, :2] = actions
        tcp_twist[:, 2] = self.tcp_hold_z_gain * z_err / self.max_linear_speed
        tcp_twist[:, 3:] = self.tcp_hold_rot_gain * rot_err / self.max_angular_speed
        return th.clamp(tcp_twist, min=-1.0, max=1.0)

    def _limit_action_to_workspace(
        self, actions: th.Tensor, tcp_xy: th.Tensor
    ) -> th.Tensor:
        """Scales down the planar action, so the TCP won't leave the workspace within the step."""
        max_step = self.max_linear_speed * self.policy_dt
        low = th.tensor(
            [self.tcp_workspace_x[0], self.tcp_workspace_y[0]], device=self.device
        )
        high = th.tensor(
            [self.tcp_workspace_x[1], self.tcp_workspace_y[1]], device=self.device
        )
        # outside of the workspace only the moves back inside are allowed
        a_min = th.clamp((low - tcp_xy) / max_step, max=0.0)
        a_max = th.clamp((high - tcp_xy) / max_step, min=0.0)
        return th.clamp(actions, min=a_min, max=a_max)

    def _tee_projection_to_goal_intersection(self) -> th.Tensor:
        """
        Fraction of the goal's footprint covered by the tee: warps the tee's mask
        pixels through its current pose into the goal's frame, rasterizes them,
        and intersects with the goal's mask.
        """
        obj_pose = self.object.get_pose()
        # tee's local frame -> world frame (current pose)
        tee_to_world = quat_to_zrot(obj_pose[:, 3:], device=self.device)
        tee_to_world[:, 0:2, 2] = obj_pose[:, 0:2]

        # tee-local -> world -> goal-local, in one transform
        tee_to_goal = th.matmul(self._world_to_goal_trans, tee_to_world)
        # warp every canonical mask grid point through it
        tees_in_goal = th.matmul(tee_to_goal, self._homo_uv.unsqueeze(0))
        tees_in_goal_xy = tees_in_goal[:, :2, :] / tees_in_goal[:, 2:3, :]

        tee_xy = tees_in_goal_xy[:, :, self._tee_mask_flat]  # [N, 2, K]

        # warped XY (goal frame) -> goal-mask pixel indices
        res = self.mask_resolution
        idx = th.floor((tee_xy + self.mask_half_width) * self._px_per_meter).long()
        valid = (
            (idx[:, 0, :] >= 0)
            & (idx[:, 0, :] < res)
            & (idx[:, 1, :] >= 0)
            & (idx[:, 1, :] < res)
        )

        n, _, k = idx.shape
        batch_idx = th.arange(n, device=self.device).view(-1, 1).expand(-1, k)

        # rasterize: mark which goal-grid pixels the warped tee footprint lands on
        final_render = th.zeros(n, res, res, dtype=th.bool, device=self.device)
        valid_flat = valid.reshape(-1)
        final_render[
            batch_idx.reshape(-1)[valid_flat],
            idx[:, 0, :].reshape(-1)[valid_flat],
            idx[:, 1, :].reshape(-1)[valid_flat],
        ] = True

        # overlap between warped tee footprint and the fixed goal footprint
        intersection = (final_render & self._tee_mask.unsqueeze(0)).sum(dim=(-1, -2))
        return intersection.float() / self._goal_area

    def _is_episode_complete(self) -> th.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length
        self.tee_off_table_buf = self._is_tee_off_table()
        self.extras["tee_off_table"] = self.tee_off_table_buf

        self.reset_buf = time_out_buf | self.tee_off_table_buf
        if self.terminate_on_success:
            self.reset_buf |= self.success_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = th.zeros_like(
            self.reset_buf, device=self.device, dtype=th.float32
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def _is_tee_off_table(self) -> th.Tensor:
        """The tee origin left the table top or the tee fell below it."""
        tee_pos = self.object.get_pose()[:, :3]
        x, y, z = tee_pos.unbind(dim=-1)
        return (
            (x < self.table_x[0])
            | (x > self.table_x[1])
            | (y < self.table_y[0])
            | (y > self.table_y[1])
            | (z < self.table_top_z - self._TEE_FALL_Z)
        )

    def _build_agent_observations(self, obs: TensorDict) -> th.Tensor:
        """
        Planar observations; the TCP height and orientation are fixed, so they are omitted.
        The tee is described by its footprint centroid, the point the pose distance relies on.
        """
        tcp_xy = obs[Modality.TCP_POSE][:, :2]
        obj_pose = obs[Modality.OBJECT_POSE]
        tee_yaw = quat_to_z_euler(obj_pose[:, 3:])
        tee_centroid = obj_pose[:, :2] + self._rotate_xy(self._tee_centroid, tee_yaw)
        goal_yaw = self.goal_yaw
        goal_centroid = self.goal_pose[:, :2] + self._rotate_xy(
            self._tee_centroid, goal_yaw
        )
        yaw_err = tee_yaw - goal_yaw

        # finite differences, available also on the real robot
        tee_xy_yaw = th.stack([obj_pose[:, 0], obj_pose[:, 1], tee_yaw], dim=-1)
        tee_delta = tee_xy_yaw - self.last_tee_xy_yaw
        # the yaw wraps around in [0, 2pi)
        tee_delta[:, 2] = th.remainder(tee_delta[:, 2] + math.pi, 2 * math.pi) - math.pi

        obs_components = [
            tee_centroid - tcp_xy,  # TCP-to-tee position difference
            goal_centroid - tee_centroid,  # tee-to-goal position difference
            th.stack([th.sin(yaw_err), th.cos(yaw_err)], dim=-1),  # tee-to-goal yaw
        ]
        obs_tensor = th.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor

    def _tee_to_goal_pose_distance(self) -> th.Tensor:
        """
        Root mean square distance between the corresponding points of the tee footprint at its
        current pose and at the goal pose. Combines the position and rotation errors into meters:
            d^2 = |dc|^2 + 2 (1 - cos(dyaw)) r_g^2
        where dc is the displacement of the footprint centroids and r_g is the radius of gyration.
        """
        tee_pose = self.object.get_pose()
        yaw = quat_to_z_euler(tee_pose[:, 3:])
        tee_centroid = tee_pose[:, :2] + self._rotate_xy(self._tee_centroid, yaw)
        goal_yaw = self.goal_yaw
        goal_centroid = self.goal_pose[:, :2] + self._rotate_xy(
            self._tee_centroid, goal_yaw
        )

        centroid_err_sq = (tee_centroid - goal_centroid).square().sum(dim=-1)
        rot_err_sq = 2 * (1 - th.cos(yaw - goal_yaw)) * self._tee_gyration_sq
        return th.sqrt(centroid_err_sq + rot_err_sq)

    def _get_tee_xy_yaw(self) -> th.Tensor:
        tee_pose = self.object.get_pose()
        yaw = quat_to_z_euler(tee_pose[:, 3:])
        return th.stack([tee_pose[:, 0], tee_pose[:, 1], yaw], dim=-1)

    @staticmethod
    def _rotate_xy(xy: th.Tensor, yaw: th.Tensor) -> th.Tensor:
        """Rotates the [2] or [N, 2] `xy` vectors by the [N] `yaw` angles."""
        cos, sin = th.cos(yaw), th.sin(yaw)
        x, y = xy[..., 0], xy[..., 1]
        return th.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)

    def _reward_pose_progress(self) -> th.Tensor:
        # Undiscounted potential difference (phi = -d / scale): sums up to the episode's total progress.
        # The discounted form would pay (1 - gamma) * d / scale per step, i.e. more when further away.
        # Divided by dt, as the reward scales are multiplied by it.
        progress = (self.last_pose_dist - self.pose_dist) / self.pose_dist_scale
        return progress / self.policy_dt

    def _reward_pose_alignment(self) -> th.Tensor:
        return 1 - th.tanh(self.pose_dist / self.pose_dist_scale)

    def _reward_tcp_approach(self) -> th.Tensor:
        # progress-style like the pose progress, so staying next to the tee isn't rewarded
        progress = (self.last_tcp_dist - self.tcp_dist) / self.pose_dist_scale
        return progress / self.policy_dt

    def _tcp_to_tee_footprint_distance(self) -> th.Tensor:
        """
        Planar distance from the TCP to the closest point of the tee footprint, i.e. the
        spot the TCP can actually reach, instead of the tee origin lying inside the crossbar.
        """
        tcp_xy = self.manipulator.get_tcp_pose()[:, :2]
        tee_pose = self.object.get_pose()
        yaw = quat_to_z_euler(tee_pose[:, 3:])
        # TCP expressed in the tee frame
        local = self._rotate_xy(tcp_xy - tee_pose[:, :2], -yaw)
        # distance to the union of boxes (zero inside)
        dists = []
        for center, half_size in self._TEE_FOOTPRINT_BOXES:
            c = th.tensor(center, device=self.device)
            h = th.tensor(half_size, device=self.device)
            q = (local - c).abs() - h
            dists.append(th.linalg.vector_norm(th.clamp(q, min=0.0), dim=-1))
        return th.stack(dists, dim=-1).min(dim=-1).values

    def _reward_success_bonus(self) -> th.Tensor:
        # divided by dt, so the scale is the bonus value of a single successful step
        return self.success_buf.float() / self.policy_dt
