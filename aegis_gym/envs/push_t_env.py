import math
from pathlib import Path

import numpy as np
import torch as th
import trimesh
from tensordict import TensorDict

from aegis_gym.config.types import CameraName, ExpConfig
from aegis_gym.envs.base_env import BaseEnv, Modality, ResetReturn, StepReturn
from aegis_gym.envs.manipulator import BaseManipulator
from aegis_gym.envs.objects import BaseURDF, ObjectProperties, ObjectType

from .scene import BaseScene

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "push_t"
_TEE_STL_PATH = _ASSETS_DIR / "T_shape.stl"
_TEE_URDF_PATH = _ASSETS_DIR / "T_shape.urdf"
_SPAWN_CLEARANCE = 0.02
_GOAL_MARKER_Z_SCALE = 0.0025
_GOAL_MARKER_LIFT = 1e-3


class PushTEnv(BaseEnv):
    DEFAULT_MODALITIES = frozenset({Modality.TCP_POSE, Modality.OBJECT_POSE})

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

        self._build_tee_canonical_mask(mesh_path=str(_TEE_STL_PATH))
        self._build_goal_transform()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def get_cfg_as_dict(self) -> dict:
        return self._cfg_env.as_dict()

    def get_num_envs(self) -> int:
        return self.num_envs

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

    def _extract_config(self) -> None:
        # TODO(issue##117) redesign the whole camera preview system
        self.show_cameras_gui = self._cfg_env.visualize_camera
        self.show_cell = self._cfg_env.visualize_cell

        self.num_obs = self._cfg_env.num_obs
        self.num_privileged_obs = None
        self.num_actions = self._cfg_env.num_actions
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

        self.reward_scales = self._cfg_env.push_t_reward_scales

        self.success_thresh = self._cfg_env.tee_success_intersection_thresh
        self.tee_friction = self._cfg_env.tee_friction
        self.goal_offset_xy = self._cfg_env.tee_goal_offset
        self.goal_z_rot = math.radians(self._cfg_env.tee_goal_z_rot_deg)
        self.spawnbox_xlength = self._cfg_env.tee_spawnbox_xlength
        self.spawnbox_ylength = self._cfg_env.tee_spawnbox_ylength
        self.spawnbox_xoffset = self._cfg_env.tee_spawnbox_xoffset
        self.spawnbox_yoffset = self._cfg_env.tee_spawnbox_yoffset

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
            dims=(0.2, 0.2, 0.04),
            pose=self.goal_pose_tuple,
            collision=True,
            fixed=False,
            color=(0.02, 0.02, 0.02),
            urdf_path=str(_TEE_URDF_PATH),
            friction=self.tee_friction,
        )
        self.object: BaseURDF = self._scene.add_entity(
            entity=ObjectType.URDF, properties=p_tee
        )

        goal_marker_pose = (
            self.goal_pose_tuple[0],
            self.goal_pose_tuple[1],
            self.goal_pose_tuple[2] + _GOAL_MARKER_LIFT,
            *self.goal_pose_tuple[3:],
        )
        p_goal_marker = ObjectProperties(
            dims=(0.2, 0.2, 0.04),
            pose=goal_marker_pose,
            collision=False,
            fixed=True,
            color=(0.8, 0.0, 0.0),
            mesh_path=str(_TEE_STL_PATH),
            scale=(1.0, 1.0, _GOAL_MARKER_Z_SCALE),
        )
        self._scene.add_entity(entity=ObjectType.MESH, properties=p_goal_marker)

    def _build_tee_canonical_mask(self, mesh_path: str) -> None:
        mesh = trimesh.load(mesh_path)
        z_mid = float(mesh.bounds[:, 2].mean())
        section = mesh.section(plane_origin=[0, 0, z_mid], plane_normal=[0, 0, 1])
        polygon = section.discrete[0][:, :2]

        radius = float(np.linalg.norm(polygon, axis=1).max())
        res = self._cfg_env.tee_mask_resolution
        half_width = max(self._cfg_env.tee_mask_half_width, radius * 1.15)

        lin = (np.arange(res, dtype=np.float64) + 0.5) / res * (
            2 * half_width
        ) - half_width
        xx, yy = np.meshgrid(lin, lin, indexing="ij")
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        mask_np = self._points_in_polygon(grid_pts, polygon).reshape(res, res)

        self._mask_res = res
        self._mask_half_width = half_width
        self._px_per_meter = res / (2 * half_width)
        self._tee_mask = th.from_numpy(mask_np).to(device=self.device)
        self._tee_mask_flat = self._tee_mask.reshape(-1)
        self._goal_area = float(mask_np.sum())

        homo = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())], axis=0)
        self._homo_uv = th.tensor(homo, dtype=th.float32, device=self.device)

    @staticmethod
    def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        px, py = polygon[:, 0], polygon[:, 1]
        n = len(polygon)
        inside = np.zeros(len(points), dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi, xj, yj = px[i], py[i], px[j], py[j]
            cond = ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
            )
            inside ^= cond
            j = i
        return inside

    def _build_goal_transform(self) -> None:
        goal_quat = th.tensor(
            [
                [
                    math.cos(self.goal_z_rot / 2),
                    0.0,
                    0.0,
                    math.sin(self.goal_z_rot / 2),
                ]
            ],
            device=self.device,
        )
        goal_zrot = self._quat_to_zrot(goal_quat)[0]
        goal_trans = th.eye(3, device=self.device)
        goal_trans[:2, :2] = goal_zrot[:2, :2]
        goal_trans[0:2, 2] = th.tensor(
            self.goal_offset_xy, device=self.device, dtype=th.float32
        )
        self._world_to_goal_trans = th.linalg.inv(goal_trans)

    def _quat_to_z_euler(self, quats: th.Tensor) -> th.Tensor:
        signs = th.ones_like(quats[:, -1])
        signs[quats[:, -1] < 0] = -1.0
        qw = th.clamp(quats[:, 0] * signs, min=-1.0, max=1.0)
        return 2 * th.acos(qw)

    def _quat_to_zrot(self, quats: th.Tensor) -> th.Tensor:
        alphas = self._quat_to_z_euler(quats)
        n = quats.shape[0]
        rot = th.zeros(n, 3, 3, device=self.device, dtype=th.float32)
        rot[:, 2, 2] = 1.0
        cos_a, sin_a = th.cos(alphas), th.sin(alphas)
        rot[:, 0, 0] = cos_a
        rot[:, 1, 1] = cos_a
        rot[:, 0, 1] = -sin_a
        rot[:, 1, 0] = sin_a
        return rot

    def _pseudo_render_intersection(self) -> th.Tensor:
        obj_pose = self.object.get_pose()
        tee_to_world = self._quat_to_zrot(obj_pose[:, 3:])
        tee_to_world[:, 0:2, 2] = obj_pose[:, 0:2]

        tee_to_goal = th.matmul(self._world_to_goal_trans.unsqueeze(0), tee_to_world)
        tees_in_goal = th.matmul(tee_to_goal, self._homo_uv.unsqueeze(0))
        tees_in_goal_xy = tees_in_goal[:, :2, :] / tees_in_goal[:, 2:3, :]

        tee_xy = tees_in_goal_xy[:, :, self._tee_mask_flat]  # [N, 2, K]

        res = self._mask_res
        idx = th.floor((tee_xy + self._mask_half_width) * self._px_per_meter).long()
        valid = (
            (idx[:, 0, :] >= 0)
            & (idx[:, 0, :] < res)
            & (idx[:, 1, :] >= 0)
            & (idx[:, 1, :] < res)
        )

        n, _, k = idx.shape
        batch_idx = th.arange(n, device=self.device).view(-1, 1).expand(-1, k)

        final_render = th.zeros(n, res, res, dtype=th.bool, device=self.device)
        valid_flat = valid.reshape(-1)
        final_render[
            batch_idx.reshape(-1)[valid_flat],
            idx[:, 0, :].reshape(-1)[valid_flat],
            idx[:, 1, :].reshape(-1)[valid_flat],
        ] = True

        intersection = (final_render & self._tee_mask.unsqueeze(0)).sum(dim=(-1, -2))
        return intersection.float() / self._goal_area

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
        self.intersection_ratio = th.zeros(
            self.num_envs, device=self.device, dtype=th.float32
        )
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

        self.manipulator.ctrl_gripper_open(envs_idx)
        self.manipulator.ctrl_go_to_home(envs_idx)

        tee_pose = self._get_random_tee_pose(envs_idx=envs_idx)
        self.object.set_pose(pose=tee_pose, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
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
            self.table_top_z + _SPAWN_CLEARANCE
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

        self._scene.pre_step()
        self.manipulator.ctrl_apply_vel_action(actions, open_gripper=True)
        self._scene.step()

        # check env termination (Sets the self.reset_buf)
        env_reset_idx = self._is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        self.intersection_ratio = self._pseudo_render_intersection()
        self.extras["success"] = self.intersection_ratio >= self.success_thresh

        # compute reward based on task
        reward = th.zeros_like(self.reset_buf, device=self.device, dtype=th.float32)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        obs = self.get_observations()
        dones = self.reset_buf
        return StepReturn(obs, reward, dones, self.extras)

    def _is_episode_complete(self) -> th.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = th.zeros_like(
            self.reset_buf, device=self.device, dtype=th.float32
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def _build_agent_observations(self, obs: TensorDict) -> th.Tensor:
        tcp_pose = obs[Modality.TCP_POSE]
        tcp_pos, tcp_quat = tcp_pose[:, :3], tcp_pose[:, 3:]
        obj_pose = obs[Modality.OBJECT_POSE]
        obj_pos, obj_quat = obj_pose[:, :3], obj_pose[:, 3:]
        goal_pos, goal_quat = self.goal_pose[:, :3], self.goal_pose[:, 3:]

        obs_components = [
            tcp_pos - obj_pos,  # tcp-to-object position difference
            tcp_quat,  # current tcp orientation (w, x, y, z)
            obj_pos - goal_pos,  # object-to-goal position difference
            obj_quat,  # current object orientation (w, x, y, z)
            goal_quat,  # target object orientation (w, x, y, z)
        ]
        obs_tensor = th.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor

    def _reward_rotation_alignment(self) -> th.Tensor:
        obj_quat = self.object.get_pose()[:, 3:]
        z_euler = self._quat_to_z_euler(obj_quat)
        rot_err_cos = th.cos(z_euler - self.goal_z_rot)
        return (((rot_err_cos + 1) / 2) ** 2) / 2

    def _reward_position_alignment(self) -> th.Tensor:
        obj_pos_xy = self.object.get_pose()[:, :2]
        goal_pos_xy = self.goal_pose[:, :2]
        dist = th.norm(obj_pos_xy - goal_pos_xy, p=2, dim=-1)
        return ((1 - th.tanh(5 * dist)) ** 2) / 2

    def _reward_tcp_proximity(self) -> th.Tensor:
        tcp_pos = self.manipulator.get_tcp_pose()[:, :3]
        obj_pos = self.object.get_pose()[:, :3]
        dist = th.norm(obj_pos - tcp_pos, p=2, dim=-1)
        return th.sqrt(th.clamp(1 - th.tanh(5 * dist), min=0.0)) / 20

    def _reward_success_bonus(self) -> th.Tensor:
        return (self.intersection_ratio >= self.success_thresh).float()
