import tempfile
from typing import Optional
from pathlib import Path
import math
import torch as th
from tensordict import TensorDict

import cv2
from PIL import Image

from config_types.domain_randomization import CameraPoseCfg
from ..config_types import EnvCfg, DomainRandomizationCfg, EntityCfg, Shape3D
from base_env import BaseEnv, ResetObservation, Observation
from scene import BaseScene

import numpy as np
import genesis as gs
from genesis.utils.geom import (
    transform_by_quat,
    transform_quat_by_quat,
)

from plotjuggler_udp import PlotJugglerUDP


class ReacherEnv(BaseEnv):

    def __init__(self, scene: BaseScene, cfg: EnvCfg, dr_cfg: DomainRandomizationCfg, enable_plot_juggler = False):
        super().__init__(scene)
        self.cfg: EnvCfg = cfg
        obj_cfg = EntityCfg(type="box", size=self.cfg.box_size, fixed=False, collision=True, color=Shape3D(0.8, 0.0, 0.0))
        self.object = self._scene.add_entity(obj_cfg)
        self.n_envs = self.cfg.num_envs
        self._scene.build()
        self.robot = self._scene.get_manipulator()

        if enable_plot_juggler:
            ip = "127.0.0.1"
            port = 9870
            self._joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            self._pj = PlotJugglerUDP(host=ip, port=port)
            print(f"[GraspEnv] Enabled UDP server for PlotJuggler at {ip}:{port}")
        self.enable_plot_juggler = enable_plot_juggler

        self.device = gs.device

        print(
            f"[GraspEnv] f_c: {1 / self.cfg.ctrl_dt} Hz | f_pi: {1 / self.cfg.policy_dt} Hz | Action: {self.cfg.sim_substeps} steps | Max speed: {self.cfg.max_linear_speed} m/s ; {self.cfg.max_angular_speed} rad/s"
        )

        self._dr_cfg = dr_cfg
        self._dr_cam_base_offsets: dict[str, np.ndarray] = {}
        self._dr_cam_extrinsics_active: bool = self._dr_cfg.cameras_extrinsics.enabled
        self._aug_profile: dict[str, th.Tensor] = self._init_aug_profile()

        if self._dr_cfg.enabled:
            self._setup_dr_pd_gains()
            self._cache_camera_base_offsets()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()


    def _init_aug_profile(self) -> dict[str, th.Tensor]:
        aug = self._dr_cfg.image_aug
        if not self._dr_cfg.enabled or not aug.per_episode_aug:
            return {}
        N = self.get_num_envs()
        return {
            "brightness_jitter": th.zeros(N, device=self.device),
            "contrast_jitter": th.zeros(N, device=self.device),
            "gaussian_noise_std": th.zeros(N, device=self.device),
            "gamma_range": th.zeros(N, device=self.device),
            "blur_active": th.zeros(N, dtype=th.bool, device=self.device),
            "channel_jitter": th.zeros(N, 3, device=self.device),
            "cutout_active": th.zeros(N, dtype=th.bool, device=self.device),
            "cutout_y": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_x": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_h": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_w": th.zeros(N, dtype=th.long, device=self.device),
        }

    def _setup_dr_pd_gains(self) -> None:
        cfg = self._dr_cfg.pd_gains
        if not cfg.enabled:
            return

        n_dofs = self.robot.get_n_dofs()
        kp_scale = (
            1.0
            + (th.rand(self.n_envs, n_dofs, device=self.device) * 2.0 - 1.0)
            * cfg.kp_noise
        )
        kv_scale = (
            1.0
            + (th.rand(self.n_envs, n_dofs, device=self.device) * 2.0 - 1.0)
            * cfg.kv_noise
        )

        self.robot.set_joints_pd_gains(kp_gain=kp_scale, kv_gain=kv_scale)


    def _cache_camera_base_offsets(self) -> None:
        self._dr_cam_base_offsets = {}
        if self.cfg.camera_setup != "default":
            return

        self._dr_cam_base_offsets = {
            "scene_cam": np.array(
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "tool_left_cam": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "tool_right_cam": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        }

    def _init_reward_functions(self) -> None:
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.cfg.reward_scales.keys():
            self.cfg.reward_scales[name] *= self.cfg.ctrl_dt * self.cfg.sim_substeps
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.n_envs,), device=gs.device, dtype=gs.tc_float
            )

        self.keypoints_offset = self.get_keypoint_offsets(unit_length=0.5)

    def _reward_keypoints(self) -> th.Tensor:
        tcp_pose = self.robot.get_tcp_position()
        tcp_pos, tcp_quat = tcp_pose[:, :3], tcp_pose[:, 3:]
        keypoints_offset = self.keypoints_offset
        object_offset = th.tensor(
            [0.0, 0.0, -0.08],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.n_envs, 1)

        finger_pos_keypoints = self._to_world_frame(
            tcp_pos + object_offset,
            tcp_quat,
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(
            self.object.get_pos(), self.object.get_quat(), keypoints_offset
        )
        dist = th.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return th.exp(-dist)

    def _to_world_frame(
        self,
        position: th.Tensor,  # [N, 3]
        quaternion: th.Tensor,  # [N, 4]
        keypoints_offset: th.Tensor,  # [N, 7, 3]
    ) -> th.Tensor:
        N, K, _ = keypoints_offset.shape

        v_flat = keypoints_offset.reshape(N * K, 3)
        quat_flat = quaternion[:, None].expand(N, K, 4).reshape(N * K, 4)
        rotated = transform_by_quat(v_flat, quat_flat)
        rotated = rotated.reshape(N, K, 3)
        world = position[:, None, :] + rotated
        return world

    def get_keypoint_offsets(self, unit_length: float) -> th.Tensor:
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at body center.
        """
        keypoint_offsets = (
            th.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=self.device,
                dtype=th.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(self.n_envs, 1, 1)


    def _init_buffers(self) -> None:
        self.episode_length_buf = th.zeros(
            (self.n_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.reset_buf = th.zeros(self.n_envs, dtype=th.bool, device=gs.device)
        self.goal_pose = th.zeros(self.n_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    
    def get_policy_dt(self) -> float:
        return self.cfg.policy_dt
    
    def get_cfg_as_dict(self) -> dict:
        return self.cfg.as_dict()
    
    def get_num_envs(self) -> int:
        return self.n_envs

    
    def reset(self) -> ResetObservation:
        self.reset_buf[:] = True
        self._reset_idx(th.arange(self.n_envs, device=self.device))
        self._log_state_to_plot_juggler()
        return ResetObservation(observations=self.get_observations(), extras=self.extras)

    def _reset_idx(self, envs_idx: th.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # Reset the robot
        self.robot.ctrl_gripper_open(envs_idx)
        self.robot.ctrl_go_to_home(envs_idx)

        # reset object
        num_reset = len(envs_idx)
        random_x = th.rand(num_reset, device=self.device) * 0.22 + 0.36  # 0.36 – 0.58
        random_y = (th.rand(num_reset, device=self.device) - 0.5) * 0.4  # -0.2 – 0.2
        random_z = th.ones(num_reset, device=self.device) * (
            self.cfg.table_size.z - self.cfg.workbench_size.z + self.cfg.box_size.z / 2
        )
        random_pos = th.stack([random_x, random_y, random_z], dim=-1)

        # downward facing quaternion to align with the hand
        q_downward = th.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(
            num_reset, 1
        )
        # randomly yaw the object
        random_yaw = (
            th.rand(num_reset, device=self.device) * 2 * math.pi - math.pi
        ) * 0.25
        q_yaw = th.stack(
            [
                th.cos(random_yaw / 2),
                th.zeros(num_reset, device=self.device),
                th.zeros(num_reset, device=self.device),
                th.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)

        self.goal_pose[envs_idx] = th.cat([random_pos, goal_yaw], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_yaw, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self.cfg.episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0

        if self._dr_cfg.enabled:
            self._randomize_camera_extrinsics(envs_idx)
            self._resample_aug_profile(envs_idx)

    def _randomize_camera_extrinsics(self, envs_idx: th.Tensor) -> None:
        cam_cfg = self._dr_cfg.cameras_extrinsics
        if not (
            cam_cfg.enabled
            and self._dr_cam_extrinsics_active
            and self._dr_cam_base_offsets
        ):
            return

        for cam_name, base_offset in self._dr_cam_base_offsets.items():
            cfg_key = "scene_cam" if cam_name == "scene_cam" else "tool_cams"
            per_cam: CameraPoseCfg = getattr(cam_cfg, cfg_key)

            perturb = self._make_random_se3_perturbation(
                per_cam.translation_std, per_cam.rotation_std_deg
            )
            perturbed_offset = (base_offset @ perturb).astype(np.float32)

            # TODO provide a meanigful API for accessing the cameras
            try:
                link = self.robot._robot_entity.get_link(
                    self._scene._cameras_link_names[cam_name]
                )
                for cam_dict in (self._scene._cameras, self._scene._debug_cameras):
                    if cam_name in cam_dict:
                        cam_dict[cam_name].attach(link, perturbed_offset)
                        cam_dict[cam_name].move_to_attach()
            except Exception:
                self._dr_cam_extrinsics_active = False
                break

    @staticmethod
    def _make_random_se3_perturbation(
        translation_std: float,
        rotation_std_deg: float,
    ) -> np.ndarray:
        t = np.random.randn(3) * translation_std
        angles = np.random.randn(3) * math.radians(rotation_std_deg)
        rx, ry, rz = angles
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(rx), -math.sin(rx)],
                [0, math.sin(rx), math.cos(rx)],
            ]
        )
        Ry = np.array(
            [
                [math.cos(ry), 0, math.sin(ry)],
                [0, 1, 0],
                [-math.sin(ry), 0, math.cos(ry)],
            ]
        )
        Rz = np.array(
            [
                [math.cos(rz), -math.sin(rz), 0],
                [math.sin(rz), math.cos(rz), 0],
                [0, 0, 1],
            ]
        )
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = t
        return T

    def _resample_aug_profile(self, envs_idx: th.Tensor) -> None:
        if not self._aug_profile:
            return
        aug = self._dr_cfg.image_aug
        n = len(envs_idx)
        dev = self.device

        def sample(max_val: float) -> th.Tensor:
            active = th.rand(n, device=dev) < 0.5
            return active.float() * (th.rand(n, device=dev) * max_val)

        self._aug_profile["brightness_jitter"][envs_idx] = sample(aug.brightness_jitter)
        self._aug_profile["contrast_jitter"][envs_idx] = sample(aug.contrast_jitter)
        self._aug_profile["gaussian_noise_std"][envs_idx] = sample(
            aug.gaussian_noise_std
        )
        self._aug_profile["gamma_range"][envs_idx] = sample(aug.gamma_range)
        self._aug_profile["blur_active"][envs_idx] = (
            th.rand(n, device=dev) < aug.blur_prob
        )

        ch = aug.channel_jitter
        active = (th.rand(n, device=dev) < 0.5).float().unsqueeze(1)
        self._aug_profile["channel_jitter"][envs_idx] = (
            active * (th.rand(n, 3, device=dev) * 2.0 - 1.0) * ch
        )

        cutout_cfg = aug.cutout
        prob = cutout_cfg.prob
        min_sz = cutout_cfg.min_size
        max_sz = cutout_cfg.max_size
        _, H, W = self.cfg.rgb_image_shape.as_tuple()

        active = th.rand(n, device=dev) < prob
        self._aug_profile["cutout_active"][envs_idx] = active

        hs = th.randint(min_sz, max_sz + 1, (n,), device=dev)
        ws = th.randint(min_sz, max_sz + 1, (n,), device=dev)
        ys = (th.rand(n, device=dev) * (H - hs).clamp(min=1)).long()
        xs = (th.rand(n, device=dev) * (W - ws).clamp(min=1)).long()

        self._aug_profile["cutout_h"][envs_idx] = hs
        self._aug_profile["cutout_w"][envs_idx] = ws
        self._aug_profile["cutout_y"][envs_idx] = ys
        self._aug_profile["cutout_x"][envs_idx] = xs


    
    def step(self, actions: th.Tensor) -> Observation:
        # Update time
        self.episode_length_buf += 1

        # Environment limitations
        actions = th.clamp(actions, min=-1.0, max=1.0)

        # Applying real-world scaling (with optional noise)
        max_lin_speed, max_ang_speed = self._get_max_speed_coeefs()
        actions[:, :3] *= max_lin_speed
        actions[:, 3:] *= max_ang_speed

        self.robot.ctrl_apply_vel_action(actions, open_gripper=True)
        self.robot.read_state()
        # TODO(issue#117) redesign the visualize-cameras feature
        if self.cfg.show_cameras_gui:
            self._get_observations_vis()
            if self._dr_cfg.debug_viewer:
                self._show_augumented_debug()
        self._log_state_to_plot_juggler()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self._reset_idx(env_reset_idx)

        # compute reward based on task
        reward = th.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.cfg.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs = self.get_observations()
        dones = self.reset_buf
        return Observation(obs, reward, dones, self.extras)

    def _get_max_speed_coeefs(self) -> tuple[float, float]:
        cfg = self._dr_cfg.max_speed
        if not cfg.enabled:
            return self.cfg.max_linear_speed, self.cfg.max_angular_speed

        lin_speed_noise = cfg.linear_speed_noise
        ang_speed_noise = cfg.angular_speed_noise

        lin_scale = (
            1.0 + (th.rand(1, device=self.device).item() * 2.0 - 1.0) * lin_speed_noise
        )
        ang_scale = (
            1.0 + (th.rand(1, device=self.device).item() * 2.0 - 1.0) * ang_speed_noise
        )

        max_lin_speed_rand = self.cfg.max_linear_speed * lin_scale
        max_ang_speed_rand = self.cfg.max_angular_speed * ang_scale

        return float(max_lin_speed_rand), float(max_ang_speed_rand)

    def _get_observations_vis(
        self,
        normalize: bool = True,
        swap_tool_cameras: bool = False,
        enable_vis_preview: bool = False,
        enable_record_obs: bool = False,
        record_dir: Optional[Path] = None,
    ) -> th.Tensor:
        rgb_list: list[th.Tensor] = [None] * len(self._scene._cameras)

        for cam_name, cam in self._scene._cameras.items():
            if swap_tool_cameras:
                # TODO(issue#121) unify cameras names
                cam_name = {
                    "tool_left_cam": "tool_right_cam",
                    "tool_right_cam": "tool_left_cam",
                }.get(cam_name, cam_name)
            cam_id = self._scene._cameras_order[cam_name]

            rgb, _, _, _ = cam.render(
                rgb=True, depth=False, segmentation=False, normal=False
            )
            rgb = rgb.permute(0, 3, 1, 2)[:, :3]  # (B, 3, H, W)
            if normalize:
                rgb = th.clamp(rgb, 0.0, 255.0).div_(255.0)

            if self._dr_cfg.enabled:
                rgb = self._apply_image_augmentation(rgb)

            rgb_list[cam_id] = rgb

        # TODO(issue#117) redesign the visualization preview
        # TODO(issue#121) unify the code between grasp_env and grasp_env_ros
        if enable_vis_preview or enable_record_obs:
            preview = self._create_vis_observation_preview(
                obs=rgb_list, normalize=normalize
            )

            if enable_vis_preview:
                cv2.imshow("Visual observation preview", preview)
                cv2.waitKey(1)

            if enable_record_obs:
                record_dir = record_dir or Path(tempfile.gettempdir()) / "aegis_vis_obs"
                record_dir.mkdir(parents=True, exist_ok=True)
                self._record_vis_observation(preview=preview, output_dir=record_dir)

        return th.cat(rgb_list, dim=1).float()

    def _record_vis_observation(self, preview: np.ndarray, output_dir: Path) -> None:
        record_step = getattr(self, "_record_step", 0)
        fname = f"frame_{record_step:08d}.png"
        Image.fromarray(preview).save(output_dir / fname)

        self._record_step = record_step + 1

    # TODO(issue#118) Extract domain randomization logic to external file
    def _apply_image_augmentation(self, rgb: th.Tensor) -> th.Tensor:
        aug = self._dr_cfg.image_aug
        if not aug.enabled:
            return rgb

        N, C, H, W = rgb.shape
        device = self.device
        prof = self._aug_profile  # {} = disabled, non-empty = per-episode replay

        def sample_magnitude(key: str, shape: tuple) -> th.Tensor:
            """Magnitude in [0, max_val]; sign is re-sampled each frame for variety."""
            mag = (
                prof[key].view(shape)
                if prof
                else th.rand(shape, device=device) * getattr(aug, key)
            )
            return mag * (th.rand(shape, device=device) * 2.0 - 1.0)

        def sample_signed(key: str, shape: tuple) -> th.Tensor:
            """Already a signed delta in the profile (channel_jitter)."""
            if prof:
                return prof[key].view(shape)
            return (th.rand(shape, device=device) * 2.0 - 1.0) * getattr(aug, key)

        # -- Brightness --
        b_delta = sample_magnitude("brightness_jitter", (N, 1, 1, 1))
        if b_delta.abs().any():
            rgb = rgb * (1.0 + b_delta)

        # -- Per-channel jitter (signed delta, no re-sampling) --
        ch_delta = sample_signed("channel_jitter", (N, C, 1, 1))
        if ch_delta.abs().any():
            rgb = rgb * (1.0 + ch_delta)

        # -- Contrast --
        c_delta = sample_magnitude("contrast_jitter", (N, 1, 1, 1))
        if c_delta.abs().any():
            mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
            rgb = (rgb - mean) * (1.0 + c_delta) + mean

        # -- Gaussian noise (magnitude only, always additive) --
        noise_std = sample_magnitude("gaussian_noise_std", (N, 1, 1, 1)).abs()
        if noise_std.any():
            rgb = rgb + th.randn_like(rgb) * noise_std

        # -- Gamma --
        g_delta = sample_magnitude("gamma_range", (N, 1, 1, 1))
        if g_delta.abs().any():
            rgb = rgb.clamp(1e-6, 1.0).pow(1.0 + g_delta)

        # -- Gaussian blur (per-env) --
        blur_active = (
            prof["blur_active"] if prof else th.rand(N, device=device) < aug.blur_prob
        )
        if blur_active.any():
            blurred = self._apply_gaussian_blur(
                rgb, kernel_size=aug.blur_kernel_size, sigma=aug.blur_sigma
            )
            rgb = th.where(blur_active.view(N, 1, 1, 1), blurred, rgb)

        # -- Cutout --
        if aug.cutout.prob > 0.0 or (prof and prof["cutout_active"].any()):
            rgb = self._apply_cutout(rgb)

        return rgb.clamp(0.0, 1.0)

    def _create_vis_observation_preview(
        self, obs: list[th.Tensor], normalize: bool
    ) -> np.ndarray:

        opencv_images: list[np.ndarray] = [None] * len(obs)
        for cam_name in self._cameras.keys():
            cam_id = self._cameras_order[cam_name]

            FIRST_IMG = 0
            img = obs[cam_id][FIRST_IMG].permute(1, 2, 0).cpu().numpy()  # NCHW -> HWC
            img = (img * 255).astype(np.uint8) if normalize else img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            height, width = img.shape[:2]
            max_side = 256
            scale = min(max_side / width, max_side / height)
            img = cv2.resize(
                img,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            cv2.putText(
                img,
                str(cam_name),
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=2,
            )
            opencv_images[cam_id] = img

        return np.hstack(opencv_images)


    def _show_augumented_debug(self) -> None:
        # TODO(issue#117) redesign the visualize-camera feature
        if not self._scene._debug_cameras:
            return

        frames = []
        for name, cam in self._scene._debug_cameras.items():
            rgb, _, _, _ = cam.render(
                rgb=True, depth=False, segmentation=False, normal=False
            )
            rgb = rgb.permute(0, 3, 1, 2)[:, :3]
            rgb = th.clamp(rgb, 0.0, 255.0) / 255.0
            if self._dr_cfg.enabled:
                rgb = self._apply_image_augmentation(rgb)
            frame = rgb[0].detach().float().cpu()
            frame_np = (frame.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(
                np.uint8
            )
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            frame_up = cv2.resize(
                frame_bgr, (256, 256), interpolation=cv2.INTER_NEAREST
            )
            cv2.putText(
                frame_up, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
            frames.append(frame_up)
        cv2.imshow("Network view", np.concatenate(frames, axis=1))
        cv2.waitKey(1)

    
    def get_observations(self) -> TensorDict:
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pos, tcp_quat = tcp_pose[:, :3], tcp_pose[:, 3:]
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()

        obs_components = [
            tcp_pos - obj_pos,  # 3D position difference
            tcp_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        obs_tensor = th.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return TensorDict({"policy": obs_tensor}, batch_size=[self.n_envs])
    
    def get_privileged_observations(self) -> TensorDict:
        raise NotImplementedError("Priviliged observations are not yet implemented")

    
    def is_episode_complete(self) -> th.Tensor:
        time_out_buf = self.episode_length_buf > self.cfg.max_episode_length

        # check if the end-effector is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = th.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def _log_state_to_plot_juggler(self) -> None:
        if not self.enable_plot_juggler:
            return

        data = {}
        # TODO change api to expose the robot entity
        robot = self._scene._robot_entity
        for name in self._joint_names:
            j = robot.get_joint(name=name)
            for idx in j.dofs_idx_local:
                # TODO(issue#119) investigate one query for obtaining all of the data
                # Query each DOF individually to get scalar values
                pos = robot.get_dofs_position([idx])
                vel = robot.get_dofs_velocity([idx])
                force = robot.get_dofs_force([idx])

                # Convert to float - handle both tensor and array shapes
                data[f"joint_states/{name}/position"] = float(pos.flatten()[0])
                data[f"joint_states/{name}/velocity"] = float(vel.flatten()[0])
                data[f"joint_states/{name}/effort"] = float(force.flatten()[0])

        all_link_positions = robot.get_links_pos()
        # all_link_quats = robot.get_links_quat()

        link_positions = all_link_positions[0]
        # link_quats = all_link_quats[0]

        ee_idx = -1  # Last link = end effector
        position = link_positions[ee_idx]

        data["ee/position/x"] = float(position[0])
        data["ee/position/y"] = float(position[1])
        data["ee/position/z"] = float(position[2])
        # TODO(issue#55) Enable orientation logging
        # data["ee/orientation/roll"] = float(roll)
        # data["ee/orientation/pitch"] = float(pitch)
        # data["ee/orientation/yaw"] = float(yaw)

        self._pj.send(data)
