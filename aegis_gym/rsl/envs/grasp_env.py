import math
from typing import Literal, Optional

import cv2
import genesis as gs
import numpy as np
import torch as th
import torch.nn.functional as F
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from genesis.utils.geom import (
    transform_by_quat,
    transform_quat_by_quat,
)

from config_types.domain_randomization import DomainRandomizationCfg, CameraPoseCfg
from .manipulator import Manipulator
from .plotjuggler_udp import PlotJugglerUDP

# Further example
# https://github.com/isaac-sim/IsaacLab/blob/857da263c08fa78664e40ab957f996b22153d181/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py


class GraspEnv(VecEnv):
    def __init__(
        self,
        env_cfg: dict,
        robot_cfg: dict,
        dr_cfg: DomainRandomizationCfg,
        show_viewer: bool = False,
        enable_plot_juggler: bool = False,
    ) -> None:
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
        self._enable_pj_logging = enable_plot_juggler

        self._cfg = env_cfg
        self.device = gs.device

        self._extract_config()
        print(
            f"[GraspEnv] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self._dr_cfg = dr_cfg
        self._dr_cam_base_offsets: dict[str, np.ndarray] = {}
        self._dr_cam_extrinsics_active: bool = self._dr_cfg.cameras_extrinsics.enabled
        self._aug_profile: dict[str, th.Tensor] = self._init_aug_profile()

        self._cameras: dict[str, gs.Camera] = {}
        # TODO(issue#117) redesign the cameras preview feature
        self._debug_cameras: dict[str, gs.Camera] = {}
        self._setup_genesis_scene(self._cfg, robot_cfg, show_viewer)
        self._cameras_link_names = {
            "scene_cam": "cam_scene_rgb_camera_frame",
            "tool_left_cam": "cam_tool_left",
            "tool_right_cam": "cam_tool_right",
        }

        self.scene.build(
            n_envs=env_cfg["num_envs"],
            # env_spacing=(1.0, 1.0),
        )

        self.robot.set_pd_gains()
        self._attach_cameras()

        if self._dr_cfg.enabled:
            self._setup_dr_pd_gains()
            self._cache_camera_base_offsets()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _extract_config(self) -> None:
        # TODO(issue##117) redesign the whole camera preview system
        self.show_cameras_gui = self._cfg["visualize_camera"]

        self.num_envs = self._cfg["num_envs"]
        self.num_obs = self._cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = self._cfg["num_actions"]
        self.image_width = self._cfg["image_resolution"][0]
        self.image_height = self._cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.show_cell = self._cfg["visualize_cell"]
        self.camera_setup: Literal["default", "scene_dual"] = self._cfg["camera_setup"]
        self.table_size = self._cfg["table_size"]
        self.workbench_size = self._cfg["workbench_size"]
        self.box_size = self._cfg["box_sizes"]["default"]

        self.ctrl_dt = self._cfg["ctrl_dt"]
        self.policy_dt = self._cfg["policy_dt"]
        self.sim_substeps = int(
            math.ceil(self._cfg["policy_dt"] / self._cfg["ctrl_dt"])
        )
        self.max_episode_length = int(
            math.ceil(self._cfg["episode_length_s"] / self.policy_dt)
        )

        self.max_linear_speed = self._cfg["action_scaling"]["max_linear_speed"]
        self.max_angular_speed = self._cfg["action_scaling"]["max_angular_speed"]

        self.reward_scales = self._cfg["reward_scales"]

    def _setup_genesis_scene(
        self, env_cfg: dict, robot_cfg: dict, show_viewer: bool
    ) -> None:
        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.policy_dt,
                substeps=self.sim_substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.policy_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_dofs_info=True,  # Enables (n_evs, n_dofs) shape
                batch_links_info=True,  # Enables (n_envs, n_links, ...) shapes
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.num_envs)),
            ),
            viewer_options=gs.options.ViewerOptions(
                # max_FPS=int(0.5 / self.ctrl_dt),
                max_FPS=int(60),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            renderer=gs.options.renderers.BatchRenderer(
                use_rasterizer=env_cfg["use_rasterizer"],
            ),
            show_viewer=show_viewer,
        )

        # == add ground ==
        plane_z = -self.workbench_size[2] if self.show_cell else 0.0
        self.scene.add_entity(
            gs.morphs.Plane(pos=(0, 0, plane_z)),
            surface=gs.surfaces.Default(color=(0.98, 0.98, 0.98)),
        )

        # == add robot ==
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            show_cell=self.show_cell,
            device=gs.device,
        )

        # == add table ==
        if self.show_cell:
            self.table = self.scene.add_entity(
                gs.morphs.Box(
                    size=self.table_size,
                    pos=(
                        self.table_size[0] / 2 + self.workbench_size[0] / 2,
                        0.0,
                        self.table_size[2] / 2 - self.workbench_size[2],
                    ),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=(1.0, 0.96, 0.92)),
                material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
            )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=self.box_size,
                fixed=env_cfg["box_fixed"],
                collision=env_cfg["box_collision"],
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(0.8, 0.0, 0.0),
                ),
            ),
        )

        # == add cameras ==
        match self.camera_setup:
            case "default":
                self._add_camera(name="scene_cam", fov=38)
                self._add_camera(name="tool_left_cam", fov=30)
                self._add_camera(name="tool_right_cam", fov=30)
            case "scene_dual":
                self._add_camera(name="scene_left_cam", pos=(1.25, 0.3, 0.3), fov=60)
                self._add_camera(name="scene_right_cam", pos=(1.25, -0.3, 0.3), fov=60)

        if self.show_cameras_gui:
            self.record_cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=self.show_cameras_gui,
                debug=True,
            )

        # == add lighting ==
        self.scene.add_light(
            pos=(0.0, 0.0, 2.46),
            dir=(1.0, 1.0, -1.0),
            color=(1.0, 1.0, 1.0),
            intensity=0.6,
            directional=False,
            castshadow=True,
            cutoff=90.0,
        )

    # TODO(issue#41): Refactor camera handling to use a unified camera registry instead of dynamic attributes
    def _add_camera(
        self,
        name: str,
        pos: tuple = (0.0, 0.0, 0.0),
        fov: float = 40,  # deg
        lookat: tuple = (0.0, 0.0, 0.0),
        res: tuple = None,
    ):
        if res is None:
            res = (self.image_width, self.image_height)
        self._cameras[name] = self.scene.add_camera(
            res=res,
            pos=pos,
            lookat=lookat,
            fov=fov,
            GUI=self.show_cameras_gui,
        )
        if self.show_cameras_gui:
            self._debug_cameras[name] = self.scene.add_camera(
                res=res,
                pos=pos,
                lookat=lookat,
                fov=fov,
                GUI=False,
            )

    def _attach_cameras(self):
        if self.camera_setup != "default":
            return

        scene_offset_T = np.array(
            [
                [0.0, 0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        tool_offset_T = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.03],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        cams_to_attach = [
            ("scene_cam", "cam_scene_rgb_camera_frame", scene_offset_T),
            ("tool_left_cam", "cam_tool_left", tool_offset_T),
            ("tool_right_cam", "cam_tool_right", tool_offset_T),
        ]

        for cam_name, link_name, offset in cams_to_attach:
            link = self.robot._robot_entity.get_link(link_name)
            for cam_dict in (self._cameras, self._debug_cameras):
                if cam_name in cam_dict:
                    cam_dict[cam_name].attach(link, offset)
                    cam_dict[cam_name].move_to_attach()

    # Required by rsl_rl
    @property
    def unwrapped(self) -> "GraspEnv":
        return self

    # Required by rsl_rl
    @property
    def step_dt(self) -> float:
        return self.policy_dt

    # Required by rsl_rl
    @property
    def cfg(self) -> dict:
        return self._cfg

    def _init_reward_functions(self) -> None:
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt * self.sim_substeps
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, device=self.device, unit_length=0.5
        )

    def _init_buffers(self) -> None:
        self.episode_length_buf = th.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.reset_buf = th.zeros(self.num_envs, dtype=th.bool, device=gs.device)
        self.goal_pose = th.zeros(self.num_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: th.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

        # reset object
        num_reset = len(envs_idx)
        random_x = th.rand(num_reset, device=self.device) * 0.22 + 0.36  # 0.36 – 0.58
        random_y = (th.rand(num_reset, device=self.device) - 0.5) * 0.4  # -0.2 – 0.2
        random_z = th.ones(num_reset, device=self.device) * (
            self.table_size[2] - self.workbench_size[2] + self.box_size[2] / 2
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
                / self._cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        if self._dr_cfg.enabled:
            self._randomize_camera_extrinsics(envs_idx)
            self._resample_aug_profile(envs_idx)

    def generate_object_poses(self, seed: int) -> th.Tensor:
        rng = th.Generator(device=self.device)
        rng.manual_seed(seed)

        random_x = (
            th.rand(self.num_envs, device=self.device, generator=rng) * 0.22 + 0.36
        )
        random_y = (
            th.rand(self.num_envs, device=self.device, generator=rng) - 0.5
        ) * 0.4
        random_z = th.ones(self.num_envs, device=self.device) * (
            self.table_size[2] - self.workbench_size[2] + self.box_size[2] / 2
        )
        random_yaw = (
            th.rand(self.num_envs, device=self.device, generator=rng) * 2 * math.pi
            - math.pi
        ) * 0.25

        q_downward = th.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        q_yaw = th.stack(
            [
                th.cos(random_yaw / 2),
                th.zeros(self.num_envs, device=self.device),
                th.zeros(self.num_envs, device=self.device),
                th.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        object_quat = transform_quat_by_quat(q_yaw, q_downward)
        object_pos = th.stack([random_x, random_y, random_z], dim=-1)

        return th.cat([object_pos, object_quat], dim=-1)

    def apply_object_poses(self, pose: th.Tensor) -> None:
        object_pos, object_quat = pose[:, :3], pose[:, 3:]
        self.object.set_pos(object_pos)
        self.object.set_quat(object_quat)
        self.goal_pose[:] = pose

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=gs.device))
        self._log_state_to_plot_juggler()
        return self.get_observations(), self.extras

    def step(self, actions: th.Tensor) -> tuple[TensorDict, th.Tensor, th.Tensor, dict]:
        # Update time
        self.episode_length_buf += 1

        # Environment limitations
        actions = th.clamp(actions, min=-1.0, max=1.0)

        # Applying real-world scaling (with optional noise)
        max_lin_speed, max_ang_speed = self._get_max_speed_coeefs()
        actions[:, :3] *= max_lin_speed
        actions[:, 3:] *= max_ang_speed

        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()
        # TODO(issue#117) redesign the visualize-cameras feature
        if self.show_cameras_gui:
            self.get_observations_vis()
            if self._dr_cfg.debug_viewer:
                self._show_augmented_debug()
        self._log_state_to_plot_juggler()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = th.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs = self.get_observations()
        dones = self.reset_buf
        return obs, reward, dones, self.extras

    def _get_max_speed_coeefs(self) -> tuple[float, float]:
        cfg = self._dr_cfg.max_speed
        if not cfg.enabled:
            return self.max_linear_speed, self.max_angular_speed

        lin_speed_noise = cfg.linear_speed_noise
        ang_speed_noise = cfg.angular_speed_noise

        lin_scale = 1.0 + (np.random.uniform(-1.0, 1.0)) * lin_speed_noise
        ang_scale = 1.0 + (np.random.uniform(-1.0, 1.0)) * ang_speed_noise

        max_lin_speed_rand = self.max_linear_speed * lin_scale
        max_ang_speed_rand = self.max_angular_speed * ang_scale

        return float(max_lin_speed_rand), float(max_ang_speed_rand)

    def calib_run(
        self,
        joints_diff: Optional[th.Tensor] = None,
        cart_diff: Optional[th.Tensor] = None,
        steps: int = 100,
    ) -> None:
        idle_steps = 300  # int(0.4 * steps)
        print(f">>> Idling for {idle_steps} steps.")
        for _ in range(idle_steps):
            self.scene.step()
            self._log_state_to_plot_juggler()

        move_steps = int(steps)
        steps_per_action = int(
            250 / 10
        )  # Control Frequency divided by Policy Frequency
        # move_per_action = 0.106 / 1000 * steps_per_action # about 2.65 mm
        move_per_action = 0.196 / 500 * steps_per_action  # about 9,8mm
        steady_error_compensation_coeff = 1.03  # 1.087 #1.044 # 1.0472

        print(f">>> Moving to relative goal for {move_steps} steps.")
        if joints_diff is not None:
            self.robot.apply_dof_rel_action(joints_diff)
        elif cart_diff is not None:
            from math import ceil

            print(f">>> Steps per action: {steps_per_action}.")
            print(f">>> Movement per action: {move_per_action} m.")
            actions_num = int(ceil(move_steps / steps_per_action))
            print(
                f">>> Assuming, that the goal will be reachable in: {actions_num} actions."
            )

            scaled_cart_diff = cart_diff / actions_num * steady_error_compensation_coeff
            print(f">>> Scaled down target: {scaled_cart_diff}")
            for action_id in range(actions_num):
                print(f">>> Applying action #{action_id + 1}")
                self.robot.apply_action(scaled_cart_diff, open_gripper=True)
                for _ in range(steps_per_action):
                    self.scene.step()
                    self._log_state_to_plot_juggler()

        print(f">>> Idling for {idle_steps} steps.")
        for _ in range(idle_steps):
            self.scene.step()
            self._log_state_to_plot_juggler()

    def get_privileged_observations(self) -> None:
        raise NotImplementedError

    def is_episode_complete(self) -> th.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the end-effector is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = th.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> TensorDict:
        ee_pos, ee_quat = (
            self.robot.ee_pose[:, :3],
            self.robot.ee_pose[:, 3:7],
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()

        obs_components = [
            ee_pos - obj_pos,  # 3D position difference
            ee_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        obs_tensor = th.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return TensorDict({"policy": obs_tensor}, batch_size=[self.num_envs])

    def get_stereo_rgb_images(self, normalize: bool = True) -> th.Tensor:
        rgb_left, _, _, _ = self.scene_left_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb_right, _, _, _ = self.scene_right_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )

        # convert to the NCHW format
        rgb_left = rgb_left.permute(0, 3, 1, 2)[:, :3]  # shape (N, 3, H, W)
        rgb_right = rgb_right.permute(0, 3, 1, 2)[:, :3]  # shape (N, 3, H, W)

        # normalize if requested
        if normalize:
            rgb_left = th.clamp(rgb_left, min=0.0, max=255.0).div_(255.0)
            rgb_right = th.clamp(rgb_right, min=0.0, max=255.0).div_(255.0)

        # concatenate left and right rgb images along channel dimension
        stereo_rgb = th.cat([rgb_left, rgb_right], dim=1)
        return stereo_rgb

    def get_observations_vis(self, normalize: bool = True) -> th.Tensor:
        cams = tuple(self._cameras.values())
        rgb_list: list[th.Tensor] = [None] * len(cams)

        for cam_id, cam in enumerate(cams):
            rgb, _, _, _ = cam.render(
                rgb=True, depth=False, segmentation=False, normal=False
            )
            rgb = rgb.permute(0, 3, 1, 2)[:, :3]  # (B, 3, H, W)
            if normalize:
                rgb = th.clamp(rgb, 0.0, 255.0) / 255.0

            if self._dr_cfg.enabled:
                rgb = self._apply_image_augmentation(rgb)

            rgb_list[cam_id] = rgb

        return th.cat(rgb_list, dim=1)

    def _reward_keypoints(self) -> th.Tensor:
        ee_pos = self.robot.ee_pose[:, :3]
        ee_quat = self.robot.ee_pose[:, 3:7]
        keypoints_offset = self.keypoints_offset
        object_offset = th.tensor(
            [0.0, 0.0, -0.08],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)

        finger_pos_keypoints = self._to_world_frame(
            ee_pos + object_offset,
            ee_quat,
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
        world = th.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(
                keypoints_offset[:, k], quaternion
            )
        return world

    @staticmethod
    def get_keypoint_offsets(
        batch_size: int, device: str, unit_length: float = 0.5
    ) -> th.Tensor:
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
                device=device,
                dtype=th.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    def grasp_and_lift_demo(self) -> float:
        total_steps = self.max_episode_length
        grab_height = 0.08
        goal_pose = self.robot.ee_pose.clone()
        goal_pose[:, 2] -= grab_height
        # lift pose (above the object)
        lift_height = 0.16
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        # final pose (above the table)
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.4
        # reset pose (home pose)
        reset_pose = th.tensor(
            [0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

        pos_threshold = 0.08
        hold_steps_required = self.max_episode_length / 10
        hold_counter = th.zeros(self.num_envs, device=self.device)

        for i in range(total_steps):
            if i < total_steps / 5:  # go down
                self.robot.go_to_goal(goal_pose, open_gripper=True)
            elif i < total_steps * 2 / 5:  # grasping
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps * 3 / 5:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 4 / 5:  # final
                self.robot.go_to_goal(final_pose, open_gripper=False)
                obj_pos = self.object.get_pos()
                target_pos = final_pose[:, :3]

                dist = th.norm(obj_pos - target_pos, dim=-1)
                in_target = dist < pos_threshold

                hold_counter[in_target] += 1
            else:  # reset
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()

        success = hold_counter >= hold_steps_required
        success_rate = success.float().mean().item()

        return success_rate

    def _setup_dr_pd_gains(self) -> None:
        cfg = self._dr_cfg.pd_gains
        if not cfg.enabled:
            return

        nominal_kp = th.tensor(
            self.robot._kp_gains, dtype=th.float32, device=self.device
        )
        nominal_kv = th.tensor(
            self.robot._kv_gains, dtype=th.float32, device=self.device
        )

        n_dofs = len(self.robot._kp_gains)
        kp_scale = (
            1.0
            + (th.rand(self.num_envs, n_dofs, device=self.device) * 2.0 - 1.0)
            * cfg.kp_noise
        )
        kv_scale = (
            1.0
            + (th.rand(self.num_envs, n_dofs, device=self.device) * 2.0 - 1.0)
            * cfg.kv_noise
        )

        kp_rand = nominal_kp.unsqueeze(0) * kp_scale
        kv_rand = nominal_kv.unsqueeze(0) * kv_scale

        self.robot.set_pd_gains(kp=kp_rand, kv=kv_rand)

    def _cache_camera_base_offsets(self) -> None:
        self._dr_cam_base_offsets = {}
        if self.camera_setup != "default":
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

    def _init_aug_profile(self) -> dict[str, th.Tensor]:
        aug = self._dr_cfg.image_aug
        if not self._dr_cfg.enabled or not aug.per_episode_aug:
            return {}
        N = self.num_envs
        return {
            "brightness_jitter": th.zeros(N, device=self.device),
            "contrast_jitter": th.zeros(N, device=self.device),
            "gaussian_noise_std": th.zeros(N, device=self.device),
            "gamma_range": th.zeros(N, device=self.device),
            "blur_active": th.zeros(N, device=self.device),
            "channel_jitter": th.zeros(N, 3, device=self.device),
            "cutout_active": th.zeros(N, dtype=th.bool, device=self.device),
            "cutout_y": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_x": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_h": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_w": th.zeros(N, dtype=th.long, device=self.device),
        }

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
        ).float()

        ch = aug.channel_jitter
        active = (th.rand(n, device=dev) < 0.5).float().unsqueeze(1)
        self._aug_profile["channel_jitter"][envs_idx] = (
            active * (th.rand(n, 3, device=dev) * 2.0 - 1.0) * ch
        )

        cutout_cfg = aug.cutout
        prob = cutout_cfg.prob
        min_sz = cutout_cfg.min_size
        max_sz = cutout_cfg.max_size
        H, W = self.image_height, self.image_width
        active = th.rand(n, device=dev) < prob
        self._aug_profile["cutout_active"][envs_idx] = active
        self._aug_profile["cutout_h"][envs_idx] = th.randint(
            min_sz, max_sz + 1, (n,), device=dev
        )
        self._aug_profile["cutout_w"][envs_idx] = th.randint(
            min_sz, max_sz + 1, (n,), device=dev
        )
        self._aug_profile["cutout_y"][envs_idx] = th.randint(
            0, max(1, H - max_sz), (n,), device=dev
        )
        self._aug_profile["cutout_x"][envs_idx] = th.randint(
            0, max(1, W - max_sz), (n,), device=dev
        )

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

            try:
                link = self.robot._robot_entity.get_link(
                    self._cameras_link_names[cam_name]
                )
                for cam_dict in (self._cameras, self._debug_cameras):
                    if cam_name in cam_dict:
                        cam_dict[cam_name].attach(link, perturbed_offset)
                        cam_dict[cam_name].move_to_attach()
            except Exception:
                self._dr_cam_extrinsics_active = False
                break

    def _apply_gaussian_blur(
        self,
        rgb: th.Tensor,
        kernel_size: int,
        sigma: float,
    ) -> th.Tensor:
        x = (
            th.arange(kernel_size, dtype=th.float32, device=self.device)
            - kernel_size // 2
        )
        k1d = th.exp(-(x**2) / (2.0 * sigma**2))
        k1d = k1d / k1d.sum()
        k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
        C = rgb.shape[1]
        kernel = k2d.unsqueeze(0).unsqueeze(0).expand(C, 1, kernel_size, kernel_size)
        pad = kernel_size // 2
        return F.conv2d(rgb, kernel, padding=pad, groups=C)

    def _apply_cutout(self, rgb: th.Tensor) -> th.Tensor:
        N, _, H, W = rgb.shape
        cutout_cfg = self._dr_cfg.image_aug.cutout
        prof = self._aug_profile

        if not prof:
            return rgb

        if "cutout_active" in prof:
            for i in range(N):
                if prof["cutout_active"][i]:
                    y = int(prof["cutout_y"][i].item())
                    x = int(prof["cutout_x"][i].item())
                    h = int(prof["cutout_h"][i].item())
                    w = int(prof["cutout_w"][i].item())
                    rgb[i, :, y : y + h, x : x + w] = 0.0
            return rgb

        prob = cutout_cfg.prob
        min_sz = cutout_cfg.min_size
        max_sz = cutout_cfg.max_size

        for i in range(N):
            if th.rand(1).item() < prob:
                sz_h = int(th.randint(min_sz, max_sz + 1, (1,)).item())
                sz_w = int(th.randint(min_sz, max_sz + 1, (1,)).item())
                y = int(th.randint(0, max(1, H - sz_h + 1), (1,)).item())
                x = int(th.randint(0, max(1, W - sz_w + 1), (1,)).item())
                rgb[i, :, y : y + sz_h, x : x + sz_w] = 0.0
        return rgb

    def _apply_image_augmentation(self, rgb: th.Tensor) -> th.Tensor:
        aug = self._dr_cfg.image_aug

        if not aug.enabled:
            return rgb

        N = rgb.shape[0]
        prof = self._aug_profile

        def mag(key: str) -> th.Tensor:
            if prof:
                return prof[key].view(N, 1, 1, 1)
            return th.full((N, 1, 1, 1), getattr(aug, key), device=self.device)

        b = mag("brightness_jitter")
        if b.any():
            factor = 1.0 + (th.rand(N, 1, 1, 1, device=self.device) * 2.0 - 1.0) * b
            rgb = rgb * factor

        ch_max = aug.channel_jitter
        if ch_max > 0.0:
            if prof is not None:
                ch_shift = prof["channel_jitter"].view(N, 3, 1, 1)
            else:
                ch_shift = (
                    th.rand(N, 3, 1, 1, device=self.device) * 2.0 - 1.0
                ) * ch_max
            rgb = rgb * (1.0 + ch_shift)

        c = mag("contrast_jitter")
        if c.any():
            mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
            factor = 1.0 + (th.rand(N, 1, 1, 1, device=self.device) * 2.0 - 1.0) * c
            rgb = (rgb - mean) * factor + mean

        sigma = mag("gaussian_noise_std")
        if sigma.any():
            rgb = rgb + th.randn_like(rgb) * sigma

        g = mag("gamma_range")
        if g.any():
            gamma = 1.0 + (th.rand(N, 1, 1, 1, device=self.device) * 2.0 - 1.0) * g
            rgb = rgb.clamp(1e-6, 1.0).pow(gamma)

        if prof is not None:
            blur_mask = prof["blur_active"].view(N, 1, 1, 1)
            if blur_mask.any():
                ks = aug.blur_kernel_size
                s = aug.blur_sigma
                blurred = self._apply_gaussian_blur(rgb, kernel_size=ks, sigma=s)
                rgb = th.where(blur_mask > 0, blurred, rgb)
        else:
            p = aug.blur_prob
            if p > 0.0 and th.rand(1).item() < p:
                ks = aug.blur_kernel_size
                s = aug.blur_sigma
                rgb = self._apply_gaussian_blur(rgb, kernel_size=ks, sigma=s)

        if aug.cutout.prob > 0.0:
            rgb = self._apply_cutout(rgb)

        return rgb.clamp(0.0, 1.0)

    def _show_augmented_debug(self) -> None:
        # TODO(issue#117) redesign the visualize-camera feature
        if not self._debug_cameras:
            return

        frames = []
        for name, cam in self._debug_cameras.items():
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

    def _log_state_to_plot_juggler(self) -> None:
        if not self._enable_pj_logging:
            return

        data = {}
        robot = self.robot._robot_entity
        for name in self._joint_names:
            j = robot.get_joint(name=name)
            for idx in j.dofs_idx_local:
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
