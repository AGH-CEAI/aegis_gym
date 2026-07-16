import math
from typing import Optional

import genesis as gs
import numpy as np
from tensordict import TensorDict
import torch as th
from genesis.vis.camera import Camera
from genesis.utils.geom import (
    transform_by_quat,
    transform_quat_by_quat,
)

from .manipulator import GenesisManipulator
from .base_env import BaseEnv, StepReturn, ResetReturn, Modality
from .objects import BaseBox, ObjectsFactory
from .plotjuggler_udp import PlotJugglerUDP

from config.types import ExpConfig, CameraPoseCfg, Control, CamerasSetup

# Further example
# https://github.com/isaac-sim/IsaacLab/blob/857da263c08fa78664e40ab957f996b22153d181/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py


class GraspEnv(BaseEnv):
    DEFAULT_MODALITIES = frozenset({Modality.TCP_POSE, Modality.OBJECT_POSE})

    def __init__(
        self,
        cfg: ExpConfig,
    ) -> None:
        super().__init__(
            scene=None, cfg=cfg.env_cfg
        )  # TODO(issue#128) introduce Scene abstraction
        self.device = cfg.get_device()

        self._observation_fns = {
            Modality.TCP_POSE: self._observe_tcp_pose,
            Modality.OBJECT_POSE: self._observe_object_pose,
            Modality.CAMERA_SCENE_RGB: self._observe_camera_scene,
            Modality.CAMERA_TOOL_LEFT_RGB: self._observe_camera_tool_left,
            Modality.CAMERA_TOOL_RIGHT_RGB: self._observe_camera_tool_right,
        }

        enable_plot_juggler = cfg.args.enable_plotjuggler
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

        self._cfg_env = cfg.env_cfg
        self._extract_config()
        self._cfg_dr = cfg.dr_cfg
        self._dr_cam_base_offsets: dict[str, np.ndarray] = {}
        self._dr_cam_extrinsics_active: bool = self._cfg_dr.cameras_extrinsics.enabled
        self.device = cfg.get_device()

        print(
            f"[GraspEnv] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self._cameras: dict[str, Camera] = {}
        # TODO(issue#117) redesign the cameras preview feature
        self._debug_cameras: dict[str, Camera] = {}
        self._setup_genesis_scene(cfg=cfg)

        # TODO(issue#41) refactor the camera_setup into more modular system
        self._cameras_link_names = {
            "scene_cam": "cam_scene_rgb_camera_frame",
            "tool_left_cam": "cam_tool_left",
            "tool_right_cam": "cam_tool_right",
        }
        self._cameras_order = {
            "scene_cam": 0,
            "tool_left_cam": 1,
            "tool_right_cam": 2,
        }

        self.scene.build(
            n_envs=self.num_envs
            # env_spacing=(1.0, 1.0),
        )

        self.robot.set_joints_pd_gains()
        self._attach_cameras()

        if self._cfg_dr.enabled:
            self._setup_dr_pd_gains()
            self._cache_camera_base_offsets()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _extract_config(self) -> None:
        # TODO(issue##117) redesign the whole camera preview system
        self.show_cameras_gui = self._cfg_env.visualize_camera

        self.num_obs = self._cfg_env.num_obs
        self.num_privileged_obs = None
        self.num_actions = self._cfg_env.num_actions
        self.image_width = self._cfg_env.image_resolution[0]
        self.image_height = self._cfg_env.image_resolution[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.show_cell = self._cfg_env.visualize_cell
        self.cameras_setup = self._cfg_env.cameras_setup
        self.table_size = self._cfg_env.table_size
        self.workbench_size = self._cfg_env.workbench_size
        self.box_size = self._cfg_env.box_size_default

        self.ctrl_dt = self._cfg_env.ctrl_dt
        self.policy_dt = self._cfg_env.policy_dt
        self.sim_substeps = int(
            math.ceil(self._cfg_env.policy_dt / self._cfg_env.ctrl_dt)
        )
        self.max_episode_length = int(
            math.ceil(self._cfg_env.episode_length_s / self.policy_dt)
        )

        self.max_linear_speed = self._cfg_env.action_max_linear_speed
        self.max_angular_speed = self._cfg_env.action_max_angular_speed

        self.reward_scales = self._cfg_env.reward_scales

    def _setup_genesis_scene(self, cfg: ExpConfig) -> None:
        env_cfg = cfg.env_cfg
        show_viewer = cfg.args.disable_headless
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
                shadow=True,
                plane_reflection=False,
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
                use_rasterizer=env_cfg.use_rasterizer,
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
        self.robot = GenesisManipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            cfg_robot=cfg.robot_cfg,
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

        self.object: BaseBox = ObjectsFactory.create_box(
            scene=self.scene, ctrl=Control.SIM, device=th.device(self.device)
        )
        self.object.create(
            dims=self.box_size,
            pose=None,
            fixed=env_cfg.box_fixed,
            collision=env_cfg.box_collision,
            color=(0.8, 0.0, 0.0),
        )

        # TODO(issue#41) refactor the camera_setup into more modular system
        match self.cameras_setup:
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
        fov: int = 40,  # deg
        lookat: tuple = (0.0, 0.0, 0.0),
        res: Optional[tuple] = None,
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
        if self.cameras_setup != CamerasSetup.DEFAULT:
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
            # TODO(issue#127) migrate this code into Scene/Manipulator layer
            link = self.robot._robot_entity.get_link(link_name)
            for cam_dict in (self._cameras, self._debug_cameras):
                if cam_name in cam_dict:
                    cam_dict[cam_name].attach(link, offset)
                    cam_dict[cam_name].move_to_attach()

    def get_policy_dt(self) -> float:
        return self.policy_dt

    def get_cfg_as_dict(self) -> dict:
        return self._cfg_env.as_dict()

    def get_num_envs(self) -> int:
        return self.num_envs

    def _observe_tcp_pose(self) -> th.Tensor:
        return self.robot.get_tcp_pose()

    def _observe_object_pose(self) -> th.Tensor:
        return self.object.get_pose()

    def _observe_camera_scene(self) -> th.Tensor:
        return self._render_rgb_camera("scene_cam")

    def _observe_camera_tool_left(self) -> th.Tensor:
        return self._render_rgb_camera("tool_left_cam")

    def _observe_camera_tool_right(self) -> th.Tensor:
        return self._render_rgb_camera("tool_right_cam")

    def _render_rgb_camera(self, camera: str) -> th.Tensor:
        rgb, _, _, _ = self._cameras[camera].render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb = rgb.permute(0, 3, 1, 2)[:, :3]  # (B, 3, H, W)
        rgb = th.clamp(rgb, 0.0, 255.0).div_(255.0)
        return rgb

    def _init_reward_functions(self) -> None:
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt * self.sim_substeps
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, unit_length=0.5
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

        # Reset the robot
        self.robot.ctrl_gripper_open(envs_idx)
        self.robot.ctrl_go_to_home(envs_idx)

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

        goal_pose = th.cat([random_pos, goal_yaw], dim=-1)
        self.goal_pose[envs_idx] = goal_pose
        self.object.set_pose(pose=goal_pose, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg_env.episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0

        if self._cfg_dr.enabled:
            self._setup_dr_pd_gains()
            self._randomize_camera_extrinsics(envs_idx)

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
        self.object.set_pose(pose=pose)
        self.goal_pose[:] = pose

    def _reset(self) -> ResetReturn:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=gs.device))
        self._log_state_to_plot_juggler()
        return ResetReturn(self.get_observations(), self.extras)

    def _step(self, actions: th.Tensor) -> StepReturn:
        # Update time
        self.episode_length_buf += 1

        # Environment limitations
        actions = th.clamp(actions, min=-1.0, max=1.0)

        # Applying real-world scaling (with optional noise)
        max_lin_speed, max_ang_speed = self._get_max_speed_coeefs()
        actions[:, :3] *= max_lin_speed
        actions[:, 3:] *= max_ang_speed

        self.robot.ctrl_apply_vel_action(actions, open_gripper=True)
        self.scene.step()

        self._log_state_to_plot_juggler()

        # check termination
        env_reset_idx = self._is_episode_complete()
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
        return StepReturn(obs, reward, dones, self.extras)

    def _get_max_speed_coeefs(self) -> tuple[float, float]:
        cfg = self._cfg_dr.max_speed
        if not cfg.enabled:
            return self.max_linear_speed, self.max_angular_speed

        lin_speed_noise = cfg.linear_speed_noise
        ang_speed_noise = cfg.angular_speed_noise

        lin_scale = (
            1.0 + (th.rand(1, device=self.device).item() * 2.0 - 1.0) * lin_speed_noise
        )
        ang_scale = (
            1.0 + (th.rand(1, device=self.device).item() * 2.0 - 1.0) * ang_speed_noise
        )

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
            self.robot.ctrl_apply_joints_diff_action(joints_diff)
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
                self.robot.ctrl_apply_vel_action(scaled_cart_diff, open_gripper=True)
                for _ in range(steps_per_action):
                    self.scene.step()
                    self._log_state_to_plot_juggler()

        print(f">>> Idling for {idle_steps} steps.")
        for _ in range(idle_steps):
            self.scene.step()
            self._log_state_to_plot_juggler()

    def _is_episode_complete(self) -> th.Tensor:
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

    def _build_agent_observations(self, obs: TensorDict) -> th.Tensor:
        tcp_pose = obs[Modality.TCP_POSE]
        tcp_pos, tcp_quat = tcp_pose[:, :3], tcp_pose[:, 3:]
        obj_pose = obs[Modality.OBJECT_POSE]
        obj_pos, obj_quat = obj_pose[:, :3], obj_pose[:, 3:]

        obs_components = [
            tcp_pos - obj_pos,  # 3D position difference
            tcp_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        return th.cat(obs_components, dim=-1)
        # return TensorDict({"policy": res}, batch_size=self._num_envs, device=self.device)

    def _reward_keypoints(self) -> th.Tensor:
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pos, tcp_quat = tcp_pose[:, :3], tcp_pose[:, 3:]
        keypoints_offset = self.keypoints_offset
        object_offset = th.tensor(
            [0.0, 0.0, -0.08],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)

        finger_pos_keypoints = self._to_world_frame(
            tcp_pos + object_offset,
            tcp_quat,
            keypoints_offset,
        )
        obj_pose = self.object.get_pose()
        object_pos_keypoints = self._to_world_frame(
            obj_pose[:, :3], obj_pose[:, 3:], keypoints_offset
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

    def get_keypoint_offsets(
        self, batch_size: int, unit_length: float = 0.5
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
                device=self.device,
                dtype=th.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    def grasp_and_lift_demo(self) -> float:
        total_steps = self.max_episode_length
        grab_height = 0.08
        goal_pose = self.robot.get_tcp_pose().clone()
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
                self.robot.ctrl_go_to_goal(goal_pose, open_gripper=True)
            elif i < total_steps * 2 / 5:  # grasping
                self.robot.ctrl_go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps * 3 / 5:  # lifting
                self.robot.ctrl_go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 4 / 5:  # final
                self.robot.ctrl_go_to_goal(final_pose, open_gripper=False)
                obj_pos = self.object.get_pose()[:, :3]
                target_pos = final_pose[:, :3]

                dist = th.norm(obj_pos - target_pos, dim=-1)
                in_target = dist < pos_threshold

                hold_counter[in_target] += 1
            else:  # reset
                self.robot.ctrl_go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()

        success = hold_counter >= hold_steps_required
        success_rate = success.float().mean().item()

        return success_rate

    def _setup_dr_pd_gains(self) -> None:
        cfg = self._cfg_dr.pd_gains
        if not cfg.enabled:
            return

        n_dofs = self.robot.get_n_dofs()
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

        self.robot.set_joints_pd_gains(kp_gain=kp_scale, kv_gain=kv_scale)

    def _cache_camera_base_offsets(self) -> None:
        self._dr_cam_base_offsets = {}
        if self.cameras_setup != CamerasSetup.DEFAULT:
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

    def _randomize_camera_extrinsics(self, envs_idx: th.Tensor) -> None:
        cam_cfg = self._cfg_dr.cameras_extrinsics
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
                # TODO(issue#127) change API to expose robot_entity
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

    def _log_state_to_plot_juggler(self) -> None:
        if not self._enable_pj_logging:
            return

        data = {}
        # TODO(issue#128) change api to expose the robot entity
        robot = self.robot._robot_entity
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
