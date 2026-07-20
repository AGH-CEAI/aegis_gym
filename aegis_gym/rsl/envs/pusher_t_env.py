import math
from typing import Optional

import genesis as gs
import numpy as np
import torch as th
from tensordict import TensorDict
from genesis.vis.camera import Camera

from .manipulator import GenesisManipulator
from .base_env import BaseEnv, StepReturn, ResetReturn, Modality
from .objects import BaseTBlock, ObjectsFactory
from .plotjuggler_udp import PlotJugglerUDP

from config.types import ExpConfig, CameraPoseCfg, Control, CamerasSetup

T_BLOCK_MASS_KG = 0.2
TARGET_THICKNESS_RATIO = 0.05


class PusherTEnv(BaseEnv):
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
            print(f"[PusherTEnv] Enabled UDP server for PlotJuggler at {ip}:{port}")
        self._enable_pj_logging = enable_plot_juggler

        self._cfg_env = cfg.env_cfg
        self._extract_config()
        self._cfg_dr = cfg.dr_cfg
        self._dr_cam_base_offsets: dict[str, np.ndarray] = {}
        self._dr_cam_extrinsics_active: bool = self._cfg_dr.cameras_extrinsics.enabled
        self.device = cfg.get_device()

        print(
            f"[PusherTEnv] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self._cameras: dict[str, Camera] = {}
        self._debug_cameras: dict[str, Camera] = {}
        self._setup_genesis_scene(cfg=cfg)

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

        self.scene.build(n_envs=self.num_envs)
        self.object.set_mass(T_BLOCK_MASS_KG)

        self.robot.set_joints_pd_gains()
        self._attach_cameras()

        if self._cfg_dr.enabled:
            self._setup_dr_pd_gains()
            self._cache_camera_base_offsets()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _extract_config(self) -> None:
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
        self.t_block_scale = self._cfg_env.t_block_scale

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

        # Success thresholds (position in meters, orientation in radians).
        self.success_pos_threshold = 0.02
        self.success_rot_threshold = math.radians(10.0)
        self.position_reward_temp = 0.08
        self.orientation_reward_temp = 0.5

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
                batch_dofs_info=True,
                batch_links_info=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.num_envs)),
                shadow=True,
                plane_reflection=False,
            ),
            viewer_options=gs.options.ViewerOptions(
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

        table_top_z = self.table_size[2] - self.workbench_size[2]
        self._target_pos = (0.47, 0.0, table_top_z)

        # == add T-block ==
        self.object: BaseTBlock = ObjectsFactory.create_t_block(
            scene=self.scene, ctrl=Control.SIM, device=th.device(self.device)
        )
        self.object.create(
            dims=(self.t_block_scale,),
            pose=None,
            fixed=False,
            collision=True,
            color=(0.0, 0.0, 0.0),
        )

        self.target: BaseTBlock = ObjectsFactory.create_t_block(
            scene=self.scene, ctrl=Control.SIM, device=th.device(self.device)
        )
        self.target.create(
            dims=(
                self.t_block_scale,
                self.t_block_scale,
                self.t_block_scale * TARGET_THICKNESS_RATIO,
            ),
            pose=(*self._target_pos, 1.0, 0.0, 0.0, 0.0),
            fixed=True,
            collision=False,
            color=(0.1, 0.8, 0.3),
            opacity=0.9,
        )

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

    def _init_buffers(self) -> None:
        self.episode_length_buf = th.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.reset_buf = th.zeros(self.num_envs, dtype=th.bool, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: th.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        self.robot.ctrl_gripper_close(envs_idx)
        self.robot.ctrl_go_to_home(envs_idx)

        num_reset = len(envs_idx)
        random_x = th.rand(num_reset, device=self.device) * 0.22 + 0.36  # 0.36 - 0.58
        random_y = (th.rand(num_reset, device=self.device) - 0.5) * 0.4  # -0.2 - 0.2
        table_top_z = self.table_size[2] - self.workbench_size[2]
        random_z = th.ones(num_reset, device=self.device) * table_top_z
        random_pos = th.stack([random_x, random_y, random_z], dim=-1)

        random_yaw = th.rand(num_reset, device=self.device) * 2 * math.pi - math.pi
        random_quat = th.stack(
            [
                th.cos(random_yaw / 2),
                th.zeros(num_reset, device=self.device),
                th.zeros(num_reset, device=self.device),
                th.sin(random_yaw / 2),
            ],
            dim=-1,
        )

        object_pose = th.cat([random_pos, random_quat], dim=-1)
        self.object.set_pose(pose=object_pose, envs_idx=envs_idx)

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

    def _reset(self) -> ResetReturn:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=gs.device))
        self._log_state_to_plot_juggler()
        return ResetReturn(self.get_observations(), self.extras)

    def _step(self, actions: th.Tensor) -> StepReturn:
        self.episode_length_buf += 1

        actions = th.clamp(actions, min=-1.0, max=1.0)

        max_lin_speed, max_ang_speed = self._get_max_speed_coeefs()
        actions[:, :3] *= max_lin_speed
        actions[:, 3:] *= max_ang_speed

        # the gripper stays closed: it acts as a rigid pusher tip
        self.robot.ctrl_apply_vel_action(actions, open_gripper=False)
        self.scene.step()

        self._log_state_to_plot_juggler()

        env_reset_idx = self._is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        reward = th.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

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

    def _is_episode_complete(self) -> th.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf = time_out_buf

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
        goal_pose = self.target.get_pose()
        goal_pos, goal_quat = goal_pose[:, :3], goal_pose[:, 3:]

        obs_components = [
            tcp_pos - obj_pos,  # 3D position difference (reach)
            tcp_quat,  # current tcp orientation (w, x, y, z)
            obj_pos - goal_pos,  # object-to-goal position difference
            obj_quat,  # current object orientation (w, x, y, z)
            goal_quat,  # target orientation (w, x, y, z)
        ]
        return th.cat(obs_components, dim=-1)

    @staticmethod
    def _quat_angle_diff(quat_a: th.Tensor, quat_b: th.Tensor) -> th.Tensor:
        cos_half_angle = th.clamp(th.abs((quat_a * quat_b).sum(-1)), 0.0, 1.0)
        return 2.0 * th.acos(cos_half_angle)

    def _reward_reach(self) -> th.Tensor:
        tcp_pos = self.robot.get_tcp_pose()[:, :3]
        obj_pos = self.object.get_pose()[:, :3]
        dist = th.norm(tcp_pos - obj_pos, p=2, dim=-1)
        return th.exp(-dist)

    def _reward_position(self) -> th.Tensor:
        obj_pos = self.object.get_pose()[:, :3]
        goal_pos = self.target.get_pose()[:, :3]
        dist = th.norm(obj_pos[:, :2] - goal_pos[:, :2], p=2, dim=-1)
        return th.exp(-dist / self.position_reward_temp)

    def _reward_orientation(self) -> th.Tensor:
        obj_quat = self.object.get_pose()[:, 3:]
        goal_quat = self.target.get_pose()[:, 3:]
        angle = self._quat_angle_diff(obj_quat, goal_quat)
        return th.exp(-angle / self.orientation_reward_temp)

    def _reward_success(self) -> th.Tensor:
        obj_pose = self.object.get_pose()
        goal_pose = self.target.get_pose()
        pos_dist = th.norm(obj_pose[:, :2] - goal_pose[:, :2], p=2, dim=-1)
        angle = self._quat_angle_diff(obj_pose[:, 3:], goal_pose[:, 3:])
        success = (pos_dist < self.success_pos_threshold) & (
            angle < self.success_rot_threshold
        )
        return success.float()

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
        robot = self.robot._robot_entity
        for name in self._joint_names:
            j = robot.get_joint(name=name)
            for idx in j.dofs_idx_local:
                pos = robot.get_dofs_position([idx])
                vel = robot.get_dofs_velocity([idx])
                force = robot.get_dofs_force([idx])

                data[f"joint_states/{name}/position"] = float(pos.flatten()[0])
                data[f"joint_states/{name}/velocity"] = float(vel.flatten()[0])
                data[f"joint_states/{name}/effort"] = float(force.flatten()[0])

        all_link_positions = robot.get_links_pos()
        link_positions = all_link_positions[0]

        ee_idx = -1  # Last link = end effector
        position = link_positions[ee_idx]

        data["ee/position/x"] = float(position[0])
        data["ee/position/y"] = float(position[1])
        data["ee/position/z"] = float(position[2])

        self._pj.send(data)
