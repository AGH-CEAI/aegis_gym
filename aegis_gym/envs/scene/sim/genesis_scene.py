import math
from typing import Any

import genesis as gs
import numpy as np
import torch as th
from genesis.vis.camera import Camera

from aegis_gym.config.types import (
    CAMERAS_LINKS,
    CameraLink,
    CameraModality,
    CameraName,
    CamerasSetup,
    Control,
    EnvCfg,
    ExpConfig,
    RobotCfg,
)
from aegis_gym.envs.manipulator import BaseManipulator, GenesisManipulator
from aegis_gym.envs.objects import (
    BaseObject,
    GenesisObjectsFactory,
    ObjectProperties,
    ObjectType,
)
from aegis_gym.envs.plotjuggler_udp import PlotJugglerUDP

from ..base_scene import BaseScene, RandomizationType


class GenesisScene(BaseScene):
    def __init__(
        self,
        cfg: ExpConfig,
        device: th.device,
    ):
        cfg_env = cfg.env_cfg
        cfg_dr = cfg.dr_cfg
        super().__init__(device=device)
        self.CONTROL_TYPE = Control.SIM
        self._randomization_fns = {
            RandomizationType.CAMERAS_EXTRINSICS: self._rand_cameras_extrinsics,
            RandomizationType.MANIPULATOR_MAX_SPEED: self._rand_manipulator_max_speed,
            RandomizationType.MANIPULATOR_PD_GAINS: self._rand_manipulator_pd_gains,
        }
        self._cfg_env = cfg_env
        self._cfg_dr = cfg_dr
        self._extract_config()

        self._enable_pj_logging = cfg.args.enable_plotjuggler
        self._setup_pj_server()

        print(
            f"[GenesisScene] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self._max_linear_speed} m/s ; {self._max_angular_speed} rad/s"
        )

        self._cameras: dict[CameraName, Camera] = {}
        self._cameras_modalities: dict[CameraName, tuple[CameraModality]] = {}
        self._dr_cam_base_offsets: dict[CameraName, np.ndarray] = {}
        self._dr_cam_extrinsics_active: bool = self._cfg_dr.cameras_extrinsics.enabled
        self._setup_scene(cfg=cfg)

        self._global_entity_cnt = 0
        self._entity_registry: dict[int, BaseObject] = {}

    def _extract_config(self) -> None:
        self.num_envs = self._cfg_env.num_envs
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
        self.sim_substeps = math.ceil(self._cfg_env.policy_dt / self._cfg_env.ctrl_dt)
        self.max_episode_length = math.ceil(
            self._cfg_env.episode_length_s / self.policy_dt
        )

        self._max_linear_speed = self._cfg_env.action_max_linear_speed
        self._max_angular_speed = self._cfg_env.action_max_angular_speed

        self.reward_scales = self._cfg_env.reward_scales

    def _setup_pj_server(self) -> None:
        self._pj: PlotJugglerUDP | None = None
        if not self._enable_pj_logging:
            return
        ip = "127.0.0.1"
        port = 9870
        self._pj_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._pj = PlotJugglerUDP(host=ip, port=port)
        print(f"[GraspEnv] Enabled UDP server for PlotJuggler at {ip}:{port}")

    def _setup_scene(self, cfg: ExpConfig) -> None:
        self.gs_scene = self._setup_create_scene(cfg.env_cfg, cfg.args.disable_headless)
        self._setup_create_plane(self.gs_scene)
        self._setup_create_table(self.gs_scene, True)
        self._setup_create_cameras(
            self.gs_scene,
            self.cameras_setup,
            cfg.args.debug_preview_vis_obs,
        )
        self._setup_create_lighting(self.gs_scene)

    def _setup_create_scene(self, env_cfg: EnvCfg, show_viewer: bool) -> gs.Scene:
        return gs.Scene(
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
                max_FPS=60,
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

    def _setup_create_plane(self, scene: gs.Scene) -> None:
        plane_z = -self.workbench_size[2] if self.show_cell else 0.0
        scene.add_entity(
            gs.morphs.Plane(pos=(0, 0, plane_z)),
            surface=gs.surfaces.Default(color=(0.98, 0.98, 0.98)),
        )

    def _setup_create_table(self, scene: gs.Scene, show_cell: bool) -> None:
        if not show_cell:
            return

        self.table = scene.add_entity(
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

    def _setup_create_cameras(
        self, scene: gs.Scene, cameras_setup: CamerasSetup, show_cameras_gui: bool
    ) -> None:

        def_pos = (0.0, 0.0, 0.0)
        match cameras_setup:
            case CamerasSetup.DEFAULT:
                self._setup_add_camera(
                    scene=scene, name=CameraName.CAMERA_SCENE, pos=def_pos, fov=38
                )
                self._setup_add_camera(
                    scene=scene, name=CameraName.CAMERA_TOOL_LEFT, pos=def_pos, fov=30
                )
                self._setup_add_camera(
                    scene=scene, name=CameraName.CAMERA_TOOL_RIGHT, pos=def_pos, fov=30
                )
            case CamerasSetup.SCENE_DUAL:
                self._setup_add_camera(
                    scene=scene,
                    name=CameraName.CAMERA_SCENE_LEFT,
                    pos=(1.25, 0.3, 0.3),
                    fov=60,
                )
                self._setup_add_camera(
                    scene=scene,
                    name=CameraName.CAMERA_SCENE_RIGHT,
                    pos=(1.25, -0.3, 0.3),
                    fov=60,
                )

        if show_cameras_gui:
            self._record_cam = scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=self.show_cameras_gui,
                debug=True,
            )

    def _setup_create_lighting(self, scene: gs.Scene) -> None:
        scene.add_light(
            pos=(0.0, 0.0, 2.46),
            dir=(1.0, 1.0, -1.0),
            color=(1.0, 1.0, 1.0),
            intensity=0.6,
            directional=False,
            castshadow=True,
            cutoff=90.0,
        )

    def _setup_add_camera(
        self,
        scene: gs.Scene,
        name: CameraName,
        pos: tuple[float, float, float],
        fov: int,  # deg
        lookat: tuple[float, float, float] = (0.0, 0.0, 0.0),
        resolution: tuple | None = None,
        show_cameras_gui: bool = False,
    ):
        if resolution is None:
            resolution = (self.image_width, self.image_height)
        self._cameras[name] = scene.add_camera(
            res=resolution,
            pos=pos,
            lookat=lookat,
            fov=fov,
            GUI=show_cameras_gui,
        )
        self._cameras_modalities[name] = tuple(CameraModality.RGB)

    def _setup_attach_cameras(self):
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

        get_link = self.manipulator.get_robot_link
        self._cameras[CameraName.CAMERA_SCENE].attach(
            get_link(link=CameraLink.CAMERA_SCENE), scene_offset_T
        )
        self._cameras[CameraName.CAMERA_SCENE].move_to_attach()
        self._cameras[CameraName.CAMERA_TOOL_LEFT].attach(
            get_link(link=CameraLink.CAMERA_TOOL_LEFT), tool_offset_T
        )
        self._cameras[CameraName.CAMERA_TOOL_LEFT].move_to_attach()
        self._cameras[CameraName.CAMERA_TOOL_RIGHT].attach(
            get_link(link=CameraLink.CAMERA_TOOL_RIGHT), tool_offset_T
        )
        self._cameras[CameraName.CAMERA_TOOL_RIGHT].move_to_attach()

        self._cache_camera_base_offsets(
            offset_scene=scene_offset_T, offset_tool=tool_offset_T
        )

    def _cache_camera_base_offsets(
        self, offset_scene: np.ndarray, offset_tool: np.ndarray
    ) -> None:
        if not self._cfg_dr.enabled:
            return
        if self.cameras_setup != CamerasSetup.DEFAULT:
            raise NotImplementedError(
                f"The cameras setup `{self.cameras_setup}` is not supported."
            )

        self._dr_cam_base_offsets = {
            CameraName.CAMERA_SCENE: offset_scene.copy(),
            CameraName.CAMERA_TOOL_LEFT: offset_tool.copy(),
            CameraName.CAMERA_TOOL_RIGHT: offset_tool.copy(),
        }

    def get_policy_dt(self) -> float:
        return self.policy_dt

    def observe_camera(self, camera: CameraName, modality: CameraModality) -> th.Tensor:
        if not modality == CameraModality.RGB:
            raise ValueError(
                "The current implementation of Genesis Simulator doesn't support different cameras modalities!"
            )
        rgb, _, _, _ = self._cameras[camera].render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb = rgb.permute(0, 3, 1, 2)[:, :3]  # (B, 3, H, W)
        rgb = th.clamp(rgb, 0.0, 255.0).div_(255.0)
        return rgb

    def shutdown(self) -> None:
        if hasattr(self, "manipulator") and self.manipulator is not None:
            del self.manipulator

    def add_entity(self, entity: ObjectType, properties: ObjectProperties) -> Any:
        f = GenesisObjectsFactory(device=self.device, gs_scene=self.gs_scene)
        obj = f.create_object(
            obj_type=entity,
            obj_properties=properties,
        )
        self._entity_registry[self._global_entity_cnt] = obj
        self._global_entity_cnt += 1
        return obj

    def add_manipulator(self, cfg: RobotCfg) -> None:
        self.manipulator = GenesisManipulator(
            num_envs=self.num_envs,
            gs_scene=self.gs_scene,
            cameras_obs_getter=self.observe_camera,
            available_cameras=self._cameras_modalities,
            cfg_robot=cfg,
            show_cell=self.show_cell,
            device=self.device,
        )

    def _build(self) -> None:
        for obj in self._entity_registry.values():
            obj.create()

        self.gs_scene.build(
            n_envs=self.num_envs
            # env_spacing=(1.0, 1.0),
        )
        self._setup_attach_cameras()
        self.manipulator.set_joints_pd_gains()

    def _get_manipulator(self) -> BaseManipulator:
        return self.manipulator

    def update_state(self) -> None:
        self._log_state_to_plot_juggler()

    def pre_step(self) -> None:
        pass

    def step(self) -> None:
        self.gs_scene.step()
        self.update_state()

    def _rand_cameras_extrinsics(self, _envs_idx: th.Tensor) -> None:
        cam_cfg = self._cfg_dr.cameras_extrinsics
        if not (
            cam_cfg.enabled
            and self._dr_cam_extrinsics_active
            and self._dr_cam_base_offsets
        ):
            return

        for cam_name, base_offset in self._dr_cam_base_offsets.items():
            if cam_name == CameraName.CAMERA_SCENE:
                per_cam = cam_cfg.scene_cam
            elif cam_name in (
                CameraName.CAMERA_TOOL_LEFT,
                CameraName.CAMERA_TOOL_RIGHT,
            ):
                per_cam = cam_cfg.tool_cams
            else:
                raise ValueError(f"Wrong camera: {cam_name}")

            perturb = self._make_random_se3_perturbation(
                per_cam.translation_std, per_cam.rotation_std_deg
            )
            perturbed_offset = (base_offset @ perturb).astype(np.float32)

            link = self.manipulator.get_robot_link(link=CAMERAS_LINKS[cam_name])
            self._cameras[cam_name].attach(link, perturbed_offset)
            self._cameras[cam_name].move_to_attach()

    def _make_random_se3_perturbation(
        self,
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

    def _rand_manipulator_max_speed(self, envs_idx: th.Tensor) -> None:
        cfg = self._cfg_dr.max_speed
        if not cfg.enabled:
            return

        lin_speed_noise = cfg.linear_speed_noise
        ang_speed_noise = cfg.angular_speed_noise

        lin_scale = (
            1.0 + (th.rand(1, device=self.device).item() * 2.0 - 1.0) * lin_speed_noise
        )
        ang_scale = (
            1.0 + (th.rand(1, device=self.device).item() * 2.0 - 1.0) * ang_speed_noise
        )

        self.manipulator.max_linear_speed = self._max_linear_speed * lin_scale
        self.manipulator.max_angular_speed = self._max_angular_speed * ang_scale

    def _rand_manipulator_pd_gains(self, envs_idx: th.Tensor) -> None:
        cfg = self._cfg_dr.pd_gains
        if not cfg.enabled:
            return

        n_dofs = self.manipulator.get_n_dofs()
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

        self.manipulator.set_joints_pd_gains(kp_gain=kp_scale, kv_gain=kv_scale)

    def _log_state_to_plot_juggler(self) -> None:
        if not self._enable_pj_logging:
            return

        data = {}
        # TODO(issue#128) change api to expose the robot entity
        robot = self.manipulator._robot_entity
        for name in self._pj_joint_names:
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

        wrench = self.manipulator.get_ft_wrench()

        # Plot only env 0 for now
        w = wrench[0]

        data["ft/force/x"] = float(w[0])
        data["ft/force/y"] = float(w[1])
        data["ft/force/z"] = float(w[2])

        data["ft/torque/x"] = float(w[3])
        data["ft/torque/y"] = float(w[4])
        data["ft/torque/z"] = float(w[5])

        # Joint Torque last sensor
        joint_torque = self.manipulator.get_joint_torque_sensor()
        jt = joint_torque[0]
        data["joint_torque/last_joint"] = float(jt[0])

        self._pj.send(data)
