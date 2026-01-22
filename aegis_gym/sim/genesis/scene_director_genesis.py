from typing import Optional

import numpy as np
import genesis as gs
import torch as th
from genesis.vis.camera import Camera

from ...scene import (
    SceneDirectorInterface,
    RobotCommanderInterface,
    EntityType,
    SceneEntity,
)
from ...sim import generate_aegis_urdf
from ..utils import Dimensions
from .robot_commander_genesis import RobotCommanderSimGenesis
from .scene_entities_genesis import EntityTypeSimGenesis
from .manipulator import Manipulator

TABLE_SIZE = Dimensions(0.55, 0.84, 0.82)
WORKBENCH_SIZE = Dimensions(0.64, 1.0, 0.806)
MAX_VISUALISATION_ENVS = 10

# TODO(issue#24): Include robot fingers in DOF configuration in Genesis
SIM_CFG = {
    "sim_dt": 0.01,
    "sim_substeps": 2,
    "ctrl_freq": 20,
    "robot_pos": [0.0, 0.0, 0.0],
    "table_size": [TABLE_SIZE.x, TABLE_SIZE.y, TABLE_SIZE.z],
    "table_pos": [
        TABLE_SIZE.x / 2 + WORKBENCH_SIZE.x / 2,
        0.0,
        TABLE_SIZE.z / 2 - WORKBENCH_SIZE.z,
    ],
    "bg_plane_pos": [0.0, 0.0, -TABLE_SIZE.z],
    "dof_names": [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        # 'robotiq_hande_left_finger_joint',
        # 'robotiq_hande_right_finger_joint',
    ],
    # TODO(issue#16): Research tuning of PD gains for UR5e
    "kp": [12000, 12000, 12000, 6000, 6000, 6000],
    "kd": [2000, 2000, 2000, 1000, 1000, 1000],
}


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * th.rand(size=shape, device=device) + lower


class SceneDirectorSimGenesis(SceneDirectorInterface):
    _instance: Optional["SceneDirectorInterface"] = None

    def __new__(cls, *args, **kwargs) -> "SceneDirectorInterface":
        if cls._instance is None:
            cls._instance = super(SceneDirectorInterface, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        device: str = "cuda",
        show_render: bool = True,
        cfg: dict = SIM_CFG,
        enable_scene_camera: bool = False,
        env_cfg: dict = {},
    ):
        super().__init__(device, show_render)
        self.cfg = cfg
        self.enable_scene_camera = enable_scene_camera
        self.motor_dofs: tuple[int] = None
        self.camera = None

        self.visualize_camera = env_cfg.get("visualize_camera", False)
        self.num_envs = env_cfg.get("num_envs", 10)
        self.use_rasterizer = env_cfg.get("use_rasterizer", False)
        self.camera_resolution = env_cfg.get("image_resolution", (64, 64))

        if not gs._initialized:
            backend = gs.gpu if device in ("cuda", "gpu") else gs.cpu
            gs.init(precision="32", backend=backend, logging_level="warning")

        self.ctrl_dt = cfg["sim_dt"]
        self.sim_substeps = cfg["sim_substeps"]
        self.steps_per_ctrl = int(1.0 / (self.ctrl_dt * cfg["ctrl_freq"]))

        self.cameras: dict[str, Camera] = {}
        self._create_scene()
        self._add_ground()
        self._add_robot()
        self._add_table()
        self._add_cameras()

    def _create_scene(self) -> None:
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.ctrl_dt, substeps=self.sim_substeps
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                # enable_self_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(
                    range(min(self.num_envs, MAX_VISUALISATION_ENVS))
                )
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            renderer=gs.options.renderers.BatchRenderer(
                use_rasterizer=self.use_rasterizer,
            ),
            show_viewer=self.show_render,
        )

    def _add_ground(self) -> None:
        plane_pos = self.cfg["bg_plane_pos"] if self.show_cell else (0, 0, 0)
        self.scene.add_entity(gs.morphs.Plane(pos=plane_pos))

    def _add_robot(self) -> None:
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=generate_aegis_urdf(),
                fixed=True,
                pos=self.cfg["robot_pos"],
                links_to_keep=["world", "robotiq_hande_end"],
            ),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=self.env_cfg["robot_cfg"],
            show_cell=self.show_cell,
            device=gs.device,
        )

    def _add_table(self) -> None:
        if not self.show_cell:
            return
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=self.cfg["table_size"],
                pos=self.cfg["table_pos"],
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )

    def _add_cameras(self):
        match self.camera_setup:
            case "default":
                self._add_camera(name="scene_cam", fov=40)
                self._add_camera(name="tool_left_cam", fov=30)
                self._add_camera(name="tool_right_cam", fov=30)
            case "scene_dual":
                self._add_camera(name="scene_left_cam", pos=(1.25, 0.3, 0.3), fov=60)
                self._add_camera(name="scene_right_cam", pos=(1.25, -0.3, 0.3), fov=60)

        if not self.visualize_camera:
            return
        self.record_cam = self.scene.add_camera(
            res=(1280, 720),
            pos=(1.5, 0.0, 0.2),
            lookat=(0.0, 0.0, 0.2),
            fov=60,
            GUI=True,
            debug=True,
        )

    def _add_camera(
        self,
        name: str,
        pos: tuple = (0.0, 0.0, 0.0),
        fov: float = 40,  # deg
        lookat: tuple = (0.0, 0.0, 0.0),
        res: tuple = None,
    ):
        if res is None:
            res = self.camera_resolution
        self.cameras[name] = self.scene.add_camera(
            res=res,
            pos=pos,
            lookat=lookat,
            fov=fov,
            GUI=self.visualize_camera,
        )

    def get_robot_commander(self) -> RobotCommanderInterface:
        return RobotCommanderSimGenesis(self.robot, self.motor_dofs, self.device)

    def shutdown(self) -> None:
        pass

    def add_entity(self, entity: EntityType) -> SceneEntity:
        return EntityTypeSimGenesis[entity](self.scene)

    def build(self, n_envs: int = 1) -> None:
        self.scene.build(n_envs=n_envs)

        self.motor_dofs = tuple(
            [
                self.robot.get_joint(name).dofs_idx_local[0]
                for name in self.cfg["dof_names"]
            ]
        )

        self.robot.set_pd_gains()
        self._attach_cameras()

    def _attach_cameras(self) -> None:
        if self.camera_setup != "default":
            return

        scene_offset_T = np.array(
            [
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
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

        gs_robot = self.robot.get_robot_entity()
        for cam_name, link_name, offset in cams_to_attach:
            cam = self.cameras[cam_name]
            cam.attach(gs_robot.get_link(link_name), offset)
            cam.move_to_attach()

    def step(self) -> None:
        for _ in range(self.steps_per_ctrl):
            self.scene.step()
