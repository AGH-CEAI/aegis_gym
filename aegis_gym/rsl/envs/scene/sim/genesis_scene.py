from pathlib import Path
from typing import Any, Optional
import time
import warnings

import numpy as np
import genesis as gs
import torch as th
from clearml import Dataset
from genesis.vis.camera import Camera

from ....config_types import EnvCfg, EntityCfg
from ..base_scene import BaseScene
from manipulator import BaseManipulator, GenesisManipulator


class GenesisScene(BaseScene):
    def __init__(self, device: th.device, env_cfg: EnvCfg):
        super().__init__(device=device)
        self.cfg = env_cfg
        self.entities: dict[str, Any] = {}

        self._cameras: dict[str, Camera] = {}
        # TODO(issue#117) redesign the cameras preview feature
        self._debug_cameras: dict[str, Camera] = {}
        self._setup_genesis_scene()

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

        # TODO add robot cfg
        self._robot_entity = self.add_robot(robot_cfg=self.cfg.robot_cfg)
        self._robot_interface = GenesisManipulator(
            robot_entity=self._robot_entity, robot_cfg=self.cfg.robot_cfg, scene=self
        )
        self._robot_interface.set_joints_pd_gains()
        self._attach_cameras()

    def _setup_genesis_scene(self) -> None:
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.cfg.policy_dt,
                substeps=self.cfg.sim_substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.cfg.policy_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_dofs_info=True,  # Enables (n_evs, n_dofs) shape
                batch_links_info=True,  # Enables (n_envs, n_links, ...) shapes
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.cfg.num_envs)),
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
                use_rasterizer=self.cfg.use_rasterizer,
            ),
            show_viewer=self.cfg.show_viewer,
        )

        # == add ground ==
        plane_z = -self.cfg.workbench_size.z if self.cfg.show_cell else 0.0
        self.scene.add_entity(
            gs.morphs.Plane(pos=(0, 0, plane_z)),
            surface=gs.surfaces.Default(color=(0.98, 0.98, 0.98)),
        )

        if self.cfg.show_cell:
            self.table = self.scene.add_entity(
                gs.morphs.Box(
                    size=self.cfg.table_size.as_tuple(),
                    pos=(
                        self.cfg.table_size.x / 2 + self.cfg.workbench_size.x / 2,
                        0.0,
                        self.cfg.table_size.z / 2 - self.cfg.workbench_size.z,
                    ),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=(1.0, 0.96, 0.92)),
                material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
            )

        # == add cameras ==
        # TODO(issue#41) refactor the camera_setup into more modular system
        match self.cfg.camera_setup:
            case "default":
                self._add_camera(name="scene_cam", fov=38)
                self._add_camera(name="tool_left_cam", fov=30)
                self._add_camera(name="tool_right_cam", fov=30)
            case "scene_dual":
                self._add_camera(name="scene_left_cam", pos=(1.25, 0.3, 0.3), fov=60)
                self._add_camera(name="scene_right_cam", pos=(1.25, -0.3, 0.3), fov=60)

        if self.cfg.show_cameras_gui:
            self.record_cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=True,
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
            res = self.cfg.rgb_image_shape.h, self.cfg.rgb_image_shape.w
        self._cameras[name] = self.scene.add_camera(
            res=res,
            pos=pos,
            lookat=lookat,
            fov=fov,
            GUI=self.cfg.show_cameras_gui,
        )
        if self.cfg.show_cameras_gui:
            self._debug_cameras[name] = self.scene.add_camera(
                res=res,
                pos=pos,
                lookat=lookat,
                fov=fov,
                GUI=False,
            )

    def shutdown(self) -> None:
        pass

    def add_entity(self, entity_cfg: EntityCfg) -> Any:
        """Add a given entity to the scene."""
        d_cfg = {
            "size": entity_cfg.size.as_tuple(),
            "fixed": entity_cfg.fixed,
            "collision": entity_cfg.collision,
        }

        # TODO change type to name, configure primitives (box/cylinder/sphere etc.)
        obj = self.scene.add_entity(
            gs.morphs.Box(**d_cfg),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=entity_cfg.color.as_tuple(),
                ),
            ),
        )

        self.entities[entity_cfg.type] = obj
        return obj

    # TODO: move to RobotCfg dataclass
    def add_robot(self, robot_cfg: dict) -> Any:
        # TODO(issue#99): Implement URDF model with cell collision handling
        if self.cfg.show_cell:
            self._urdf_model_id = robot_cfg["urdf_model_id"]["cell"]
        else:
            self._urdf_model_id = robot_cfg["urdf_model_id"]["no_cell"]

        if self._urdf_model_id:
            print(f"[GenesisScene] URDF ClearML dataset ID: {self._urdf_model_id}")
        self._urdf_path = self._resolve_aegis_urdf()
        print(f"[GenesisScene] URDF path: {self._urdf_path}")

        material = gs.materials.Rigid(gravity_compensation=1.0)
        morph = gs.morphs.URDF(
            file=self._urdf_path,
            fixed=True,
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            # TODO(issue#98) move URDF things into the ClearML dataset
            links_to_keep=[
                "ur_base",
                "robotiq_hande_end",
                "cam_tool_right",
                "cam_tool_left",
                "cam_scene_rgb_camera_frame",
            ],
        )
        return self.scene.add_entity(material=material, morph=morph)

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

    def _build(self) -> None:
        self.scene.build(n_envs=self.cfg.num_envs)

    def _attach_cameras(self):
        if self.cfg.camera_setup != "default":
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
            # TODO expose links in the API
            link = self._robot_entity.get_link(link_name)
            for cam_dict in (self._cameras, self._debug_cameras):
                if cam_name in cam_dict:
                    cam_dict[cam_name].attach(link, offset)
                    cam_dict[cam_name].move_to_attach()

    def _get_manipulator(self) -> BaseManipulator:
        return self._robot_interface

    def read_state(self) -> None:
        pass

    def get_n_envs(self) -> int:
        return self.cfg.num_envs
