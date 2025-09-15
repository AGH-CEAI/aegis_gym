import genesis as gs
import torch as th

from ...envs import EnvRenderMode
from ...scene import (
    SceneDirectorInterface,
    RobotCommanderInterface,
    EntityType,
    SceneEntity,
)
from ...sim import generate_aegis_urdf
from .robot_commander_genesis import RobotCommanderSimGenesis
from .scene_entities_genesis import EntityTypeSimGenesis

SIM_CFG = {
    "dt": 0.05,
    "robot_pos": [0.0, 0.0, 0.0],
    "table_pos": [0.0, 0.6, 0.41],
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
    "kp": [600, 600, 400, 400, 200, 200],
    "kd": [60, 60, 40, 40, 20, 20],
}


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * th.rand(size=shape, device=device) + lower


class SceneDirectorSimGenesis(SceneDirectorInterface):
    def __init__(
        self,
        device: str = "cuda",
        render_mode=EnvRenderMode.RGB_ARRAY,
        cfg: dict = SIM_CFG,
    ):
        super().__init__(device, render_mode)
        self.cfg = cfg
        self.motor_dofs: tuple[str] = None

        if not gs._initialized:
            backend = gs.gpu if device in ("cuda", "gpu") else gs.cpu
            gs.init(precision="32", backend=backend, logging_level="warning")
        self.dt = cfg["dt"]

        self._create_scene()

    def _create_scene(self) -> None:
        show_viewer = True if self.render_mode == EnvRenderMode.HUMAN else False
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                # enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=generate_aegis_urdf(),
                fixed=True,
                pos=self.cfg["robot_pos"],
            ),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )

        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.84, 0.55, 0.82),
                pos=self.cfg["table_pos"],
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )

    def get_robot_commander(self) -> RobotCommanderInterface:
        return RobotCommanderSimGenesis(self.robot, self.motor_dofs, self.device)

    def shutdown(self) -> None:
        pass

    def add_entity(self, entity: EntityType) -> SceneEntity:
        return EntityTypeSimGenesis[entity](self.scene)

    def build(self) -> None:
        self.scene.build()

        self.motor_dofs = tuple(
            [self.robot.get_joint(name).dof_idx_local for name in self.cfg["dof_names"]]
        )
        self.robot.set_dofs_kp(self.cfg["kp"], self.motor_dofs)
        self.robot.set_dofs_kv(self.cfg["kd"], self.motor_dofs)

    def step(self) -> None:
        self.scene.step()
