from aegis_gym.sim import generate_aegis_urdf
from aegis_gym.sim.sim_manager_interface import SimManagerInterface
import genesis as gs


SIM_CFG = {
    "dt": 0.05,
    "robot_pos": [0.0, 0.0, 0.0],
    "table_pos": [0.0, 0.6, 0.41],
}


class SimManagerGenesis(SimManagerInterface):
    def __init__(
        self,
        show_viewer: bool = False,
        device: str = "cuda",
        cfg: dict = SIM_CFG,
    ):
        super().__init__()

        if not gs._initialized:
            # TODO make it more flexible
            backend = gs.gpu if device in ("cuda", "gpu") else gs.cpu
            gs.init(precision="32", backend=backend, logging_level="warning")
        self.dt = cfg["dt"]

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
                pos=cfg["robot_pos"],
            ),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )

        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.84, 0.55, 0.82),
                pos=cfg["table_pos"],
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )

    def add_entity(self, entity: gs.Morph, **kwargs) -> gs.Entity:
        return self.scene.add_entity(entity, **kwargs)

    def build(self) -> None:
        self.scene.build()

    def step(self) -> None:
        self.scene.step()

    def get_robot(self) -> gs.Entity:
        return self.robot

    def get_scene(self) -> gs.Entity:
        return self.scene
