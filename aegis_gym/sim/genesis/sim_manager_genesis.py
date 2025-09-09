from ...sim import generate_aegis_urdf, SimManagerInterface
import genesis as gs
import torch as th

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
    # TODO Take HOME position from SRDF file.
    "default_joint_angles": {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -2.10,
        "elbow_joint": 2.10,
        "wrist_1_joint": -1.57,
        "wrist_2_joint": -1.57,
        "wrist_3_joint": 0.0,
        # "robotiq_hande_left_finger_joint": 0.025,
        # "robotiq_hande_right_finger_joint": 0.025,
    },
}


class SimManagerGenesis(SimManagerInterface):
    def __init__(
        self,
        show_viewer: bool = False,
        device: str = "cuda",
        cfg: dict = SIM_CFG,
    ):
        super().__init__()
        self.device = device
        self.cfg = cfg

        if not gs._initialized:
            # TODO make it more flexible
            backend = gs.gpu if device in ("cuda", "gpu") else gs.cpu
            gs.init(precision="32", backend=backend, logging_level="warning")
        self.dt = cfg["dt"]

        self.dof_home = th.tensor(
            [cfg["default_joint_angles"][name] for name in cfg["dof_names"]],
            device=device,
        )

        self._create_scene(show_viewer)

    def _create_scene(self, show_viewer: bool) -> None:
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

    def add_entity(
        self, entity: gs.morphs.Morph, **kwargs
    ) -> gs.engine.entities.base_entity.Entity:
        return self.scene.add_entity(entity, **kwargs)

    def build(self) -> None:
        self.scene.build()

        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local for name in self.cfg["dof_names"]
        ]
        self.robot.set_dofs_kp(self.cfg["kp"], self.motor_dofs)
        self.robot.set_dofs_kv(self.cfg["kd"], self.motor_dofs)

    def step(self) -> None:
        self.scene.step()

    def get_robot(self) -> gs.engine.entities.base_entity.Entity:
        return self.robot

    def get_scene(self) -> gs.engine.entities.base_entity.Entity:
        return self.scene
