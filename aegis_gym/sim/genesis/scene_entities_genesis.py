import numpy as np
import genesis as gs

from ...scene import EntityType, Target, Box


class TargetSimGenesis(Target):
    def __init__(self, scene: gs.Scene):
        super().__init__()
        self._scene = scene
        # TODO: set the typings for these vars
        self._pose: np.ndarray = None
        self._obj = None

    def create(self) -> None:
        pose = (-0.1, 0.76, 0.82)
        self._obj = self._scene.add_entity(
            gs.morphs.Cylinder(height=0.00001, radius=0.04, pos=pose, fixed=True),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )
        self._pose = np.array(pose)

    def set_pose(self, pose: np.ndarray) -> None:
        print("TODO: set_pose() for TargetSimGensis is not yet implemented.")

    def get_pose(self) -> np.ndarray:
        return self._pose


class BoxSimGenesis(Box):
    def __init__(self, scene: gs.Scene):
        super().__init__()
        self._scene = scene
        # TODO: set the typings for these vars
        self._pose: np.ndarray = None
        self._obj = None

    def create(self) -> None:
        pose = (0.0, 0.7, 0.84)
        size = (0.04, 0.04, 0.04)

        self._obj = self._scene.add_entity(
            gs.morphs.Box(
                size=size,
                pos=pose,
            ),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0)),
            material=gs.materials.Rigid(rho=8000.0, friction=0.6, coup_friction=0.6),
        )
        self._pose = np.array(pose)

    def set_pose(self, pose: np.ndarray) -> None:
        print("TODO: set_pose() for BoxSimGensis is not yet implemented.")

    def get_pose(self) -> np.ndarray:
        return self._pose


EntityTypeSimGenesis = {
    EntityType.TARGET: TargetSimGenesis,
    EntityType.BOX: BoxSimGenesis,
}
