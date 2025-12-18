import genesis as gs
from genesis.engine.entities import RigidEntity
import torch as th

from ...scene import EntityType, Target, Box


class TargetSimGenesis(Target):
    def __init__(self, scene: gs.Scene, device: str = "cuda"):
        super().__init__(device)
        self._scene = scene
        self._pose: th.Tensor = None
        self._obj: RigidEntity = None

    def create(self) -> None:
        pos = (0.76, -0.1, 0.82)
        self._obj = self._scene.add_entity(
            gs.morphs.Cylinder(height=0.00001, radius=0.04, pos=pos, fixed=True),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )
        pose = [*pos, 1.0, 0.0, 0.0, 0.0]
        self._pose = th.tensor(pose, device=self._device)

    # TODO(issue #29): Verify necessity of tensor cloning
    def set_pose(self, pose: th.Tensor) -> None:
        self._obj.set_pos(pos=pose[:3].view(3), zero_velocity=True)
        self._obj.set_quat(quat=pose[3:].view(4), zero_velocity=True)
        self._pose = pose.clone()

    def get_pose(self) -> th.Tensor:
        pos = th.tensor(self._obj.get_pos(), device=self._device)
        ori = th.tensor(self._obj.get_quat(), device=self._device)
        return th.cat([pos, ori]).clone()


class BoxSimGenesis(Box):
    def __init__(self, scene: gs.Scene, device: str = "cuda"):
        super().__init__(device)
        self._scene = scene
        self._pose: th.Tensor = None
        self._obj: RigidEntity = None

    def create(self) -> None:
        # TODO(issue#37) This should be parametrized (and will be in a future refactor)
        # This is currently based on the robot's cell definition and aegis_moveit_config/config/scene_objects.yaml parameters.
        pos = (0.0, 0.7, -0.84)
        size = (0.04, 0.04, 0.04)
        self._obj = self._scene.add_entity(
            gs.morphs.Box(
                size=size,
                pos=pos,
            ),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0)),
            material=gs.materials.Rigid(rho=8000.0, friction=0.6, coup_friction=0.6),
        )
        pose = [*pos, 1.0, 0.0, 0.0, 0.0]
        self._pose = th.tensor(pose, device=self._device)

    # TODO(issue #29): Verify necessity of tensor cloning
    def set_pose(self, pose: th.Tensor) -> None:
        self._obj.set_pos(pos=pose[:3].view(3), zero_velocity=True)
        self._obj.set_quat(quat=pose[3:].view(4), zero_velocity=True)
        self._pose = pose.clone()

    def get_pose(self) -> th.Tensor:
        pos = th.tensor(self._obj.get_pos(), device=self._device)
        ori = th.tensor(self._obj.get_quat(), device=self._device)
        return th.cat([pos, ori]).clone()


EntityTypeSimGenesis = {
    EntityType.TARGET: TargetSimGenesis,
    EntityType.BOX: BoxSimGenesis,
}
