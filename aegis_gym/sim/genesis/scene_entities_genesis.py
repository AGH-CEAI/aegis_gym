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
    def __init__(self, scene: gs.Scene, n_envs: int = 1, device: str = "cuda"):
        super().__init__(device)
        self._scene = scene
        self._pose: th.Tensor = None
        self._obj: RigidEntity = None
        self.n_envs = n_envs

    def create(self, cfg: dict) -> None:
        self._size = cfg.get("box_size", (0.03, 0.08, 0.06))
        self._fixed = cfg.get("box_fixed", True)
        self._collision = cfg.get("box_collision", False)
        self._color = cfg.get("box_color", (1.0, 0.0, 0.0))

        pos = (0.0, 0.7, -0.84)
        self._obj = self._scene.add_entity(
            gs.morphs.Box(
                size=self._size,
                fixed=self._fixed,
                collision=self._collision,
                pos=pos,
            ),
            # material=gs.materials.Rigid(rho=8000.0, friction=0.6, coup_friction=0.6),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=self._color,
                ),
            ),
        )
        self._pose = th.tensor([*pos, 1.0, 0.0, 0.0, 0.0], device=self._device)

    # TODO(issue #29): Verify necessity of tensor cloning
    def set_pose(self, pose: th.Tensor, envs_idx: int = 0) -> None:
        self._obj.set_pos(pos=pose[:3].view(3), zero_velocity=True, envs_idx=envs_idx)
        self._obj.set_quat(quat=pose[3:].view(4), zero_velocity=True, envs_idx=envs_idx)
        self._pose = pose.clone()

    def set_pos(self, pos: th.Tensor, envs_idx: int = 0) -> None:
        self._obj.set_pos(pos, envs_idx=envs_idx)

    def set_quat(self, quat: th.Tensor, envs_idx: int = 0) -> None:
        self._obj.set_quat(quat, envs_idx=envs_idx)

    def get_pose(self) -> th.Tensor:
        pos = th.tensor(self._obj.get_pos(), device=self._device)
        ori = th.tensor(self._obj.get_quat(), device=self._device)
        return th.cat([pos, ori]).clone()

    def get_pos(self) -> th.Tensor:
        return self._obj.get_pos()

    def get_quat(self) -> th.Tensor:
        return self._obj.get_quat()


EntityTypeSimGenesis = {
    EntityType.TARGET: TargetSimGenesis,
    EntityType.BOX: BoxSimGenesis,
}
