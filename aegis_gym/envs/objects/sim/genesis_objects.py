import genesis as gs
import torch as th

from ..base_objects import BaseBox, BaseMesh, ObjectProperties


class GenesisBox(BaseBox):
    def __init__(
        self, gs_scene: gs.Scene, properties: ObjectProperties, device: th.device
    ):
        super().__init__(properties=properties, device=device)
        self._gs_scene = gs_scene

    def create(
        self,
    ) -> None:
        p = self.properties
        self._obj = self._gs_scene.add_entity(
            gs.morphs.Box(
                size=p.dims,
                pos=(0, 0, 0) if not p.pose else p.pose[:3],
                quat=None if not p.pose else p.pose[3:],
                fixed=p.fixed,
                collision=p.collision,
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=p.color,
                ),
            ),
        )

    def get_pose(self, envs_idx: th.Tensor | int | None = None) -> th.Tensor:
        pos = self._obj.get_pos()
        quat = self._obj.get_quat()
        res = th.cat([pos, quat], dim=1)
        if envs_idx is None:
            return res
        return res[envs_idx, :]

    def set_pose(
        self, pose: th.Tensor, envs_idx: th.Tensor | int | None = None
    ) -> None:
        self._obj.set_pos(pos=pose[:, :3], envs_idx=envs_idx)
        self._obj.set_quat(quat=pose[:, 3:], envs_idx=envs_idx)


class GenesisMesh(BaseMesh):
    def __init__(
        self, gs_scene: gs.Scene, properties: ObjectProperties, device: th.device
    ):
        super().__init__(properties=properties, device=device)
        self._gs_scene = gs_scene

    def create(
        self,
    ) -> None:
        p = self.properties
        material = None
        if p.friction is not None:
            material = gs.materials.Rigid(friction=p.friction, coup_friction=p.friction)

        self._obj = self._gs_scene.add_entity(
            gs.morphs.Mesh(
                file=p.mesh_path,
                pos=(0, 0, 0) if not p.pose else p.pose[:3],
                quat=None if not p.pose else p.pose[3:],
                scale=p.scale,
                fixed=p.fixed,
                collision=p.collision,
                convexify=False,
                decompose_nonconvex=True,
                decimate=False,
            ),
            material=material,
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=p.color,
                ),
            ),
        )

    def get_pose(self, envs_idx: th.Tensor | int | None = None) -> th.Tensor:
        pos = self._obj.get_pos()
        quat = self._obj.get_quat()
        res = th.cat([pos, quat], dim=1)
        if envs_idx is None:
            return res
        return res[envs_idx, :]

    def set_pose(
        self, pose: th.Tensor, envs_idx: th.Tensor | int | None = None
    ) -> None:
        self._obj.set_pos(pos=pose[:, :3], envs_idx=envs_idx)
        self._obj.set_quat(quat=pose[:, 3:], envs_idx=envs_idx)
