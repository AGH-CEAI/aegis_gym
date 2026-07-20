from pathlib import Path
from typing import Optional

import torch as th
import genesis as gs

from ..base_objects import BaseBox, BaseTBlock
from envs.scene import BaseScene

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


class GenesisBox(BaseBox):
    def __init__(self, scene: BaseScene | gs.Scene, device: th.device):
        super().__init__(scene=scene, device=device)

    def create(
        self,
        dims: tuple[float, float, float],
        pose: Optional[tuple],
        fixed: bool = False,
        collision: bool = False,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> None:
        self._obj = self._scene.add_entity(
            gs.morphs.Box(
                size=dims,
                pos=(0, 0, 0) if not pose else pose[:3],
                quat=None if not pose else pose[3:],
                fixed=fixed,
                collision=collision,
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=color,
                ),
            ),
        )

    def get_pose(self, envs_idx: Optional[th.Tensor | int] = None) -> th.Tensor:
        pos = self._obj.get_pos()
        quat = self._obj.get_quat()
        res = th.cat([pos, quat], dim=1)
        if envs_idx is None:
            return res
        return res[envs_idx, :]

    def set_pose(
        self, pose: th.Tensor, envs_idx: Optional[th.Tensor | int] = None
    ) -> None:
        self._obj.set_pos(pos=pose[:, :3], envs_idx=envs_idx)
        self._obj.set_quat(quat=pose[:, 3:], envs_idx=envs_idx)


class GenesisTBlock(BaseTBlock):
    def __init__(self, scene: BaseScene | gs.Scene, device: th.device):
        super().__init__(scene=scene, device=device)

    def create(
        self,
        dims: tuple[float, ...],
        pose: Optional[tuple],
        fixed: bool = False,
        collision: bool = False,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        opacity: Optional[float] = None,
    ) -> None:
        scale = tuple(dims) if len(dims) == 3 else (float(dims[0]) if dims else 1.0)
        self._obj = self._scene.add_entity(
            gs.morphs.Mesh(
                file=str(_ASSETS_DIR / "T_shape.stl"),
                scale=scale,
                pos=(0, 0, 0) if not pose else pose[:3],
                quat=None if not pose else pose[3:],
                fixed=fixed,
                collision=collision,
                align=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=color,
                ),
                opacity=opacity,
            ),
        )

    def get_pose(self, envs_idx: Optional[th.Tensor | int] = None) -> th.Tensor:
        pos = self._obj.get_pos()
        quat = self._obj.get_quat()
        res = th.cat([pos, quat], dim=1)
        if envs_idx is None:
            return res
        return res[envs_idx, :]

    def set_pose(
        self, pose: th.Tensor, envs_idx: Optional[th.Tensor | int] = None
    ) -> None:
        self._obj.set_pos(pos=pose[:, :3], envs_idx=envs_idx)
        self._obj.set_quat(quat=pose[:, 3:], envs_idx=envs_idx)

    def set_mass(self, mass: float) -> None:
        self._obj.set_mass(mass)
