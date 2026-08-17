import genesis as gs
import torch as th

from ..base_objects import ObjectProperties
from ..base_objects_factory import BaseObjectsFactory
from .genesis_objects import GenesisBox, GenesisMesh, GenesisURDF


class GenesisObjectsFactory(BaseObjectsFactory):
    def __init__(self, device: th.device, gs_scene: gs.Scene):
        super().__init__(device=device)
        self._gs_scene = gs_scene

    def create_box(self, properties: ObjectProperties) -> GenesisBox:
        return GenesisBox(
            gs_scene=self._gs_scene, properties=properties, device=self.device
        )

    def create_mesh(self, properties: ObjectProperties) -> GenesisMesh:
        return GenesisMesh(
            gs_scene=self._gs_scene, properties=properties, device=self.device
        )

    def create_urdf(self, properties: ObjectProperties) -> GenesisURDF:
        return GenesisURDF(
            gs_scene=self._gs_scene, properties=properties, device=self.device
        )
