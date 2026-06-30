from typing import Optional

import torch as th
import genesis as gs

from ..base_objects import BaseBox
from ...scene import BaseScene


class RosGrpcBox(BaseBox):
    def __init__(
        self,
        scene: BaseScene | gs.Scene,  # TODO(issue128) change BaseScene to RosGrpcScene
        device: th.device,
    ):
        super().__init__(scene=scene, device=device)

    def create(self, pose: tuple, *args, **kwargs) -> None:
        # TODO(issue#131) read poses from RosGrpc bridge
        self.pose = th.tensor(pose, device=self.device).repeat(1, 1)

    def get_pose(self, envs_idx: Optional[th.Tensor | int] = None) -> th.Tensor:
        return self.pose

    def set_pose(self, pose: th.Tensor, envs_idx: Optional[th.Tensor | int]) -> None:
        self.pose = pose
