import torch as th

from ..base_objects import BaseBox, ObjectProperties


class RosGrpcBox(BaseBox):
    def __init__(
        self,
        properties: ObjectProperties,
        device: th.device,
    ):
        super().__init__(properties=properties, device=device)

    def create(self) -> None:
        # TODO(issue#131) read poses from RosGrpc bridge
        self.pose = th.tensor(self.properties.pose, device=self.device).repeat(1, 1)

    def get_pose(self, envs_idx: th.Tensor | int | None = None) -> th.Tensor:
        return self.pose.unsqueeze(0)

    def set_pose(
        self, pose: th.Tensor, envs_idx: th.Tensor | int | None = None
    ) -> None:
        self.pose = pose.squeeze(0)
