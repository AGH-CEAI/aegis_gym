import torch as th
from ..scene import EntityType, Target, Box


class TargetROS(Target):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._pose: th.Tensor = None

    def create(self) -> None:
        pass

    def set_pose(self, pose: th.Tensor) -> None:
        self._pose = pose

    def get_pose(self) -> th.Tensor:
        return self._pose


class BoxROS(Box):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        print(
            ">>>> WARNING <<<< The Box scene object is not implemented for ROS usage."
        )

    def create(self) -> None:
        pass

    def set_pose(self, pose: th.Tensor) -> None:
        pass

    def get_pose(self) -> th.Tensor:
        # TODO implement
        return th.zeros()


EntityTypeROS = {
    EntityType.TARGET: TargetROS,
    EntityType.BOX: BoxROS,
}
