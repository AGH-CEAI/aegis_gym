from typing import Any
import torch as th

from ..envs.env_types import EnvRenderMode
from .robot_commander_mock import RobotCommanderMock
from .scene_director_interface import SceneDirectorInterface
from .scene_entities import EntityType, SceneEntity, Target, Box


class TargetMock(Target):
    def __init__(self):
        super().__init__(device="cpu")

    def create(self) -> None:
        pass

    def set_pose(self, pose: Any) -> None:
        pass

    def get_pose(self) -> th.Tensor:
        return th.tensor([0.0, 0.0, 0.0], dtype=th.float32, device=self._device)


class BoxMock(Box):
    def __init__(self):
        super().__init__(device="cpu")

    def create(self) -> None:
        pass

    def set_pose(self, pose: Any) -> None:
        pass

    def get_pose(self) -> th.Tensor:
        return th.tensor([0.0, 0.0, 0.0], dtype=th.float32, device=self._device)


EntityTypeMock = {
    EntityType.TARGET: TargetMock,
    EntityType.BOX: BoxMock,
}


class SceneDirectorMock(SceneDirectorInterface):
    def __init__(self):
        super().__init__(device="cpu", render_mode=EnvRenderMode.NONE)

    def get_robot_commander(self) -> RobotCommanderMock:
        return RobotCommanderMock(self.device)

    def shutdown(self) -> None:
        pass

    def add_entity(self, entity: EntityType, pose: Any, **kwargs) -> SceneEntity:
        return EntityTypeMock[entity]()

    def build(self) -> None:
        pass

    def step(self) -> None:
        pass
