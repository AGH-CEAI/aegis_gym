import numpy as np
from typing import Any

from .robot_commander_mock import RobotCommanderMock
from .scene_director_interface import SceneDirectorInterface
from .scene_entities import EntityType, SceneEntity, Target, Box


class TargetMock(Target):
    def __init__(self):
        super().__init__()

    def create(self) -> None:
        pass

    def set_pose(self, pose: Any) -> None:
        pass

    def get_pose(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)


class BoxMock(Box):
    def __init__(self):
        super().__init__()

    def create(self) -> None:
        pass

    def set_pose(self, pose: Any) -> None:
        pass

    def get_pose(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)


EntityTypeMock = {
    EntityType.TARGET: TargetMock,
    EntityType.BOX: BoxMock,
}


class SceneDirectorMock(SceneDirectorInterface):
    def __init__(self):
        super().__init__()

    def get_robot_commander(self) -> RobotCommanderMock:
        return RobotCommanderMock()

    def shutdown(self) -> None:
        pass

    def add_entity(self, entity: EntityType, pose: Any, **kwargs) -> SceneEntity:
        return EntityTypeMock[entity]()

    def build(self) -> None:
        pass

    def step(self) -> None:
        pass
