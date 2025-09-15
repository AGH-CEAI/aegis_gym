from abc import ABC, abstractmethod
from typing import Any

from ..envs.env_types import EnvRenderMode
from .robot_commander_interface import RobotCommanderInterface
from .scene_entities import EntityType, SceneEntity


# Usage:
# 1) Create a selected implementation
# 2) Add entities with add_entity()
# 3) Build the scene with build()
# 4) Get the RobotCommander with get_robot_commander() and interacte
# 4) "Simulate" next step with step()
class SceneDirectorInterface(ABC):
    def __init__(self, device: str, render_mode: EnvRenderMode) -> None:
        super().__init__()
        self.device = device
        self.render_mode = render_mode

    @abstractmethod
    def get_robot_commander(self) -> RobotCommanderInterface:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def add_entity(self, entity: EntityType) -> Any[SceneEntity]:
        raise NotImplementedError

    @abstractmethod
    def build(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError
