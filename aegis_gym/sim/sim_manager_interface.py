from abc import ABC, abstractmethod
from typing import Any


class SimManagerInterface(ABC):
    @abstractmethod
    def add_entity(self, entity: Any, **kwargs) -> Any:
        pass

    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def get_robot(self) -> Any:
        pass

    @abstractmethod
    def get_scene(self) -> Any:
        pass
