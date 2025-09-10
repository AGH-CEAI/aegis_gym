from abc import ABC, abstractmethod
from typing import Any


class SimManagerInterface(ABC):
    @abstractmethod
    def add_entity(self, entity: Any, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_robot(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_scene(self) -> Any:
        raise NotImplementedError
