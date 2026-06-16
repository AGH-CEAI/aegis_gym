from abc import ABC, abstractmethod
import torch as th

from ..manipulator import BaseManipulator


class BaseScene(ABC):
    """
    Base class for implementing whole interaction with simulator or real world.
    """

    def __init__(self, device: th.device = th.device("cpu")):
        super().__init__()
        self._is_build = False
        self.device: th.device = device

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the scene connection."""
        ...

    @abstractmethod
    def add_entity(self, entity: str) -> None:
        # TODO implement entity enum
        """Add a given entity to the scene."""
        ...

    @abstractmethod
    def add_robot(self) -> None:
        """Add the Aegis robot to the scene."""
        ...

    def build(self) -> None:
        """
        Build the scenem, based on previously added entities (`add_entity()`) and robot (`add_robot()`).
        Must be called before any robot control.
        """
        if self._is_build:
            raise RuntimeError("Scene should be build only once!")
        self._build()
        self._is_build = True

    @abstractmethod
    def _build(self) -> None: ...

    def get_manipulator(self) -> BaseManipulator:
        """
        Return the `Manipulator` object to control the robot. The scene must be previously build.
        """
        if not self._is_build():
            raise RuntimeError(
                "The access to the robot can not be given before calling `build()` on the scene!"
            )
        return self._get_manipulator()

    @abstractmethod
    def _get_manipulator(self) -> BaseManipulator: ...

    @abstractmethod
    def read_state(self) -> None:
        """Update the internatl state with data from the scene."""
        ...
