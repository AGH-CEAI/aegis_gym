from abc import ABC, abstractmethod
from enum import auto
from typing import Callable, Any

import torch as th
from strenum import StrEnum


from envs.objects import ObjectType
from config.types import Control, RobotCfg
from ..manipulator.base_manipulator import BaseManipulator


class RandomizationType(StrEnum):
    MANIPULATOR_PD_GAINS = auto()
    MANIPULATOR_MAX_SPEED = auto()
    CAMERAS_EXTRINSICS = auto()
    SCENE_LIGHTING = auto()


class BaseScene(ABC):
    """
    Base class for implementing whole interaction with simulator or real world.
    """

    CONTROL_TYPE: Control
    _randomization_fns: dict[RandomizationType, Callable[[th.Tensor], None]]

    def __init__(self, device: th.device = th.device("cpu")):
        super().__init__()
        self._is_build = False
        self.device: th.device = device

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the scene connection."""
        ...

    @property
    def available_randomizations(self) -> frozenset[RandomizationType]:
        """Returns set of available randomizations types"""
        return frozenset(self._randomization_fns)

    @abstractmethod
    def get_policy_dt(self) -> float:
        """Get the value of the policy dt."""
        ...

    @abstractmethod
    def add_entity(self, entity: ObjectType) -> Any:
        """Add a given entity to the scene."""
        ...

    @abstractmethod
    def add_manipulator(self, cfg: RobotCfg) -> None:
        """Add the Aegis robot to the scene."""
        ...

    def build(self) -> None:
        """
        Build the scene, based on previously added entities (`add_entity()`) and robot (`add_robot()`).
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
        if not self._is_build:
            raise RuntimeError(
                "The access to the robot can not be given before calling `build()` on the scene!"
            )
        return self._get_manipulator()

    @abstractmethod
    def _get_manipulator(self) -> BaseManipulator: ...

    @abstractmethod
    def update_state(self) -> None:
        """Update the internal state with data from the scene."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Process the scene step (mainly for simulations)."""
        ...
