from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import auto
from typing import Any

import torch as th
from strenum import StrEnum

from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import Control, RobotCfg
from aegis_gym.envs.plotjuggler_udp import PlotJugglerUDP

from ..manipulator.base_manipulator import BaseManipulator
from ..objects.base_objects import ObjectProperties, ObjectType


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

    PJ_HOST = "127.0.0.1"
    PJ_PORT = 9870
    PJ_JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )

    def __init__(self, device: th.device, enable_plotjuggler: bool = False):
        super().__init__()
        self._is_build = False
        self.device: th.device = device
        self._pj: PlotJugglerUDP | None = None
        if enable_plotjuggler:
            self._pj = PlotJugglerUDP(host=self.PJ_HOST, port=self.PJ_PORT)
            get_logger(type(self).__name__).info(
                f"Enabled UDP server for PlotJuggler at {self.PJ_HOST}:{self.PJ_PORT}"
            )

    def shutdown(self) -> None:
        """Shutdown the scene connection."""
        if self._pj is not None:
            self._pj.close()
            self._pj = None
        self._shutdown()

    @abstractmethod
    def _shutdown(self) -> None: ...

    def randomize_domain(
        self, rand_type: RandomizationType, env_idx: th.Tensor
    ) -> None:
        """Calls domain randomization function."""
        if rand_type not in self._randomization_fns:
            raise IndexError(
                f"The `{rand_type}` randomization is not defined for `{type(self).__name__}` scene."
                "TIP: Check available randomizations via `get_available_randomizations()`."
            )
        self._randomization_fns[rand_type](env_idx)

    def get_available_randomizations(self) -> frozenset[RandomizationType]:
        """Returns set of available randomizations types."""
        return frozenset(self._randomization_fns)

    @abstractmethod
    def get_policy_dt(self) -> float:
        """Get the value of the policy dt."""
        ...

    @abstractmethod
    def add_entity(self, entity: ObjectType, properties: ObjectProperties) -> Any:
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

    def update_state(self) -> None:
        """Update the internal state with data from the scene."""
        self._update_state()
        self._log_state_to_plot_juggler()

    @abstractmethod
    def _update_state(self) -> None: ...

    @abstractmethod
    def pre_step(self) -> None:
        """Prepare the scene for a new action."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Process the scene step."""
        ...

    def _log_state_to_plot_juggler(self) -> None:
        if self._pj is None or not self._is_build:
            return
        data = self._collect_pj_data()
        data.update(self._collect_pj_extra_data())
        self._pj.send(data)

    def _collect_pj_data(self) -> dict[str, float]:
        manip = self._get_manipulator()
        n = len(self.PJ_JOINT_NAMES)
        pos = manip.get_joints_positions()[0, :n].tolist()
        vel = manip.get_joints_velocities()[0, :n].tolist()
        eff = manip.get_joints_efforts()[0, :n].tolist()

        data: dict[str, float] = {}
        for i, name in enumerate(self.PJ_JOINT_NAMES):
            data[f"joint_states/{name}/position"] = pos[i]
            data[f"joint_states/{name}/velocity"] = vel[i]
            data[f"joint_states/{name}/effort"] = eff[i]

        for axis, v in zip("xyz", manip.get_tcp_position()[0].tolist()):
            data[f"ee/position/{axis}"] = v
        # TODO(issue#55) Enable orientation logging

        wrench = manip.get_ft_wrench()[0].tolist()
        # Gravity removed, so it is comparable with a tared sensor on the robot rather
        # than carrying the tool's own 13.3 N.
        comp = manip.get_ft_wrench_compensated()[0].tolist()
        for i, axis in enumerate("xyz"):
            data[f"ft_sensor/force/{axis}"] = wrench[i]
            data[f"ft_sensor/torque/{axis}"] = wrench[i + 3]
            data[f"ft_sensor/compensated/force/{axis}"] = comp[i]
            data[f"ft_sensor/compensated/torque/{axis}"] = comp[i + 3]
        return data

    def _collect_pj_extra_data(self) -> dict[str, float]:
        return {}
