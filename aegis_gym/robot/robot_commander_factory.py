import warnings
from enum import auto

# Python 3.11+: Change to builtin StrEnum
from strenum import StrEnum

from .robot_commander_interface import RobotCommanderInterface
from .robot_commander_mock import RobotCommanderMock


class RobotCommanderType(StrEnum):
    MOCK = auto()
    ROS = auto()
    SIM_GENESIS = "SimGenesis"


def get_robot_commander(
    mode: RobotCommanderType = RobotCommanderType.ROS,
) -> RobotCommanderInterface:
    match mode:
        case RobotCommanderType.MOCK:
            return RobotCommanderMock()

        case RobotCommanderType.ROS:
            try:
                from .robot_commander_ros import RobotCommanderROS
            except ImportError:
                RobotCommanderROS = None

            if RobotCommanderROS:
                return RobotCommanderROS()
            warnings.warn(
                "\nFailed to import RobotCommanderROS. Double check if the ROS is sourced properly via `source /opt/ros/ROS_DISTRO/setup.sh`."
            )
            raise ImportError

        case RobotCommanderType.SIM_GENESIS:
            try:
                from ..sim.genesis.robots_commander_genesis import (
                    RobotCommanderSimGenesis,
                )
            except ImportError:
                RobotCommanderSimGenesis = None

            if RobotCommanderSimGenesis:
                return RobotCommanderSimGenesis()
            warnings.warn(
                "\nFailed to import RobotCommanderSimGenesis. Double check if the 'aegis_gym' is instatalled with optional dependencies: 'pip3 install ./aegis_gym.whl[sim-genesis]'."
            )
            raise ImportError

        case _:
            print(f"Not defined RobotCommanderInterface '{mode.name}'.")
            raise ValueError
