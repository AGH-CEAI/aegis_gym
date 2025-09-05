import warnings
from enum import auto

# Python 3.11+: Change to builtin StrEnum
from strenum import StrEnum

from .robot_commander_interface import RobotCommanderInterface
from .robot_commander_mock import RobotCommanderMock

try:
    from .robot_commander import RobotCommander
except ImportError:
    RobotCommander = None


class RobotCommanderType(StrEnum):
    MOCK = auto()
    REAL = auto()


def get_ros_interface(
    mode: RobotCommanderType = RobotCommanderType.REAL,
) -> RobotCommanderInterface:
    match mode:
        case RobotCommanderType.MOCK:
            return RobotCommanderMock()
        case RobotCommanderType.REAL:
            if RobotCommander:
                return RobotCommander()
            warnings.warn(
                "\nFailed to import ROSInterface. Double check if the ROS is sourced properly via `source /opt/ros/ROS_DISTRO/setup.sh`."
            )
            raise ImportError
        case _:
            print(f"Not defined ROSInterfaceType '{mode.name}'.")
            raise ValueError
