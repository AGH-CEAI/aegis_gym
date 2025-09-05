import warnings
from enum import auto

# TODO for Python 3.11: change to build in StrEnum
from strenum import StrEnum

from .robot_commander_interface import BaseROSInterface
from .robot_commander_mock import ROSInterfaceMock

try:
    from .robot_commander import ROSInterface
except ImportError:
    ROSInterface = None


class ROSInterfaceType(StrEnum):
    MOCK = auto()
    REAL = auto()


def get_ros_interface(
    mode: ROSInterfaceType = ROSInterfaceType.REAL,
) -> BaseROSInterface:
    match mode:
        case ROSInterfaceType.MOCK:
            return ROSInterfaceMock()
        case ROSInterfaceType.REAL:
            if ROSInterface:
                return ROSInterface()
            warnings.warn(
                "\nFailed to import ROSInterface. Double check if the ROS is sourced properly via `source /opt/ros/ROS_DISTRO/setup.sh`."
            )
            raise ImportError
        case _:
            print(f"Not defined ROSInterfaceType '{mode.name}'.")
            raise ValueError
